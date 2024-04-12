import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import itertools
import networkx as nx
from sklearn.preprocessing import scale
from cxl.constraint.skeleton import create_skeleton_learner
from numpy.typing import NDArray
from copy import deepcopy
from cxl.graph.graph_utils import orient_edge
from cxl.config import HardwareConfig
from tqdm import tqdm


class MMDloss(torch.nn.Module):

    def __init__(self, input_size, bandwidths=None):
        """Init the model."""
        super(MMDloss, self).__init__()
        if bandwidths is None:
            bandwidths = torch.Tensor([0.01, 0.1, 1, 10, 100])
        else:
            bandwidths = bandwidths
        s = torch.cat(
            [
                torch.ones([input_size, 1]) / input_size,
                torch.ones([input_size, 1]) / -input_size,
            ],
            0,
        )

        self.register_buffer("bandwidths", bandwidths.unsqueeze(0).unsqueeze(0))
        self.register_buffer("S", (s @ s.t()))

    def forward(self, x, y):
        X = torch.cat([x, y], 0)
        XX = X @ X.t()
        X2 = (X * X).sum(dim=1).unsqueeze(0)
        exponent = -2 * XX + X2.expand_as(XX) + X2.t().expand_as(XX)
        b = (
            exponent.unsqueeze(2).expand(-1, -1, self.bandwidths.shape[2])
            * -self.bandwidths
        )
        lossMMD = torch.sum(self.S.unsqueeze(2) * b.exp())
        return lossMMD


class PairwiseCGNN(nn.Module):

    def __init__(self, batch_size, nh=5) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(2, nh)
        self.l2 = torch.nn.Linear(nh, 1)
        self.register_buffer("noise", torch.Tensor(batch_size, 1))
        self.act = torch.nn.ReLU()
        self.criterion = MMDloss(batch_size)
        self.layers = torch.nn.Sequential(self.l1, self.act, self.l2)
        self.nb_epochs = 10
        self.nb_test = 10
        self.batch_size = 100
        self.dataloader_workers = 1
        self.lr = 0.01

    def forward(self, x):
        self.noise.normal_()
        return self.layers(torch.cat([x, self.noise], 1))

    def train(self, data):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        total_loss = 0
        dataloader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.dataloader_workers,
        )

        for epoch in range(self.nb_epochs + self.nb_test):
            print(epoch)
            for idx, (x, y) in enumerate(dataloader):

                optimizer.zero_grad()
                prediction = self.forward(x)
                loss = self.criterion(
                    torch.cat([x, prediction], 1), torch.cat([x, y], 1)
                )

                if epoch < self.nb_epochs:
                    loss.backward()
                    optimizer.step()
                else:
                    total_loss += loss.data

        score = total_loss.cpu().numpy() / self.nb_test
        return score

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


def score_cause_effect(data, x, y, batch_size, nh):

    xy_net = PairwiseCGNN(batch_size, nh)
    yx_net = PairwiseCGNN(batch_size, nh)
    xy_net.reset_parameters()
    yx_net.reset_parameters()

    X = torch.Tensor(scale(data[:, x])).view(-1, 1)
    Y = torch.Tensor(scale(data[:, y])).view(-1, 1)
    xy_score = xy_net.train(TensorDataset(X, Y))
    yx_score = yx_net.train(TensorDataset(Y, X))

    return (xy_score, yx_score)


class CGNN_block(nn.Module):
    """CGNN 'block' which represents a FCM equation between a cause and its parents."""

    def __init__(self, sizes):
        """Init the block with the network sizes."""
        super(CGNN_block, self).__init__()
        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward through the network."""
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class CGNN(nn.Module):

    def __init__(
        self,
        adj_matrix,
        batch_size,
        nh=10,
        confounding=False,
        initial_graph=None,
        **kwargs,
    ):
        super().__init__()
        self.topological_order = [
            i for i in nx.topological_sort(nx.DiGraph(adj_matrix))
        ]
        self.adjacency_matrix = adj_matrix
        self.confounding = confounding
        if initial_graph is None:
            self.i_adj_matrix = self.adjacency_matrix
        else:
            self.i_adj_matrix = initial_graph
        self.blocks = torch.nn.ModuleList()
        self.generated = [None for i in range(self.adjacency_matrix.shape[0])]
        self.register_buffer(
            "noise", torch.zeros((batch_size, self.adjacency_matrix.shape[0]))
        )
        self.criterion = MMDloss(batch_size)
        self.register_buffer("score", torch.FloatTensor([0]))
        self.batch_size = batch_size

        for i in range(self.adjacency_matrix.shape[0]):
            self.blocks.append(
                CGNN_block([int(self.adjacency_matrix[:, i].sum()) + 1, nh, 1])
            )

    def run(self, dataset, nb_epochs, test_epochs, lr, dataloader_workers):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=dataloader_workers,
        )
        total_loss = 0
        for epoch in range(nb_epochs + test_epochs):
            for i, obs in enumerate(dataloader):
                optimizer.zero_grad()
                samples = self.forward()
                # print(obs[0])
                # exit()
                mmd_loss = self.criterion(samples, obs[0])
                mmd_loss.backward()
                optimizer.step()
                if epoch > nb_epochs:
                    # gonna need to modif it to make it addable on gpus
                    total_loss += mmd_loss.data
        return total_loss

    def forward(self):
        self.noise.data.normal_()
        if not self.confounding:
            for i in self.topological_order:
                self.generated[i] = self.blocks[i](
                    torch.cat(
                        [
                            v
                            for c in [
                                [
                                    self.generated[j]
                                    for j in np.nonzero(self.adjacency_matrix[:, i])[0]
                                ],
                                [self.noise[:, [i]]],
                            ]
                            for v in c
                        ],
                        1,
                    )
                )
        else:
            for i in self.topological_order:
                self.generated[i] = self.blocks[i](
                    torch.cat(
                        [
                            v
                            for c in [
                                [
                                    self.generated[j]
                                    for j in np.nonzero(self.adjacency_matrix[:, i])[0]
                                ],
                                [
                                    self.corr_noise[min(i, j), max(i, j)]
                                    for j in np.nonzero(self.i_adj_matrix[:, i])[0]
                                ][self.noise[:, [i]]],
                            ]
                            for v in c
                        ],
                        1,
                    )
                )
        return torch.cat(self.generated, 1)

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()


def cgnn_score(data, graph, batch_size):
    obs = TensorDataset(torch.Tensor(data))
    net = CGNN(graph, batch_size)
    net.reset_parameters()
    return net.run(obs, 100, 100, 0.05, 1)


def hill_climb(graph, data, batch_size):
    print("starting hill climb")
    _, n = graph.shape
    curr_graph = graph
    curr_score = cgnn_score(data, graph, batch_size)

    visited = {nx.from_numpy_array(graph, create_using=nx.DiGraph)}
    while True:
        improvable = False
        for i, j in itertools.permutations(range(n), 2):
            if curr_graph[i, j] == 1:
                neighbour = deepcopy(curr_graph)
                orient_edge(neighbour, j, i)
                tadjmat = nx.from_numpy_array(neighbour, create_using=nx.DiGraph)
                if nx.is_directed_acyclic_graph(tadjmat) and tadjmat not in visited:
                    visited.add(tadjmat)
                    score = cgnn_score(data, neighbour, batch_size)
                    if score > curr_score:
                        curr_graph = neighbour
                        curr_score = score
                        improvable = True
                        break
        if not improvable:
            break
    return curr_graph


class GNNLearner:
    def __init__(
        self,
        hardware_config: HardwareConfig = None,
        batch_size: int = 100,
        nh: int = 10,
        verbose=False,
    ) -> None:
        self.skeleton_learner = create_skeleton_learner(0.05, None, hardware_config)
        self.batch_size = batch_size
        self.nh = nh
        self.verbose = verbose

    def fit(self, observations: NDArray) -> None:
        _, n = observations.shape
        graph, *_ = self.skeleton_learner.learn(observations)
        for x, y in (
            t := tqdm(itertools.combinations(range(n), 2), disable=not self.verbose)
        ):
            t.set_description(f"processing edge {x} - {y}")
            if graph[x, y] == 1:
                print(f"edge: {x} - {y}")
                xy_score, yx_score = score_cause_effect(
                    observations, x, y, self.batch_size, nh=self.nh
                )
                if xy_score < yx_score:
                    graph[y, x] = 0
                else:
                    graph[x, y] = 0
        graph = hill_climb(graph, observations, self.batch_size)
        return graph
