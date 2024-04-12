from numpy.typing import NDArray
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(
        self, n_in, n_xdims, n_hid, n_out, graph, batch_size, do_prob=0.0, factor=True
    ):
        super(Encoder, self).__init__()

        self.A = nn.Parameter(
            Variable(torch.from_numpy(graph).double(), requires_grad=True)
        )
        self.factor = factor
        self.W = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.f1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.f2 = nn.Linear(n_hid, n_out, bias=True)
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(0.1))
        self.z_positive = nn.Parameter(
            torch.ones_like(torch.from_numpy(graph)).double()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, sample, rel_rec, rel_send):

        new_A = torch.sinh(3 * self.A)
        lhs = torch.eye(new_A.shape[0]).double() - new_A.transpose(0, 1)
        A = torch.eye(new_A.size()[0]).double()
        h1 = F.relu(self.f1(sample))
        x = self.f2(h1)
        logits = lhs @ (x + self.W) - self.W
        return x, logits, new_A, A, self.z, self.z_positive, self.A, self.W


class Decoder(nn.Module):

    def __init__(
        self,
        n_in_node,
        n_in_z,
        n_out,
        encoder,
        data_variable_size,
        batch_size,
        n_hid,
        do_prob=0.0,
    ):
        # super().__init__()
        super(Decoder, self).__init__()

        self.f1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.f2 = nn.Linear(n_hid, n_out, bias=True)
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        self.dropout_prob = do_prob
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # for m in self.modules():
        #     nn.init.xavier_normal(m.weight.data)

    def forward(
        self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa
    ):

        # adj_A_new1 = (I-A^T)^(-1)
        lhs = torch.inverse(
            torch.eye(origin_A.shape[0]).double() - origin_A.transpose(0, 1)
        )

        mat_z = torch.matmul(lhs, input_z + Wa) - Wa

        H3 = F.relu(self.f1((mat_z)))
        out = self.f2(H3)
        return mat_z, out, adj_A_tilt


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def matrix_poly(matrix, d):
    x = torch.eye(d).double() + torch.div(matrix, d)
    return torch.matrix_power(x, d)


def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(
        torch.pow(mean1 - mean2, 2), 2.0 * np.exp(2.0 * variance)
    )
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0))) * 0.5


from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


class CGNNLearner:

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def fit(self, data) -> NDArray:

        batch_size = 100
        feat_train = torch.FloatTensor(data)
        train_data = TensorDataset(feat_train, feat_train)
        train_loader = DataLoader(train_data, batch_size=batch_size)

        gamma = 0.5
        lambda_A = 0.0
        # lambda_A = 0.6
        lr = 0.1
        tau_A = 0.0
        factor = True
        C_A = 0.5
        _, n = data.shape
        x_dims = 1
        z_dims = 1
        encoder_dropout = 0.0
        decoder_dropout = 0.0
        data_variable_size = n
        graph = np.zeros((n, n))
        encoder_hidden = 64
        graph_threshold = 0.1
        decoder_hidden = 64
        encoder = Encoder(
            data_variable_size * x_dims,
            x_dims,
            encoder_hidden,
            int(z_dims),
            graph,
            batch_size=batch_size,
            do_prob=encoder_dropout,
            factor=factor,
        ).double()
        decoder = Decoder(
            data_variable_size * x_dims,
            z_dims,
            x_dims,
            encoder,
            data_variable_size=data_variable_size,
            batch_size=batch_size,
            n_hid=decoder_hidden,
            do_prob=decoder_dropout,
        ).double()
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=0.1
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=0.3, gamma=gamma
        )
        off_diag = np.ones([n, n]) - np.eye(n)

        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        rel_rec = torch.DoubleTensor(rel_rec)
        rel_send = torch.DoubleTensor(rel_send)
        rel_rec = Variable(rel_rec)
        rel_send = Variable(rel_send)

        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        rel_rec = Variable(rel_rec)
        rel_send = Variable(rel_send)

        def acyclicity_constraint(A, m):
            expm_A = matrix_poly(A * A, m)
            h_A = torch.trace(expm_A) - m
            return h_A

        prox_plus = torch.nn.Threshold(0.0, 0.0)

        def stau(w, tau):
            w1 = prox_plus(torch.abs(w) - tau)
            return torch.sign(w) * w1

        encoder.train()
        decoder.train()
        scheduler.step()
        for idx, (minidata, relations) in (
            t := tqdm(enumerate(train_loader), disable=not self.verbose)
        ):
            data = minidata.reshape((100, 5, 1))
            t.set_description(f"processing minibatch {idx}")
            data, relations = Variable(data).double(), Variable(relations).double()
            relations = relations.unsqueeze(2)
            optimizer.zero_grad()
            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = (
                encoder(data, rel_rec, rel_send)
            )  # logits is of size: [num_sims, z_dims]
            edges = logits
            dec_x, output, adj_A_tilt_decoder = decoder(
                data,
                edges,
                data_variable_size * x_dims,
                rel_rec,
                rel_send,
                origin_A,
                adj_A_tilt_encoder,
                Wa,
            )
            target = data
            preds = output
            variance = 0.0

            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = tau_A * torch.sum(torch.abs(one_adj_A))
            h_A = acyclicity_constraint(origin_A, data_variable_size)
            loss += (
                lambda_A * h_A
                + 0.5 * C_A * h_A * h_A
                + 100.0 * torch.trace(origin_A * origin_A)
                + sparse_loss
            )
            # +  0.01 * torch.sum(variance * variance)
            loss.backward()
            loss = optimizer.step()

            myA.data = stau(myA.data, tau_A * lr)

            graph = origin_A.data.clone().numpy()
            graph[np.abs(graph) < graph_threshold] = 0
        return graph
