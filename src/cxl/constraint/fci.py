import numpy as np
from numpy.typing import NDArray
from cxl.config import HardwareConfig
from .skeleton import create_skeleton_learner
from cxl.graph.graph_utils import (
    remark_edges,
    neighbors,
    is_adjacent,
    find_unshielded_triples,
    parents_of,
    get_pds,
    remove_edge,
    orient_edge,
    is_adjacent,
    is_parent,
    is_collider,
    discriminating_path,
)
from cxl.independence.fisher import FisherZTest
import itertools

from .common import SeparationSetMapping


class FCILearner:
    """
    Class for learning the structure of a causal graph using the FCI algorithm.

    Attributes:
        alpha (float): The significance level for conditional independence tests.
        hardware_config (HardwareConfig | None): Hardware configuration for computation.
        max_depth (int | None): Maximum depth for the skeleton learning phase.
        selection_bias (bool): Flag indicating whether to apply selection bias correction.
        indep_tester_factory: Factory for creating independence testers.
        verbose (bool): Flag indicating whether to display verbose output.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        hardware_config: HardwareConfig | None = None,
        max_depth: int | None = None,
        selection_bias=False,
        indep_tester_factory=FisherZTest,
        verbose=False,
    ) -> None:
        """
        Initialize the FCILearner.

        Args:
            alpha (float): The significance level for conditional independence tests.
            hardware_config (HardwareConfig | None): Hardware configuration for computation.
            max_depth (int | None): Maximum depth for the skeleton learning phase.
            selection_bias (bool): Flag indicating whether to apply selection bias correction.
            indep_tester_factory: Factory for creating independence testers.
            verbose (bool): Flag indicating whether to display verbose output.
        """
        self.skeleton_learner = create_skeleton_learner(
            alpha, max_depth, hardware_config, indep_tester_factory, verbose
        )
        self.indep_tester_factory = indep_tester_factory
        self.alpha = alpha
        self.verbose = verbose
        self.selection_bias = selection_bias

    def fit(self, observations: NDArray) -> NDArray:
        """
        Fit the model to the observations and return the causal graph.

        Args:
            observations (NDArray): The observational data.

        Returns:
            NDArray: The learned causal graph.
        """
        indep_tester = self.indep_tester_factory(observations, self.alpha)
        skeleton, separation_sets, unshielded_triples = self.skeleton_learner.learn(
            observations
        )
        remark_edges(skeleton, 2)
        self._orient_v_structures(skeleton, unshielded_triples, separation_sets)
        graph, separation_sets, unshielded_triples = self._finalise_skeleton(
            skeleton, separation_sets, indep_tester
        )
        self._orient_v_structures(graph, unshielded_triples, separation_sets)
        rules = [r1, r2, r3, r4]
        modified = True
        while modified:
            modified = any(rule(graph, sepset=separation_sets) for rule in rules)
        if self.selection_bias:
            rules = [r5, r6, r7]
            while modified:
                modified = any(rule(graph) for rule in rules)
        modified = True

        return graph

    def _orient_v_structures(
        self,
        graph: NDArray,
        unshielded_triples: list[int],
        separation_sets: SeparationSetMapping,
    ):
        """
        Orient the V-structures in the graph.

        Args:
            graph (NDArray): The causal graph.
            unshielded_triples (list[int]): List of unshielded triples in the graph.
            separation_sets: Separation sets for the graph.
        """
        for x, y, z in unshielded_triples:
            if y not in separation_sets[(x, z)]:
                # orient  x -> y <- z
                orient_edge(graph, x, y, -1)
                orient_edge(graph, z, y, -1)

    def _finalise_skeleton(
        self, graph: NDArray, separation_sets: SeparationSetMapping, indep_tester
    ) -> tuple[NDArray, SeparationSetMapping, list[tuple[int, int, int]]]:
        """
        Finalize the skeleton learning.

        Args:
            graph (NDArray): The causal graph.
            separation_sets: Separation sets for the graph.
            indep_tester: Independence tester.

        Returns:
            tuple: Finalized causal graph, separation sets, and unshielded triples.
        """
        n = len(graph)
        # might need to deepcopy seperation set we will see if we need it later
        pds = {}
        for xi in range(n):
            pds[xi] = get_pds(graph, xi)
            for xj in neighbors(graph, xi):
                l = -1
                # get all subsets might be a better idea maybe
                while not is_adjacent(graph, xi, xj) and l <= len(pds[xi] - {xj}):
                    l += 1
                    for y in itertools.combinations(pds[xi], l):
                        if indep_tester.is_conditionally_independent(xi, xj, y):
                            remove_edge(graph, xi, xj)
                            Y = set(y)
                            separation_sets[(xi, xj)] = Y
                            separation_sets[(xj, xi)] = Y
                            break
        remark_edges(graph, 2)
        unshielded_triples = list(find_unshielded_triples(graph, 2))
        return (graph, separation_sets, unshielded_triples)


def r1(graph: NDArray, *args, **kwargs) -> bool:
    """
    Meek Rule 1: Orientation rule for causal graph skeleton learning.

    Args:
        graph (NDArray): The adjacency matrix representing the causal graph.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if any modifications were made to the graph, False otherwise.
    """
    n = len(graph)
    modified = False
    for alpha, beta, gamma in itertools.permutations(range(n), 3):
        if (
            graph[alpha, beta] == 1
            and graph[gamma, beta] == 2
            and not is_adjacent(graph, alpha, gamma)
        ):

            # orient edge
            orient_edge(graph, beta, gamma, tail_marker=-1)
            modified = True
    return modified


def r2(graph: NDArray, *args, **kwargs) -> bool:
    """
    Meek Rule 2: Orientation rule for causal graph skeleton learning.

    Args:
        graph (NDArray): The adjacency matrix representing the causal graph.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if any modifications were made to the graph, False otherwise.
    """
    n = len(graph)
    modified = False
    for alpha, beta, gamma in itertools.permutations(range(n), 3):
        if (
            ((is_parent(graph, alpha, beta) and graph[beta, gamma] == 1))
            or (
                graph[alpha, beta] == 1
                and is_parent(graph, beta, gamma, tail_marker=-1)
            )
        ) and (graph[alpha, gamma] == 2):
            graph[alpha, gamma] = 1
            modified = True
    return modified


def r3(graph: NDArray, *args, **kwargs) -> bool:
    """
    Meek Rule 3: Orientation rule for causal graph skeleton learning.

    Args:
        graph (NDArray): The adjacency matrix representing the causal graph.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if any modifications were made to the graph, False otherwise.
    """
    n = len(graph)
    modified = False
    for alpha, beta, theta, gamma in itertools.permutations(range(n), 4):
        if (
            is_collider(graph, alpha, beta, gamma)
            and _is_maybe_collider(graph, alpha, theta, gamma)
            and graph[theta, beta] == 2
        ):
            graph[theta, beta] = 1
            modified = True
    return modified


def r4(graph: NDArray, *args, **kwargs) -> bool:
    """
    Meek Rule 4: Orientation rule for causal graph skeleton learning.

    Args:
        graph (NDArray): The adjacency matrix representing the causal graph.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if any modifications were made to the graph, False otherwise.
    """
    sepset = kwargs["sepset"]
    n = len(graph)
    modified = False
    for nb in range(n):
        c_nodes = np.where(graph[:, nb] == 2)[0].reshape((-1))
        for nc in c_nodes:
            for na in parents_of(graph, nc, tail_marker=-1):
                if graph[nb, na] == 1:
                    nd = discriminating_path(graph, na, nb, nc)
                    if nd is not None:
                        if nb in sepset[(nd, nc)]:
                            # b -> c
                            orient_edge(graph, nb, nc, -1)
                        else:
                            # a <-> b <-> c
                            orient_edge(graph, na, nb, 1)
                            orient_edge(graph, nb, nc, 1)
                        modified = True
    return modified


def _is_maybe_collider(graph, alpha, theta, gamma):
    return (
        graph[alpha, theta] == 2
        and graph[gamma, theta] == 2
        and graph[alpha, gamma] == 0
        and graph[gamma, theta] == 0
    )


def r5(graph: NDArray) -> bool:
    """
    Meek Rule 5: Orientation rule for causal graph skeleton learning.

    Args:
        graph (NDArray): The adjacency matrix representing the causal graph.

    Returns:
        bool: True if any modifications were made to the graph, False otherwise.
    """
    n = len(graph)
    idxs = np.where((graph.T == 2) & (graph == 2))
    symmetric_edges = list(zip(idxs[0], idxs[1]))
    modified = False
    for alpha, beta in symmetric_edges:
        alpha_neighbours = {
            node
            for node in neighbors(graph, alpha)
            if graph[alpha, node] == 2
            and graph[node, alpha] == 2
            and not is_adjacent(graph, node, beta)
        }
        beta_neighbours = {
            node
            for node in neighbors(graph, beta)
            if graph[beta, node] == 2
            and graph[node, beta] == 2
            and not is_adjacent(graph, node, alpha)
        }

        for n_alpha in alpha_neighbours:
            for n_beta in beta_neighbours:
                path = uncover_path(graph, alpha, beta, n_alpha, n_beta)
                if path is not None:
                    graph[alpha, beta] = -1
                    graph[beta, alpha] = -1

                    for u, v in zip(path, path[1:]):
                        graph[u, v] = -1
                        graph[v, u] = -1
                    modified = True
    return modified


def r6(graph: NDArray) -> tuple[NDArray, bool]:
    """
    Meek Rule 6: Orientation rule for causal graph skeleton learning.

    Args:
        graph (NDArray): The adjacency matrix representing the causal graph.

    Returns:
        bool: True if any modifications were made to the graph, False otherwise.
    """
    n = len(graph)
    modified = False
    for alpha, beta, gamma in itertools.permutations(n, 3):
        if (
            graph[alpha, beta] == -1
            and graph[beta, alpha] == -1
            and graph[gamma, beta] == 2
        ):
            graph[gamma, beta] = -1
            modified = True
    return modified


def r7(graph: NDArray) -> bool:
    """
    Meek Rule 7: Orientation rule for causal graph skeleton learning.

    Args:
        graph (NDArray): The adjacency matrix representing the causal graph.

    Returns:
        bool: True if any modifications were made to the graph, False otherwise.
    """
    n = len(graph)
    modified = False
    for alpha, beta, gamma in itertools.permutations(n, 3):
        if (
            graph[alpha, beta] == 2
            and graph[beta, alpha] == -1
            and graph[gamma, beta] == 2
            and not is_adjacent(graph, alpha, gamma)
        ):
            graph[gamma, beta] = -1
            modified = True
    return modified


def uncover_path(graph, x: int, y: int, nx: int, ny: int) -> list[int] | None:
    if (
        graph[nx, ny] == 2
        and graph[ny, nx] == 2
        and not is_adjacent(graph, nx, y)
        and not is_adjacent(graph, x, ny)
    ):
        return [ny, ny]
    x_neighbours = neighbors(graph, nx)
    c_nodes = {
        n
        for n in x_neighbours
        if graph[x, nx] == 2 and graph[nx, x] == 2 and not is_adjacent(graph, x, nx)
    }

    for node_c in c_nodes:
        path = uncover_path(nx, y, node_c, ny)
        if path is not None:
            return [nx, *path]
    return None
