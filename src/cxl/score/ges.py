from numpy.typing import NDArray
import numpy as np
import itertools
from cxl.graph.graph_utils import (
    is_adjacent,
    get_neighbours,
    remove_edge,
    parents_of,
    children_of,
    orient_edge,
    na,
    neighbors,
    semi_neighbours,
    adj,
    create_empty_graph,
    topological_ordering,
    is_clique,
)
from typing import Generator, Callable


LocalScoreFunction = Callable[[np.ndarray, int, list[int]], float]


class GESLearner:
    """
    The Greedy Equivalent Search (GES) algorithm for learning the structure of Bayesian networks.

    Attributes:
        None

    Methods:
        fit: Fit the GES algorithm to the given observations and return the learned causal graph.

    Helper Functions:
        _fes: Forward Edge Search algorithm for GES.
        _bes: Backward Edge Search algorithm for GES.
    """

    def __init__(self) -> None:
        """
        Initialize the GESLearner.

        Args:
            None

        Returns:
            None
        """
        pass

    def fit(self, observations: NDArray) -> NDArray:
        """
        Fit the GES algorithm to the given observations and return the learned causal graph.

        Args:
            observations (NDArray): The observations used for learning.

        Returns:
            NDArray: The learned causal graph.
        """
        _, n = observations.shape
        graph = create_empty_graph(n)
        graph = _fes(graph, observations)
        graph = _bes(graph, observations)
        return graph


def _fes(graph: NDArray, observations: NDArray) -> NDArray:
    """
    Forward Edge Search algorithm for GES.

    Args:
        graph (NDArray): The current causal graph.
        observations (NDArray): The observations used for learning.

    Returns:
        NDArray: The updated causal graph after applying Forward Edge Search.
    """
    changed = True
    while changed:
        edge, T = _fs(observations, graph)
        if edge is None:
            changed = False
        else:
            graph = _apply_insert(graph, *edge, T)
            graph = _pdagtocpdag(graph)
    return graph


def _bes(graph: NDArray, observations: NDArray) -> NDArray:
    """
    Backward Edge Search algorithm for GES.

    Args:
        graph (NDArray): The current causal graph.
        observations (NDArray): The observations used for learning.

    Returns:
        NDArray: The updated causal graph after applying Backward Edge Search.
    """
    changed = True
    while changed:
        edge, H = _bs(graph, observations)
        if edge is None:
            changed = False
        else:
            graph = _apply_delete(*edge, H, graph)
            graph = _pdagtocpdag(graph)
    return graph


def _fs(data: NDArray, graph: NDArray) -> tuple[tuple[int, int], set[int]]:
    """
    Forward Edge Search algorithm for GES.

    Args:
        data (NDArray): The observations used for learning.
        graph (NDArray): The current causal graph.

    Returns:
        tuple[tuple[int, int], set[int]]: A tuple containing the edge (x, y) and the set of parents (T) of x
        such that adding x -> y with parents T improves the score of the graph.
    """
    n = len(graph)
    edge = None
    subset = None
    min_score = 1e7
    for x, y in itertools.permutations(range(n), 2):
        if not is_adjacent(graph, x, y):
            T0 = _build_T0(x, y, graph)
            subsets = _get_subsets(T0)
            valid_subsets = [T for T in subsets if _test_insert(graph, x, y, T)]
            for T in valid_subsets:
                chscore = _insert(x, y, T, graph, data)
                if chscore < min_score:
                    edge = (x, y)
                    min_score = chscore
                    subset = T
    if min_score > 0:
        return (None, None)
    return (edge, subset)


def _build_T0(x: int, y: int, graph: NDArray) -> set[int]:
    neighbours_y = get_neighbours(graph, y)
    adjacent_x = adj(graph, x)
    T0 = neighbours_y - adjacent_x
    return T0


def _get_subsets(s: set[int]) -> Generator[set[int], None, None]:
    yield set()
    for k in range(1, len(s)):
        yield from (set(t) for t in itertools.combinations(s, k))


def _test_insert(graph: NDArray, x: int, y: int, T: set[int]) -> bool:
    NAyx = na(graph, y, x)
    cond1 = is_clique(graph, list(NAyx | T))
    cond2 = _test_cond2(graph, y, x, NAyx | T)
    return cond1 and cond2


def _test_cond2(graph: NDArray, x: int, y: int, condset: set[int]) -> bool:

    def dfs(u: int, visited: set[int], currpath: list[int]) -> int:
        visited.add(u)
        if u == y:
            visited.remove(u)
            return any(x in condset for x in currpath)
        else:
            neighbours = semi_neighbours(graph, u)
            for v in neighbours:
                if v not in visited:
                    res = dfs(v, visited, currpath + [v])
                    if res == False:
                        return False
            visited.remove(u)
            return True

    return dfs(x, set(), list())


def _bic(data: np.ndarray, node: int, parents: list[int]) -> float:
    cov = np.cov(data.T)
    n = data.shape[0]
    lambda_value = 1
    if len(parents) == 0:
        return n * np.log(cov[node, node])
    yX = np.mat(cov[np.ix_([node], parents)])
    XX = np.mat(cov[np.ix_(parents, parents)])
    H = np.log(cov[node, node] - yX * np.linalg.inv(XX) * yX.T)
    return n * H + np.log(n) * len(parents) * lambda_value


def _insert(
    x: int,
    y: int,
    T: set[int],
    graph: NDArray,
    data: NDArray,
    score_fn: LocalScoreFunction = _bic,
) -> float:
    y_parents = parents_of(graph, y)
    NAyx = na(graph, y, x)
    insert_score = score_fn(data, y, list(y_parents | {x} | T | NAyx))
    curr_score = score_fn(data, y, list(y_parents | T | NAyx))
    delta = insert_score - curr_score
    return delta


def _bs(
    graph: NDArray, observations: NDArray
) -> tuple[tuple[int, int] | None, int | None]:
    """
    Backward Edge Search algorithm for GES.

    Args:
        graph (NDArray): The current causal graph.
        observations (NDArray): The observations used for learning.

    Returns:
        tuple[tuple[int, int] | None, int | None]: A tuple containing the edge (x, y) and the set of parents (H)
        of y such that deleting x -> y with parents H improves the score of the graph.
    """
    n = len(graph)
    edge = None
    subset = None
    min_score = 0
    for x, y in itertools.permutations(range(n), 2):
        if not is_adjacent(graph, x, y):
            continue
        H0 = _build_H0(x, y, graph)
        subsets = _get_subsets(H0)
        valid_subsets = (
            H for H in subsets if not _test_delete(graph, x, y, H)
        )  # not sure about this oen need to check notation of when you cross the arrow
        for H in valid_subsets:
            chscore = _delete(x, y, H, graph, observations)
            if chscore < min_score:
                edge = (x, y)
                min_score = chscore
                subset = H
    return (edge, subset)


def _build_H0(x: int, y: int, graph: NDArray) -> set[int]:
    """
    Build the initial set of potential parent nodes for y in backward edge search.

    Args:
        x (int): The potential parent node x.
        y (int): The target node y.
        graph (NDArray): The current causal graph.

    Returns:
        set[int]: The set of potential parent nodes for y.
    """
    neighbours_y = get_neighbours(graph, y)
    adjacent_x = adj(graph, x)
    H0 = neighbours_y & adjacent_x
    return H0


def _test_delete(graph: np.ndarray, x: int, y: int, H: set[int]) -> bool:
    """
    Test if deleting an edge x -> y with parent set H is valid.

    Args:
        graph (np.ndarray): The current causal graph.
        x (int): The parent node x.
        y (int): The child node y.
        H (set[int]): The set of potential parent nodes for y.

    Returns:
        bool: True if deleting x -> y with parents H is valid, False otherwise.
    """
    NAyx = na(graph, y, x)
    return is_clique(graph, list(NAyx - H))


def _apply_insert(graph: np.ndarray, x: int, y: int, T: set[int]) -> NDArray:
    """
    Apply an edge insertion operation to the graph.

    Args:
        graph (np.ndarray): The current causal graph.
        x (int): The parent node x.
        y (int): The child node y.
        T (set[int]): The set of parent nodes to insert.

    Returns:
        NDArray: The updated causal graph after the insertion operation.
    """
    new_graph = graph.copy()

    orient_edge(new_graph, x, y)
    T_indices = list(T)
    # orient edges y -> t
    new_graph[y, T_indices] = 0
    new_graph[T_indices, y] = 1
    return new_graph


def _delete(
    x: int,
    y: int,
    H: set[int],
    graph: NDArray,
    data: NDArray,
    score_fn: LocalScoreFunction,
) -> float:
    """
    Perform the deletion operation in GES.

    Args:
        x (int): The parent node x.
        y (int): The child node y.
        H (set[int]): The set of parent nodes to be deleted.
        graph (NDArray): The current causal graph.
        data (NDArray): The observations used for learning.
        score_fn (LocalScoreFunction): The score function used for evaluating the graph.

    Returns:
        float: The change in score resulting from the deletion operation.
    """
    y_parents = parents_of(graph, y)
    NAyx = na(graph, y, x)
    delete_score = score_fn(data, y, list(y_parents | (NAyx - H) - {x}))
    curr_score = score_fn(data, y, list(y_parents | (NAyx - H)))
    return delete_score - curr_score


def _apply_delete(x: int, y: int, H: set[int], graph: NDArray) -> NDArray:
    """
    Apply an edge deletion operation to the graph.

    Args:
        x (int): The parent node x.
        y (int): The child node y.
        H (set[int]): The set of parent nodes to be deleted.
        graph (NDArray): The current causal graph.

    Returns:
        NDArray: The updated causal graph after the deletion operation.
    """
    new_graph = graph.copy()
    remove_edge(new_graph, x, y)
    H_indices = list(H)
    new_graph[H_indices, x] = 0
    new_graph[H_indices, y] = 0
    return new_graph


def _pdagtocpdag(graph: NDArray) -> NDArray:
    """
    Convert a PDAG (Partially Directed Acyclic Graph) to a CPDAG (Completed Partially Directed Acyclic Graph).

    Args:
        graph (NDArray): The input PDAG.

    Returns:
        NDArray: The resulting CPDAG.
    """
    graph = _pdag_to_dag(graph)
    res = _dag_to_cpdag(graph)
    return res


###########################################################################
# Code below was taken from the ges library: https://github.com/juangamella/ges/blob/master/ges/main.py
###########################################################################


def _pdag_to_dag(P, debug=False):
    G = only_directed(P)
    indexes = list(range(len(P)))
    while P.size > 0:
        found = False
        i = 0
        while not found and i < len(P):
            sink = len(children_of(P, i)) == 0
            n_i = neighbors(P, i)
            adj_i = adj(P, i)
            adj_neighbors = np.all([adj_i - {y} <= adj(P, y) for y in n_i])
            found = sink and adj_neighbors
            if found:
                real_i = indexes[i]
                real_neighbors = [indexes[j] for j in n_i]
                for j in real_neighbors:
                    G[j, real_i] = 1
                all_but_i = list(set(range(len(P))) - {i})
                P = P[all_but_i, :][:, all_but_i]
                indexes.remove(real_i)
            else:
                i += 1
        if not found:
            raise ValueError("PDAG does not admit consistent extension")
    return G


def only_directed(P):
    """
    https://github.com/juangamella/ges/blob/master/ges/main.py

    Extract only the directed edges from a partially directed graph.

    Args:
        P (NDArray): The input partially directed graph.

    Returns:
        NDArray: The graph containing only the directed edges.
    """
    mask = (P != 0) & (P.T == 0)
    G = np.zeros_like(P)
    G[mask] = P[mask]
    return G


def _dag_to_cpdag(graph: NDArray) -> NDArray:
    """
    https://github.com/juangamella/ges/blob/master/ges/main.py
    Convert a DAG (Directed Acyclic Graph) to a CPDAG (Completed Partially Directed Acyclic Graph).

    Args:
        graph (NDArray): The input DAG.

    Returns:
        NDArray: The resulting CPDAG.
    """
    ordered = order_edges(graph)
    labelled = label_edges(ordered)
    cpdag = np.zeros_like(labelled)
    cpdag[labelled == 1] = labelled[labelled == 1]
    fros, tos = np.where(labelled == -1)
    for x, y in zip(fros, tos):
        cpdag[x, y], cpdag[y, x] = 1, 1
    return cpdag


def label_edges(ordered):
    """
    https://github.com/juangamella/ges/blob/master/ges/main.py

    Label the edges of a partially directed graph.

    Args:
        ordered (NDArray): The ordered graph.

    Returns:
        NDArray: The graph with labeled edges.
    """
    # Validate the input
    no_edges = (ordered != 0).sum()
    if sorted(ordered[ordered != 0]) != list(range(1, no_edges + 1)):
        raise ValueError("The ordering of edges is not valid:", ordered[ordered != 0])
    # define labels: 1: compelled, -1: reversible, -2: unknown
    COM, REV, UNK = 1, -1, -2
    labelled = (ordered != 0).astype(int) * UNK
    while (labelled == UNK).any():
        unknown_edges = (ordered * (labelled == UNK).astype(int)).astype(float)
        unknown_edges[unknown_edges == 0] = -np.inf
        (x, y) = np.unravel_index(np.argmax(unknown_edges), unknown_edges.shape)
        Ws = np.where(labelled[:, x] == COM)[0]
        end = False
        for w in Ws:
            if labelled[w, y] == 0:
                labelled[list(parents_of(labelled, y)), y] = COM
                end = True
                break
            else:
                labelled[w, y] = COM
        if not end:
            z_exists = len(parents_of(labelled, y) - {x} - parents_of(labelled, x)) > 0
            unknown = np.where(labelled[:, y] == UNK)[0]
            assert x in unknown
            labelled[unknown, y] = COM if z_exists else REV
    return labelled


def order_edges(G):
    """
    https://github.com/juangamella/ges/blob/master/ges/main.py

    Order the edges of a graph.

    Args:
        G (NDArray): The input graph.

    Returns:
        NDArray: The ordered graph.
    """
    order = topological_ordering(G)
    ordered = (G != 0).astype(int) * -1
    i = 1
    while (ordered == -1).any():
        froms, tos = np.where(ordered == -1)
        with_unlabelled = np.unique(np.hstack((froms, tos)))
        y = sort(with_unlabelled, reversed(order))[0]
        unlabelled_parents_y = np.where(ordered[:, y] == -1)[0]
        x = sort(unlabelled_parents_y, order)[0]
        ordered[x, y] = i
        i += 1
    return ordered


def sort(L, order=None):
    """
    Sorts the elements of a list according to a specified order.

    Args:
        L (list): The list to be sorted.
        order (list, optional): The specified order. Defaults to None.

    Returns:
        list: The sorted list.
    """
    L = list(L)
    if order is None:
        return sorted(L)
    else:
        order = list(order)
        pos = np.zeros(len(order), dtype=int)
        pos[order] = range(len(order))
        positions = [pos[l] for l in L]
        return [tup[1] for tup in sorted(zip(positions, L))]
