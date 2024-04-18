import numpy as np
from numpy.typing import NDArray
from typing import Self, Iterator
import itertools
from collections import deque


def create_complete_graph(n: int, marker: int = 1) -> NDArray:
    """
    Create a complete graph with the specified number of nodes.

    Args:
        n (int): Number of nodes in the graph.
        marker (int, optional): Marker value for the edges. Defaults to 1.

    Returns:
        NDArray: Complete graph with edges marked.
    """
    graph = np.full((n, n), marker, dtype=np.int8) - np.identity(n, dtype=np.int8)
    return graph


def create_empty_graph(n: int) -> NDArray:
    """
    Create an empty graph with the specified number of nodes.

    Args:
        n (int): Number of nodes in the graph.

    Returns:
        NDArray: Empty graph.
    """
    graph = np.empty((n, n), dtype=np.int8)
    return graph


def is_disconnected(graph, x: int, y: int) -> bool:
    """
    Check if there is no edge between two nodes in the graph.

    Args:
        graph: Graph representation.
        x (int): First node.
        y (int): Second node.

    Returns:
        bool: True if there is no edge between x and y, False otherwise.
    """
    return graph[x, y] == 0 and graph[y, x] == 0


def adj(graph, x: int) -> set[int]:
    """
    Get the set of adjacent nodes to a given node in the graph.

    Args:
        graph: Graph representation.
        x (int): Node index.

    Returns:
        set[int]: Set of adjacent nodes to x.
    """
    return set(np.where((graph[x, :] != 0) | (graph[:, x] != 0))[0])


def remove_edge(graph, x: int, y: int) -> None:
    """
    Remove an edge between two nodes in the graph.

    Args:
        graph: Graph representation.
        x (int): First node.
        y (int): Second node.

    Returns:
        None
    """
    graph[x, y] = 0
    graph[y, x] = 0


def orient_edge(
    graph, x: int, y: int, tail_marker: int = 0, head_marker: int = 1
) -> None:
    """
    Orient an edge between two nodes in the graph.

    Args:
        graph: Graph representation.
        x (int): Parent node.
        y (int): Child node.
        tail_marker (int, optional): Marker for the tail of the edge. Defaults to 0.
        head_marker (int, optional): Marker for the head of the edge. Defaults to 1.

    Returns:
        None
    """
    graph[x, y] = head_marker
    graph[y, x] = tail_marker


def is_parent(
    graph, x: int, y: int, tail_marker: int = 0, head_marker: int = 1
) -> bool:
    """
    Check if x is a parent of y in the graph.

    Args:
        graph: Graph representation.
        x (int): Potential parent node.
        y (int): Potential child node.
        tail_marker (int, optional): Marker for the tail of the edge. Defaults to 0.
        head_marker (int, optional): Marker for the head of the edge. Defaults to 1.

    Returns:
        bool: True if x is a parent of y, False otherwise.
    """
    return (graph[x, y] == head_marker) and (graph[y, x] == tail_marker)


def is_indirected(graph, x: int, y: int, marker: int = 1):
    """
    Check if there is an undirected edge between two nodes in the graph.

    Args:
        graph: Graph representation.
        x (int): First node.
        y (int): Second node.
        marker (int, optional): Marker for the undirected edge. Defaults to 1.

    Returns:
        bool: True if there is an undirected edge between x and y, False otherwise.
    """
    return graph[x, y] == marker and graph[y, x] == marker


def find_unshielded_triples(graph, marker=1) -> Iterator[int]:
    """
    Find unshielded triples in the graph.

    Args:
        graph: Graph representation.
        marker (int, optional): Marker for the edges. Defaults to 1.

    Yields:
        Iterator[int]: Unshielded triples found in the graph.
    """
    n = graph.shape[0]
    for x, y, z in itertools.permutations(range(n), 3):
        if (
            graph[x, y] == marker
            and graph[y, x] == marker
            and graph[y, z] == marker
            and graph[z, y] == marker
            and graph[x, z] == 0
            and graph[z, x] == 0
        ):
            yield (x, y, z)


def get_neighbours(graph, x: int, mark=1) -> set[int]:
    """
    Get neighbors of a node in the graph.

    Args:
        graph: Graph representation.
        x (int): Node index.
        mark (int, optional): Marker for the edges. Defaults to 1.

    Returns:
        set[int]: Set of neighbors of the node.
    """
    return set(np.where((graph[x, :] == mark) & (graph[:, x] == mark))[0])


def is_collider(graph, x: int, y: int, z: int) -> bool:
    """
    Check if there is a collider between three nodes in the graph.

    Args:
        graph: Graph representation.
        x (int): Node index.
        y (int): Node index.
        z (int): Node index.

    Returns:
        bool: True if there is a collider, False otherwise.
    """
    return is_parent(graph, x, y) and is_parent(graph, z, y)


def is_adjacent(
    graph,
    x: int,
    y: int,
) -> bool:
    """
    Check if two nodes are adjacent in the graph.

    Args:
        graph: Graph representation.
        x (int): Node index.
        y (int): Node index.

    Returns:
        bool: True if nodes are adjacent, False otherwise.
    """
    return (graph[x, y] != 0) or (graph[y, x] != 0)


def get_pds(graph, xi: int) -> set[int]:
    """
    Get possible d-separators for a node xi in the graph.

    Args:
        graph: Graph representation.
        xi (int): Node index.

    Returns:
        set[int]: Set of possible d-separators for xi.
    """
    pds = set()
    q = deque([(xi, xj) for xj in get_neighbours(graph, xi)])
    while q:
        n1, n2 = q.popleft()
        for n3 in get_neighbours(graph, n2):
            if is_possible_collider(graph, n1, n2, n3):
                pds.add(n3)
                q.append((n2, n3))
    return pds


def remark_edges(graph: NDArray, marker: int) -> NDArray:
    """
    Mark edges in the graph with the given marker.

    Args:
        graph: Graph representation.
        marker (int): Marker value.

    Returns:
        NDArray: Graph with marked edges.
    """
    edges = np.nonzero(graph)
    graph[edges] = marker


def parents_of(
    graph: NDArray, x: int, tail_marker: int = 0, head_marker: int = 1
) -> set[int]:
    """
    Get the parents of a node in the graph.

    Args:
        graph: Graph representation.
        x (int): Node index.
        tail_marker (int, optional): Tail marker value. Defaults to 0.
        head_marker (int, optional): Head marker value. Defaults to 1.

    Returns:
        set[int]: Set of parent nodes.
    """
    return set(np.where((graph[x, :] == tail_marker) & (graph[:, x] == head_marker))[0])


def discriminating_path(graph: NDArray, a: int, b: int, c: int) -> int | None:
    """
    Find a node on a discriminating path from a to c through b.

    Args:
        graph: Graph representation.
        a (int): Node index.
        b (int): Node index.
        c (int): Node index.

    Returns:
        int | None: Node index on the discriminating path or None if not found.
    """
    d_nodes = set(np.where(graph[:, a] == 1)[0])
    a_nodes = set()
    for nd in d_nodes:
        if not is_adjacent(graph, nd, c):
            return nd
        else:
            if is_parent(graph, nd, c) and graph[nd, na] == 1:
                a_nodes.add(nd)

    for nna in a_nodes:
        nd = discriminating_path(nna, b, c)
        if nd is not None:
            return nd
    return None


def semi_neighbours(graph: NDArray, x: int) -> set[int]:
    """
    Get semi-neighbours of a node in the graph.

    Args:
        graph: Graph representation.
        x (int): Node index.

    Returns:
        set[int]: Set of semi-neighbour nodes.
    """
    return set(np.where(graph[x, :] == 1)[0])


def children_of(graph, i):
    """
    Get the children of a node in the graph.

    Args:
        graph: Graph representation.
        i (int): Node index.

    Returns:
        set[int]: Set of child nodes.
    """
    return set(np.where((graph[i, :] != 0) & (graph[:, i] == 0))[0])


def neighbors(A, i: int):
    """
    Get the neighbors of a node in the graph.

    Args:
        A: Graph representation.
        i (int): Node index.

    Returns:
        set[int]: Set of neighboring nodes.
    """
    return set(np.where((A[i, :] != 0) & (A[:, i] != 0))[0])


def na(graph: NDArray, y: int, x: int) -> set[int]:
    """
    Get the neighbours of y that are adjacent to x in the graph.

    Args:
        graph: Graph representation.
        y (int): Node index.
        x (int): Node index.

    Returns:
        set[int]: Set of nodes that are neighbors of y and adjacent to x.
    """
    return neighbors(graph, y) & adj(graph, x)


def topological_ordering(graph: NDArray) -> list[int]:
    """
    Perform topological ordering of the nodes in a directed acyclic graph (DAG).

    Args:
        graph: Directed acyclic graph representation.

    Returns:
        list[int]: Topologically ordered list of node indices.

    Raises:
        ValueError: If the given graph is not a DAG (contains cycles).
    """
    # Run the algorithm from the 1962 paper "Topological sorting of
    # large networks" by AB Kahn
    graph = graph.copy()
    sinks = list(np.where(graph.sum(axis=0) == 0)[0])
    ordering = []
    while len(sinks) > 0:
        i = sinks.pop()
        ordering.append(i)
        for j in children_of(graph, i):
            graph[i, j] = 0
            if len(parents_of(graph, j)) == 0:
                sinks.append(j)
    # If A still contains edges there is at least one cycle
    if graph.sum() > 0:
        raise ValueError("The given graph is not a DAG")
    else:
        return ordering


def is_clique(graph: NDArray, nodes: list[int]) -> bool:
    """
    Check if the given set of nodes forms a clique in the graph.

    Args:
        graph: Graph representation.
        nodes (list[int]): List of node indices.

    Returns:
        bool: True if the nodes form a clique, False otherwise.
    """
    for u, v in itertools.combinations(nodes, 2):
        if not is_adjacent(graph, u, v):
            return False
    return True


def is_possible_collider(graph: NDArray, x: int, m: int, y: int) -> bool:
    """
    Check if it is possible for nodes x, m, and y to form a collider pattern.

    Args:
        graph: Graph representation.
        x (int): Node index.
        m (int): Node index.
        y (int): Node index.

    Returns:
        bool: True if it is possible for nodes x, m, and y to form a collider, False otherwise.
    """
    if x == y or is_disconnected(graph, x, m) or is_disconnected(graph, m, y):
        return False
    else:
        return is_adjacent(graph, x, y) or is_collider(graph, x, m, y)


def get_all_edges(graph: NDArray, mark=1):
    """
    Generate all edges in the graph with a given marker.

    Args:
        graph: Graph representation.
        mark (int, optional): Marker value for edges. Defaults to 1.

    Yields:
        tuple[int, int]: Tuple representing an edge (node index, node index).
    """
    n = len(graph)
    for i, j in itertools.combinations(range(n), 2):
        if graph[i, j] == mark:
            yield (i, j)


def max_number_of_neighbors(adjacency_matrix):
    # Sum along the rows to get the number of neighbors for each node
    neighbors_count = np.sum(adjacency_matrix, axis=1)
    # Find the maximum number of neighbors
    max_neighbors = np.max(neighbors_count)
    return max_neighbors
