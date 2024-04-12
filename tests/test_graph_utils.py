import numpy as np
import networkx as nx
from cxl.graph.graph_utils import *
import itertools


def create_collider(graph, x, y, z):
    graph[x, y] = 1
    graph[y, x] = 0
    graph[z, y] = 1
    graph[x, y] = 0


def test_create_empty():
    graph = np.zeros((5, 5), dtype=np.int8)
    assert nx.is_empty(nx.from_numpy_array(graph, create_using=nx.DiGraph))


def test_create_complete_graph():
    graph = create_complete_graph(5)
    for i, j in itertools.permutations(range(5), 2):
        assert graph[i, j] == 1


def test_adj_one_variable():
    graph = np.zeros((5, 5), dtype=np.int8)
    print(graph)
    graph[0, 1] = 1
    graph[0, 2] = 1
    graph[0, 4] = 1
    assert adj(graph, 0) == {1, 2, 4}


def test_adj_two_variables():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[0, 1] = 1
    graph[0, 2] = 1
    graph[0, 4] = 1
    graph[2, 3] = 1
    assert adj(graph, 0) == {1, 2, 4}
    assert adj(graph, 2) == {3, 0}


def test_is_disconnected():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[0, 3] = 1
    graph[0, 1] = 1
    graph[1, 2] = 1
    assert not is_disconnected(graph, 0, 1)
    assert is_disconnected(graph, 1, 3)


def test_remove_edge():
    graph = create_complete_graph(5)
    remove_edge(graph, 0, 1)
    assert is_disconnected(graph, 0, 1)
    for i, j in itertools.combinations(range(5), 2):
        if not (i, j) == (0, 1):
            assert not is_disconnected(graph, i, j)


def test_is_indirected():
    graph = create_complete_graph(5)
    assert is_indirected(graph, 0, 1)


def test_is_adjacent():
    graph = create_complete_graph(5)
    assert is_adjacent(graph, 0, 1)
    assert is_adjacent(graph, 0, 2)
    assert is_adjacent(graph, 3, 1)


def test_is_parent():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[1, 0] = 1
    assert is_parent(graph, 1, 0)


def test_orient_edge():
    graph = create_complete_graph(5)
    orient_edge(graph, 0, 1)
    assert is_parent(graph, 0, 1)


def test_is_collider():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[0, 1] = 1
    graph[2, 1] = 1
    assert is_collider(graph, 0, 1, 2)


def test_parents_of():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[0, 1] = 1
    graph[2, 1] = 1
    assert parents_of(graph, 1) == {0, 2}


def test_is_clique():
    graph = create_complete_graph(3)
    assert is_clique(graph, [0, 1, 2])


def test_is_not_clique():
    graph = create_complete_graph(3)
    graph[0, 1] = 0
    graph[1, 0] = 0
    assert not is_clique(graph, [0, 1, 2])


def test_topoligical_ordering():
    graph = np.zeros((3, 3), dtype=np.int8)
    graph[0, 1] = 1
    graph[1, 2] = 1
    assert topological_ordering(graph) == [0, 1, 2]


def test_na():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[0, 1] = 1
    graph[1, 0] = 1
    graph[0, 2] = 1
    graph[2, 0] = 1
    graph[2, 1] = 1
    graph[2, 3] = 1
    assert na(graph, 0, 2) == {1}


def test_children_of():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[1, 0] = 1
    graph[1, 2] = 1
    assert children_of(graph, 1) == {0, 2}


def test_neighbors():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[1, 0] = 1
    graph[0, 1] = 1
    graph[1, 2] = 1
    graph[2, 1] = 1
    assert neighbors(graph, 1) == {0, 2}


def test_semi_neighbors():
    graph = np.zeros((5, 5), dtype=np.int8)
    graph[1, 0] = 1
    graph[0, 1] = 1
    graph[1, 2] = 1
    graph[2, 1] = 1
    graph[1, 3] = 1
    assert semi_neighbours(graph, 1) == {0, 2, 3}


def test_is_possible_collider():
    graph = np.zeros((3, 3), dtype=np.int8)
    graph[0, 1] = 1
    graph[2, 1] = 1
    assert is_possible_collider(graph, 0, 1, 2)


def test_get_all_edges():
    graph = np.zeros((3, 3), dtype=np.int8)
    graph[0, 1] = 1
    graph[1, 2] = 1
    edges = set(get_all_edges(graph))
    assert (0, 1) in edges or (1, 0) in edges
    assert (1, 2) in edges or (2, 1) in edges


def test_pds():
    pass


def test_discriminating_path_simple():
    pass


def test_discriminating_path():
    pass
