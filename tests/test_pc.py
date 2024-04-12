import numpy as np
from numpy.typing import NDArray
from cxl.graph.graph_utils import create_empty_graph
from numpy.testing import assert_equal
from cxl.constraint.pc import PCLearner


def generate_linear_gaussian(n_samples: int) -> tuple[NDArray, NDArray]:

    p = np.random.randn(n_samples)
    q = np.random.randn(n_samples)
    o = (1 * p) + np.random.randn(n_samples)
    r = p + q + 0.1 * np.random.randn(n_samples)
    s = 0.7 * r + 0.1 * np.random.randn(n_samples)

    ground_truth = create_empty_graph(5)
    ground_truth[0, 2] = 1
    ground_truth[0, 3] = 1
    ground_truth[1, 3] = 1
    ground_truth[3, 4] = 1

    return np.vstack([p, q, o, r, s]).T, ground_truth


def test_simple_pc():
    p = np.random.randn(1000)
    q = np.random.randn(1000)
    o = (1 * p) + np.random.randn(1000)
    observations = np.vstack([p, q, o]).T
    gt = create_empty_graph(3)
    gt[0, 2] = 1
    gt[2, 0] = 1
    learner = PCLearner()
    graph = learner.fit(observations)
    assert_equal(graph, gt)


def test_pc():
    p = np.random.randn(1000)
    q = np.random.randn(1000)
    r = p + q + 0.1 * np.random.randn(1000)
    s = 0.7 * r + 0.1 * np.random.randn(1000)
    observations = np.vstack([p, q, r, s]).T
    ground_truth = np.zeros((4, 4), dtype=np.int8)
    ground_truth[0, 2] = 1
    ground_truth[1, 2] = 1
    ground_truth[2, 3] = 1
    learner = PCLearner()
    g = learner.fit(observations)
    assert_equal(g, ground_truth)
