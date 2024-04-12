from cxl.pairwise.anm import ANMLearner
import numpy as np


def test_anm_dependent():
    learner = ANMLearner()
    N = 1000
    x0 = np.random.randn(N)
    x3 = x0**3 + x0 + 0.8 * np.random.randn(1000)
    observations = np.vstack([x0, x3]).T
    graph = learner.fit(observations)
    assert graph[0, 1] == 1 and graph[1, 0] == 0


def test_anm_independent():
    learner = ANMLearner()
    N = 1000
    x0 = np.random.randn(N)
    x3 = 0.8 * np.random.randn(1000)
    observations = np.vstack([x0, x3]).T
    graph = learner.fit(observations)
    assert graph[0, 1] == 0 and graph[1, 0] == 0
