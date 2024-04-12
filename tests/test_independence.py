import numpy as np
from cxl.independence.fisher import FisherZTest


def test_fisher_independent():
    p = np.random.randn(1000)
    q = np.random.randn(1000)
    o = (1 * p) + np.random.randn(1000)
    observations = np.vstack([p, q, o]).T
    indep_tester = FisherZTest(observations, 0.05)
    assert indep_tester.is_independent(0, 1)


def test_fisher_dependent():
    p = np.random.randn(1000)
    q = np.random.randn(1000)
    o = (1 * p) + np.random.randn(1000)
    observations = np.vstack([p, q, o]).T
    indep_tester = FisherZTest(observations, 0.05)
    assert not indep_tester.is_independent(0, 2)
