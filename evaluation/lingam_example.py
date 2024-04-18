from cxl.pairwise.lingam import LingamLearner
import numpy as np
from cxl.utils import pretty_print_graph


def main():
    n = 1000
    e = lambda n: np.random.laplace(0, 1, n)
    x3 = e(n)
    x2 = 0.3 * x3 + e(n)
    x1 = 0.3 * x3 + 0.3 * x2 + e(n)
    x0 = 0.3 * x2 + 0.3 * x1 + e(n)
    x4 = 0.3 * x1 + 0.3 * x0 + e(n)
    X = np.vstack([x0, x1, x2, x3, x4]).T
    learner = LingamLearner(verbose=True)
    graph = learner.fit(X, 0.2)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
