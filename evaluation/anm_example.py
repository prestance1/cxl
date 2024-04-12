from cxl.utils import pretty_print_graph
from cxl.pairwise.anm import ANMLearner
import numpy as np


def main() -> None:
    learner = ANMLearner()
    N = 1000
    x0 = np.random.randn(N)
    x3 = x0**3 + x0 + 0.8 * np.random.randn(1000)
    observations = np.vstack([x0, x3]).T
    graph = learner.fit(observations)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
