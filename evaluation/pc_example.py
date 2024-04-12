from cxl.constraint.pc import PCLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph
import numpy as np
import pandas as pd


def main():
    learner = PCLearner()
    observations, _ = generate_linear_gaussian(1000)
    graph = learner.fit(observations)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
