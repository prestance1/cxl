from cxl.constraint.skeleton import CPUSkeletonLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph
import numpy as np
import pandas as pd


def main():
    learner = CPUSkeletonLearner()
    observations, _ = generate_linear_gaussian(1000)
    graph, *_ = learner.learn(observations)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
