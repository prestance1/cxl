from cxl.constraint.pc import PCLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph
import numpy as np
import pandas as pd


def prepare_data():
    pass


def main():
    X, ground_truth = prepare_data()
    learner = PCLearner()
    graph = learner.fit(X)


if __name__ == "__main__":
    main()
