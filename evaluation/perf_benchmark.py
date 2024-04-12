from cxl.utils import *
from cxl.score.ges import GESLearner
from cxl.constraint.pc import PCLearner
from cxl.gradient.notears import NotearsLearner
from cxl.pairwise.lingam import LingamLearner
from cxl.utils import (
    generate_linear_gaussian,
    pretty_print_graph,
    generate_linear_gaussian_p,
)
from cxl.config import HardwareConfig, ComputeBackend
import numpy as np
import pandas as pd


@benchmark
def no_tears_mem():
    print("notears:")
    learner = NotearsLearner()
    observations, _ = generate_linear_gaussian(1000)
    graph = learner.fit(observations, 0.5)


@benchmark
def pc_mem():
    print("pc:")
    learner = PCLearner()
    observations, _ = generate_linear_gaussian(1000)
    graph = learner.fit(observations)
    print(graph)


@benchmark
def ges_mem():
    print("ges:")
    learner = GESLearner()
    observations, _ = generate_linear_gaussian(1000)
    graph = learner.fit(observations)


@benchmark
def lingam_mem():
    print("lingam")
    n = 1000
    e = lambda n: np.random.laplace(0, 1, n)
    x3 = e(n)
    x2 = 0.3 * x3 + e(n)
    x1 = 0.3 * x3 + 0.3 * x2 + e(n)
    x0 = 0.3 * x2 + 0.3 * x1 + e(n)
    x4 = 0.3 * x1 + 0.3 * x0 + e(n)
    X = np.vstack([x0, x1, x2, x3, x4]).T
    learner = LingamLearner()
    graph = learner.fit(X, 0.2)


@benchmark
def par_t_pc():
    print("pc thread parallel")
    learner = PCLearner(hardware_config=HardwareConfig(ComputeBackend.MULTITHREAD))
    observations = generate_linear_gaussian_p(500, 1000)
    graph = learner.fit(observations)


@benchmark
def par_p_pc():
    print("pc process parallel")
    learner = PCLearner(hardware_config=HardwareConfig(ComputeBackend.MULTICORE))
    observations = generate_linear_gaussian_p(500, 1000)
    graph = learner.fit(observations)


def main():
    pc_mem()
    ges_mem()
    no_tears_mem()
    # par_p_pc()
    # par_t_pc()
    lingam_mem()


if __name__ == "__main__":
    main()
