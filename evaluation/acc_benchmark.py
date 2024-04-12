from cxl.utils import *
from cxl.score.ges import GESLearner
from cxl.constraint.pc import PCLearner
from cxl.gradient.notears import NotearsLearner
from cxl.pairwise.lingam import LingamLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph
import numpy as np
import pandas as pd


def no_tears_mem():
    print("notears:")
    learner = NotearsLearner()
    observations, gt = generate_linear_gaussian(1000)
    graph = learner.fit(observations, 0.5)
    print({"gscore": g_score(graph, gt), "fpr": fpr(graph, gt), "f1": f1(graph, gt)})


def pc_mem():
    print("pc:")
    learner = PCLearner()
    observations, gt = generate_linear_gaussian(1000)
    graph = learner.fit(observations)
    print({"gscore": g_score(graph, gt), "fpr": fpr(graph, gt), "f1": f1(graph, gt)})


def ges_mem():
    print("ges:")
    learner = GESLearner()
    observations, gt = generate_linear_gaussian(1000)
    graph = learner.fit(observations)
    print({"gscore": g_score(graph, gt), "fpr": fpr(graph, gt), "f1": f1(graph, gt)})


def lingam_mem():
    n = 1000
    e = lambda n: np.random.laplace(0, 1, n)
    x3 = e(n)
    x2 = 0.3 * x3 + e(n)
    x1 = 0.3 * x3 + 0.3 * x2 + e(n)
    x0 = 0.3 * x2 + 0.3 * x1 + e(n)
    x4 = 0.3 * x1 + 0.3 * x0 + e(n)
    gt = np.zeros((5, 5), dtype=np.int8)
    gt[2, 0] = 1
    gt[1, 0] = 1
    gt[3, 1] = 1
    gt[2, 1] = 1
    gt[1, 4] = 1
    gt[0, 4] = 1
    X = np.vstack([x0, x1, x2, x3, x4]).T
    learner = LingamLearner()
    graph = learner.fit(X, 0.2)
    print({"gscore": g_score(graph, gt), "fpr": fpr(graph, gt), "f1": f1(graph, gt)})


def main():
    pc_mem()
    ges_mem()
    no_tears_mem()
    lingam_mem()


if __name__ == "__main__":
    main()
