from cxl.constraint.skeleton import GPUSkeletonLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph


def main():
    learner = GPUSkeletonLearner()
    observations, _ = generate_linear_gaussian(n_samples=1000)
    graph, *_ = learner.learn(observations)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
