from cxl.generative.cgnn import GNNLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph


def main():
    learner = GNNLearner()
    observations, _ = generate_linear_gaussian(1000)
    graph = learner.fit(observations)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
