from cxl.generative.vae import CGNNLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph


def main() -> None:
    learner = CGNNLearner()
    observations, _ = generate_linear_gaussian(10000)
    graph = learner.fit(observations)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
