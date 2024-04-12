from cxl.gradient.notears import NotearsLearner
from cxl.utils import generate_linear_gaussian, pretty_print_graph


def main() -> None:
    learner = NotearsLearner()
    observations, _ = generate_linear_gaussian(1000)
    graph = learner.fit(observations, 0.5)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
