from cxl.constraint.skeleton.discover_skeleton_parallel import ParallelSkeletonLearner
from cxl.config import HardwareConfig, ComputeBackend
from cxl.utils import generate_linear_gaussian, pretty_print_graph


def main():
    hw_config = HardwareConfig(ComputeBackend.MULTICORE)
    learner = ParallelSkeletonLearner(hw_config=hw_config)
    observations, _ = generate_linear_gaussian(1000)
    graph, *_ = learner.learn(observations)
    pretty_print_graph(graph)


if __name__ == "__main__":
    main()
