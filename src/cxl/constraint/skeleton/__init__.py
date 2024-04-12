from cxl.config import HardwareConfig, ComputeBackend
from .discover_skeleton_cpu import CPUSkeletonLearner
from .discover_skeleton_gpu import GPUSkeletonLearner
from .discover_skeleton_parallel import ParallelSkeletonLearner


def create_skeleton_learner(
    alpha,
    max_depth: int | None,
    hardware_config: HardwareConfig | None,
    indep_tester_factory,
    verbose: bool,
):
    """
    Create a skeleton learner based on the provided parameters.

    Args:
        alpha (float): The significance level for conditional independence tests.
        max_depth (int | None): Maximum depth of the graph (DAG). If None, determined automatically.
        hardware_config (HardwareConfig | None): Configuration for hardware acceleration.
        indep_tester_factory (Callable): Factory function for creating independence testers.
        verbose (bool): Whether to output verbose logging.

    Returns:
        SkeletonLearner: An instance of a skeleton learner.
    """

    if hardware_config is None:
        return CPUSkeletonLearner(alpha, max_depth, indep_tester_factory, verbose)
    if hardware_config.compute_backend == ComputeBackend.MULTICORE:
        return ParallelSkeletonLearner(
            hardware_config, alpha, max_depth, indep_tester_factory, verbose
        )
    elif hardware_config.compute_backend == ComputeBackend.GPU:
        return GPUSkeletonLearner(alpha, max_depth, verbose)
    else:
        return CPUSkeletonLearner(alpha, max_depth, indep_tester_factory, verbose)
