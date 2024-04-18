from cxl.constraint.pc import PCLearner
from cxl.utils import *
from cxl.config import HardwareConfig, ComputeBackend


# where do we go from now


class MyIndependenceTester:

    def __init__(self, alpha, observations) -> None:
        pass

    def is_conditionally_independent(
        self, x: int, y: int, conditioning_set: set[int]
    ) -> bool:
        return True


def main():
    hw_config = HardwareConfig(compute_backend=ComputeBackend.MULTICORE, max_workers=16)
    observations, _ = generate_linear_gaussian(1000)
    learner = PCLearner(
        indep_tester_factory=MyIndependenceTester,
        hardware_config=hw_config,
        verbose=True,
    )
    graph = learner.fit(observations)
