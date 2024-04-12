import numpy as np
from numpy.typing import NDArray
from cxl.config import HardwareConfig
from .skeleton import create_skeleton_learner
import itertools
from cxl.graph.graph_utils import orient_edge, is_parent, is_indirected
from cxl.independence.fisher import FisherZTest
from .common import SeparationSetMapping


class PCLearner:
    """
    Class for learning the structure of a causal graph using the PC algorithm.

    Attributes:
        alpha (float): The significance level for conditional independence tests.
        max_depth (int | None): Maximum depth for the skeleton learning phase.
        hardware_config (HardwareConfig | None): Hardware configuration for computation.
        verbose (bool): Flag indicating whether to display verbose output.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_depth: int | None = None,
        hardware_config: HardwareConfig | None = None,
        indep_tester_factory=FisherZTest,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the PCLearner.

        Args:
            alpha (float): The significance level for conditional independence tests.
            max_depth (int | None): Maximum depth for the skeleton learning phase.
            hardware_config (HardwareConfig | None): Hardware configuration for computation.
            verbose (bool): Flag indicating whether to display verbose output.
        """
        self.skeleton_learner = create_skeleton_learner(
            alpha, max_depth, hardware_config, indep_tester_factory, verbose
        )

    def fit(self, observations: NDArray) -> NDArray:
        """
        Fit the model to the observations.

        Args:
            observations (NDArray): The observational data.

        Returns:
            NDArray: The learned causal graph.
        """
        graph, separation_sets, unshielded_triples = self.skeleton_learner.learn(
            observations
        )
        self._orient_v_structures(graph, unshielded_triples, separation_sets)
        self._propagate_chains(graph)
        return graph

    def _orient_v_structures(
        self,
        graph: NDArray,
        unshielded_triples: list[int],
        separation_sets: SeparationSetMapping,
    ):
        """
        Orient the V-structures in the graph.

        Args:
            graph (NDArray): The causal graph.
            unshielded_triples (list[int]): List of unshielded triples in the graph.
            separation_sets: Separation sets for the graph.
        """
        for x, y, z in unshielded_triples:
            if y not in separation_sets[(x, z)]:
                # orient  x -> y <- z
                orient_edge(graph, x, y)
                orient_edge(graph, z, y)

    def _propagate_chains(self, graph: NDArray):
        """
        Apply propagation rule to the chains in the graph.

        Args:
            graph (NDArray): The causal graph.
        """
        n = len(graph)
        for x, y, z in itertools.permutations(range(n), 3):
            if is_parent(graph, x, y) and is_indirected(graph, y, z):
                # orient  x -> y -> z
                orient_edge(graph, y, z)
