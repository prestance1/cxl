from numpy.typing import NDArray
from .utils import get_max_depth
from cxl.graph.graph_utils import (
    create_complete_graph,
    is_disconnected,
    adj,
    remove_edge,
    find_unshielded_triples,
)
from ..common import SeparationSetMapping
from cxl.independence.fisher import FisherZTest
import itertools
from tqdm import tqdm


class CPUSkeletonLearner:
    """
    Class for learning the skeleton of a causal graph using CPU computation.

    Attributes:
        _max_depth (int | None): Maximum depth for the skeleton learning phase.
        ind_tester_factory: Factory for generating independence testers.
        alpha (float): The significance level for conditional independence tests.
        verbose (bool): Flag indicating whether to display verbose output.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_depth: int | None = None,
        indep_tester_factory=FisherZTest,
        verbose=False,
    ) -> None:
        """
        Initialize the CPUSkeletonLearner.

        Args:
            alpha (float): The significance level for conditional independence tests.
            max_depth (int | None): Maximum depth for the skeleton learning phase.
            indep_tester_factory: Factory for generating independence testers.
            verbose (bool): Flag indicating whether to display verbose output.
        """
        self._max_depth: int | None = max_depth
        self.ind_tester_factory = indep_tester_factory
        self.alpha = alpha
        self.verbose = verbose

    def learn(
        self, observations: NDArray
    ) -> tuple[NDArray, SeparationSetMapping, list[tuple[int, int, int]]]:
        """
        Learn the skeleton of the causal graph.

        Args:
            observations (NDArray): The observational data.

        Returns:
            tuple[NDArray, SeparationSetMapping, list[tuple[int, int, int]]]: The learned skeleton, separation sets, and unshielded triples.
        """
        _, no_variables = observations.shape
        max_depth = (
            get_max_depth(no_variables) if self._max_depth is None else self._max_depth
        )
        indep_tester = self.ind_tester_factory(observations, self.alpha)
        separation_sets = {}
        graph = create_complete_graph(no_variables)
        for depth in (t := tqdm(range(max_depth + 1), disable=not self.verbose)):
            for x, y in itertools.combinations(range(no_variables), 2):
                if is_disconnected(graph, x, y):
                    continue
                t.set_description(f"exploring condition sets at depth: {depth}")
                Z = adj(graph, x) - {y}
                for sepset in itertools.combinations(Z, depth):
                    sepset = set(sepset)
                    if indep_tester.is_conditionally_independent(x, y, sepset):
                        remove_edge(graph, x, y)
                        separation_sets[(x, y)] = sepset
                        separation_sets[(y, x)] = sepset
                        break
        unshielded_triples = list(find_unshielded_triples(graph))
        return (graph, separation_sets, unshielded_triples)
