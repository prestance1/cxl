from numpy.typing import NDArray
from cxl.graph.graph_utils import (
    create_complete_graph,
    adj,
    remove_edge,
    find_unshielded_triples,
    get_all_edges,
)
from .utils import get_max_depth
from concurrent.futures import Executor
import functools
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from cxl.config import HardwareConfig
from cxl.independence.fisher import FisherZTest
import math
from cxl.config import ComputeBackend
from tqdm import tqdm
from ..common import SeparationSetMapping


class ParallelSkeletonLearner:
    """
    Class for learning the skeleton of a causal graph in parallel.

    Attributes:
        hw_config (HardwareConfig): Hardware configuration for computation.
        alpha (float): The significance level for conditional independence tests.
        max_depth (int | None): Maximum depth for the skeleton learning phase.
        indep_tester_factory: Factory for creating independence testers.
        verbose (bool): Flag indicating whether to display verbose output.
        executor (Executor): Executor for parallel execution.
        max_workers (int): Maximum number of workers for parallel execution.
    """

    def __init__(
        self,
        hw_config: HardwareConfig,
        alpha: float = 0.05,
        max_depth: int | None = None,
        indep_tester_factory=FisherZTest,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ParallelSkeletonLearner.

        Args:
            hw_config (HardwareConfig): Hardware configuration for computation.
            alpha (float): The significance level for conditional independence tests.
            max_depth (int | None): Maximum depth for the skeleton learning phase.
            indep_tester_factory: Factory for creating independence testers.
            verbose (bool): Flag indicating whether to display verbose output.
        """
        if hw_config.compute_backend is ComputeBackend.MULTITHREAD:
            self.executor: Executor = ThreadPoolExecutor(hw_config.max_workers)
        else:
            self.executor: Executor = ProcessPoolExecutor(hw_config.max_workers)
        self.max_depth = max_depth
        self.ind_tester_factory = indep_tester_factory
        self.alpha = alpha
        self.max_workers = hw_config.max_workers
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
        indep_tester = FisherZTest(observations, self.alpha)
        m, no_variables = observations.shape
        separation_sets = {}
        graph = create_complete_graph(no_variables)

        max_depth = (
            get_max_depth(no_variables) if self.max_depth is None else self.max_depth
        )
        with self.executor:
            for depth in (t := tqdm(range(max_depth + 1), disable=not self.verbose)):
                J = list(get_all_edges(graph))
                batch_size = int(math.ceil(len(J) / self.max_workers))
                batches = list(itertools.batched(J, batch_size))
                run_mini_pc = functools.partial(
                    run_pc_at_depth, depth=depth, indep_tester=indep_tester, graph=graph
                )
                t.set_description(f"executing level {depth}")
                results = self.executor.map(
                    run_mini_pc, batches
                )  # distribute tasks to workers
                # sync results
                t.set_description("synchronising results")

                for x, y, z in itertools.chain(*results):
                    remove_edge(graph, x, y)
                    separation_sets[(x, y)] = z
                    separation_sets[(y, x)] = z

        unshielded_triples = find_unshielded_triples(graph)
        return (graph, separation_sets, unshielded_triples)


def run_pc_at_depth(
    edges: list[int], depth: int, indep_tester, graph: NDArray
) -> list[tuple[int, int, set[int]]]:
    """
    Run PC algorithm at a specific depth.

    Args:
        edges (list[int]): List of edges.
        depth (int): Depth for conditional independence tests.
        indep_tester: Independence tester.
        graph (NDArray): The causal graph.

    Returns:
        list[tuple[int, int, set[int]]]: List of tuples representing discovered edges.
    """
    results = []
    for x, y in edges:
        adj_x_y = adj(graph, x) - {y}
        found = False
        if len(adj_x_y) >= depth:
            for z_x in itertools.combinations(adj_x_y, depth):
                if indep_tester.is_conditionally_independent(x, y, z_x):
                    results.append((x, y, set(z_x)))
                    found = True
                    break
        if found:
            continue
        adj_y_x = adj(graph, y) - {x}
        if len(adj_y_x) >= depth:
            for z_y in itertools.combinations(adj_y_x, depth):
                if indep_tester.is_conditionally_independent(x, y, z_y):
                    results.append((x, y, set(z_y)))
                    break
    return results
