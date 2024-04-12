from numba import cuda
import math
from numpy.typing import NDArray
import numpy as np
from .utils import (
    get_max_depth,
    calculate_blocks_and_threads_compact,
    calculate_blocks_and_threads_kernel_level_n,
    calculate_blocks_and_threads_kernel_level_0,
)
from cxl.graph.graph_utils import create_complete_graph, find_unshielded_triples
from scipy.stats import norm
from tqdm import tqdm
from ..common import SeparationSetMapping


class GPUSkeletonLearner:
    """
    Class for learning the skeleton of a causal graph using GPU acceleration.

    Attributes:
        _alpha (float): The significance level for conditional independence tests.
        _max_depth (int | None): Maximum depth for the skeleton learning phase.
        verbose (bool): Flag indicating whether to display verbose output.
    """

    def __init__(
        self, alpha: float = 0.05, max_depth: int | None = None, verbose: bool = False
    ) -> None:
        """
        Initialize the GPUSkeletonLearner.

        Args:
            alpha (float): The significance level for conditional independence tests.
            max_depth (int | None): Maximum depth for the skeleton learning phase.
            verbose (bool): Flag indicating whether to display verbose output.
        """
        self._alpha = alpha
        self._max_depth = max_depth
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
        m, no_variables = observations.shape
        max_depth = (
            get_max_depth(no_variables) if self._max_depth is None else self._max_depth
        )
        A = create_complete_graph(no_variables)
        d_A = cuda.to_device(A)

        separation_sets = np.full((no_variables, no_variables, max_depth), -1)
        d_separation_sets = cuda.to_device(separation_sets)

        C = np.corrcoef(observations.T)
        d_C = cuda.to_device(C)

        for l in (t := tqdm(range(max_depth + 1), disable=not self.verbose)):
            tau = _calculate_threshold(m, self._alpha, l)
            if l == 0:
                t.set_description("executing level 0...")
                row, col = calculate_blocks_and_threads_kernel_level_0(no_variables)
                _execute_level_0[row, col](d_C, tau, d_A, no_variables)
            else:
                t.set_description(f"executing level {l}...")
                d_Ag = cuda.to_device(np.zeros_like(A))
                # copying back to host is so suboptimal do better than this
                row, col = calculate_blocks_and_threads_compact(no_variables)
                _scan[row, col](no_variables, d_A, d_Ag)
                if get_max_neighbor_count(d_Ag.copy_to_host()) - 1 < l:
                    break

                row, col = calculate_blocks_and_threads_kernel_level_n(no_variables)
                _execute_level_n[row, col](
                    l,
                    d_C,
                    d_A,
                    d_Ag,
                    d_separation_sets,
                    tau,
                    no_variables,
                )
            cuda.synchronize()
        graph = d_A.copy_to_host()
        separation_sets = unpack_sepsets(d_separation_sets.copy_to_host())
        unshielded_triples = find_unshielded_triples(graph)
        return (graph, separation_sets, unshielded_triples)


def unpack_sepsets(separation_sets: NDArray) -> SeparationSetMapping:
    n, *_ = separation_sets.shape
    result = {
        (x, y): {z for z in separation_sets[x, y, :] if z != -1}
        for x in range(n)
        for y in range(n)
    }
    return result


def _calculate_threshold(m, alpha, level):
    tau = norm.ppf(1 - alpha / 2) / math.sqrt(m - level - 3)
    return tau


@cuda.jit
def _execute_level_n(level, C, A, Ag, separation_sets, tau, n):

    vi = cuda.blockIdx.x
    vj = cuda.blockIdx.y

    if vi == vj:
        return

    if vi >= n or vj >= n:
        return

    if vi < vj:
        vi, vj = vj, vi
    neighbours = _get_neighbours(Ag, vi)
    neighbour_count = int(_get_neighbour_count(Ag, vi))
    sepset_neighbours_count = max(neighbour_count - 1, 0)
    if sepset_neighbours_count < level:
        return
    sepset_count = int(binomial_coeff(sepset_neighbours_count, level))
    for i in range(cuda.threadIdx.x, sepset_count, cuda.blockDim.x):
        d_sepset = cuda.local.array(level, dtype=np.int32)
        calc_sepset_indices(sepset_neighbours_count, i, d_sepset, level)
        for s_idx in range(0, level):
            assert d_sepset[s_idx] > 0
            assert d_sepset[s_idx] <= neighbour_count
        resolve_sepset_indices(neighbours, d_sepset, vj, level)
        p = fisher_z_test(vi, vj, level, C, d_sepset)
        # p = _fisher_z_test(vi, vj, level, d_correlation_matrix, d_sepset, n)
        if p < tau:
            remove_edge(level, vi, vj, A, separation_sets, d_sepset)
        if A[vi, vj] == 0:
            break


@cuda.jit
def _execute_level_0(correlation_matrix, tau, graph, n):
    id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    vi = int(math.sqrt(2 * id + 0.25) - 0.5)
    vj = int(id - ((vi * (vi + 1) / 2)))

    if vi == vj:
        return

    if vi < vj:
        vi, vj = vj, vi

    if vi < n and vj < n:
        Z = correlation_matrix[vj, vi]
        p = abs(0.5 * math.log(abs(1 + Z) / (1 - Z)))
        if p < tau:
            graph[vi, vj] = 0
            graph[vj, vi] = 0


@cuda.jit(device=True)
def _get_neighbour_count(d_compacted, v):
    return d_compacted[0, v]


@cuda.jit(device=True)
def binomial_coeff(n: int, k: int) -> int:
    if n < k:
        return 0
    result = 1
    if k > n - k:
        k = n - k
    for i in range(k):
        result *= n - i
        result /= i + 1
    return result


@cuda.jit(device=True)
def binomial_coeff(n: int, k: int) -> int:
    if n < k:
        return 0
    result = 1
    if k > n - k:
        k = n - k
    for i in range(k):
        result *= n - i
        result /= i + 1
    return result


# this is just the subcorrelation matrix
@cuda.jit(device=True)
def _fill_m0(vi, vj, m0, correlation_matrix):
    m0[0, 0] = 1.0
    m0[0, 1] = correlation_matrix[vi, vj]
    m0[1, 0] = correlation_matrix[vj, vi]
    m0[1, 1] = 1.0


@cuda.jit(device=True)
def _fill_m1(vi, vj, m1, correlation_matrix, sepset, level):
    for i in range(0, level):
        m1[0, i] = correlation_matrix[vi, sepset[i]]
        m1[1, i] = correlation_matrix[vj, sepset[i]]


@cuda.jit(device=True)
def _fill_m2(level, m2, correlation_matrix, sepset):
    for i in range(level):
        for k in range(level):
            m2[i, k] = correlation_matrix[sepset[i], sepset[k]]


# }


@cuda.jit(device=True)
def _compute_m2_inv(m2):
    L = np.linalg.cholesky(m2.T @ m2)
    R = np.linalg.inv(L.T @ L)
    res = L @ R @ R @ L.T @ m2.T
    return res


@cuda.jit(device=True)
def cholesky_decomposition(A, L, n):
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = math.sqrt(A[i, i] - s)
            else:
                L[i, j] = 1.0 / L[j, j] * (A[i, j] - s)


@cuda.jit(device=True)
def compute_h(m0, m1, m2_inv):
    temp = m2_inv @ m1.T
    temp2 = m1 @ temp
    H = m0 - temp2
    return H


@cuda.jit(device=True)
def fisher_z_transform(H):
    rho = H[0, 1] / (math.sqrt(abs(H[0, 0])) * math.sqrt(abs(H[1, 1])))
    Z = abs(0.5 * (math.log(abs((1 + rho))) - math.log(abs(1 - rho))))
    return Z

    # rho = H[0, 1] / (math.sqrt(abs(H[0, 0])) * math.sqrt(abs(H[1, 1])))
    # Z = abs(0.5 * math.log(abs(1 + rho))) - math.log(abs(1 - rho))
    # return Z


@cuda.jit(device=True)
def fisher_z_test(vi, vj, level, d_correlation_matrix, sepset):
    m0 = cuda.local.array((2, 2), dtype=np.float32)  # good
    m1 = cuda.shared.array((2, level), dtype=np.float32)  # good
    m2 = cuda.shared.array((level, level), dtype=np.float32)  # good
    _fill_m0(vi, vj, m0, d_correlation_matrix)
    _fill_m1(vi, vj, m1, d_correlation_matrix, sepset, level)
    _fill_m2(level, m2, d_correlation_matrix, sepset)
    # m2_inv = cuda.local.array((level, level), dtype=np.float32)
    m2_inv = _compute_m2_inv(m2)
    # H = cuda.local.array((2, 2), dtype=np.float32)
    H = compute_h(m0, m1, m2_inv)
    p = fisher_z_transform(H)
    return p


@cuda.jit(device=True)
def remove_edge(level, vi, vj, graph, separation_sets, sepset):
    if vi < vj:
        vi, vj = vj, vi
    if cuda.atomic.cas(graph, (vi, vj), 1, 0) == 1:
        graph[vj, vi] = 0
        for i in range(level):
            separation_sets[vi, vj, i] = sepset[i]
            separation_sets[vj, vi, i] = sepset[i]


@cuda.jit(device=True)
def _get_neighbours(d_compacted, v):
    return d_compacted[1:, v]


@cuda.jit
def _scan(n, d_skeleton, d_compacted):
    col_idx = cuda.grid(1)
    count = 0
    if col_idx < n:
        for i in range(0, n):
            exists = d_skeleton[i, col_idx] * (i != col_idx)
            count += exists
            d_compacted[count * exists, col_idx] = i
        d_compacted[0, col_idx] = count


def get_max_neighbor_count(Ag: NDArray) -> int:

    return Ag[0, :].max()


@cuda.jit(device=True)
def calc_sepset_indices(neighbour_count, sepset_index, sepset, level):
    sepset[0] = 0
    s = 0
    while s <= sepset_index:
        sepset[0] += 1
        s += binomial_coeff(neighbour_count - sepset[0], level - 1)
    s -= binomial_coeff(neighbour_count - sepset[0], level - 1)
    for c in range(1, level):
        sepset[c] = sepset[c - 1]
        while s <= sepset_index:
            sepset[c] += 1
            s += binomial_coeff(neighbour_count - sepset[c], level - (c + 1))
        s -= binomial_coeff(neighbour_count - sepset[c], level - (c + 1))


@cuda.jit(device=True)
def resolve_sepset_indices(neighbours, sepset, vj, level):
    for i in range(level):
        idx = sepset[i] - 1
        idx += neighbours[idx] >= vj
        sepset[i] = neighbours[idx]
