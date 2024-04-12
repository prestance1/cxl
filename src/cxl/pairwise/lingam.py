from numpy.typing import NDArray
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from cxl.graph.graph_utils import create_empty_graph
from tqdm import tqdm


class LingamLearner:
    """
    Implements the Linear Non-Gaussian Acyclic Model (LiNGAM) algorithm for causal discovery.

    Attributes:
        verbose (bool): Whether to display progress information.
    """

    def __init__(self, verbose=False) -> None:
        """
        Initializes the LingamLearner.

        Args:
            verbose (bool, optional): Whether to display progress information. Defaults to False.
        """
        self.verbose = verbose

    def fit(self, observations: NDArray, threshold: float) -> NDArray:
        """
        Fit the LiNGAM model to the given observations.

        Args:
            observations (NDArray): Observational data.
            threshold (float): Threshold for determining causal connections.

        Returns:
            NDArray: Causal graph indicating causal relationships.
        """
        K = []
        _, p = observations.shape
        U = set(range(p))
        X = observations.copy()
        X = scale(X)
        for _ in (t := tqdm(range(p - 1), disable=not self.verbose)):
            t.set_description("finding min entropy...")
            m = min(U, key=lambda xj: _kernel(xj, U, X))
            t.set_description("updating residuals...")
            X = _update_residuals(m, U, X)
            K.append(m)
            U.remove(m)
        last = next(iter(U))
        K.append(last)
        causal_graph = np.zeros((p, p), dtype=np.float64)
        for i in range(1, p):
            coefs = regress_on_parents(K[i], K[:i], observations)
            causal_graph[K[:i], K[i]] = coefs

        causal_graph: NDArray = abs(causal_graph) > threshold
        return causal_graph.astype(np.int8)


def _update_residuals(m: int, U: set[int], X: NDArray) -> NDArray:
    """
    Update residuals based on the selected variable.

    Args:
        m (int): Selected variable.
        U (set[int]): Set of remaining variables.
        X (NDArray): Observational data.

    Returns:
        NDArray: Updated residuals.
    """
    R = X.copy()
    for i in U - {m}:
        R[:, i] = _residual(X[:, i], X[:, m])
    return R


def regress_on_parents(var: int, parents: list[int], X: NDArray) -> NDArray:
    """
    Estimate connection strengths using linear regression.

    Args:
        var (int): Target variable.
        parents (list[int]): Parent variables.
        X (NDArray): Observational data.

    Returns:
        NDArray: Estimated connection strengths.
    """
    model = LinearRegression()
    model.fit(X[:, parents], X[:, var])
    return model.coef_


def _kernel(xj: int, U: set[int], X: np.ndarray) -> float:
    """
    Compute the kernel function.

    Args:
        xj (int): Variable index.
        U (set[int]): Set of remaining variables.
        X (np.ndarray): Observational data.

    Returns:
        float: Kernel value.
    """
    T_kernel = sum(
        _mutual_information(X[:, xj], _residual(X[:, i], X[:, xj]), [2e-3, 0.5])
        for i in U
        if i != xj
    )
    return T_kernel


def _residual(xi: int, xj: int) -> NDArray:
    """
    Compute residuals.

    Args:
        xi: Variable.
        xj: Variable.

    Returns:
        Residuals.
    """
    return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj


def _mutual_information(x1: NDArray, x2: NDArray, param: list[float]):
    """
    Compute mutual information.

    Args:
        x1: Variable.
        x2: Variable.
        param: Parameters.

    Returns:
        Mutual information.
    """
    kappa, sigma = param
    n = len(x1)
    X1 = np.tile(x1, (n, 1))
    K1 = np.exp(-1 / (2 * sigma**2) * (X1**2 + X1.T**2 - 2 * X1 * X1.T))
    X2 = np.tile(x2, (n, 1))
    K2 = np.exp(-1 / (2 * sigma**2) * (X2**2 + X2.T**2 - 2 * X2 * X2.T))

    tmp1 = K1 + n * kappa * np.identity(n) / 2
    tmp2 = K2 + n * kappa * np.identity(n) / 2
    K_kappa = np.r_[np.c_[tmp1 @ tmp1, K1 @ K2], np.c_[K2 @ K1, tmp2 @ tmp2]]
    D_kappa = np.r_[
        np.c_[tmp1 @ tmp1, np.zeros([n, n])], np.c_[np.zeros([n, n]), tmp2 @ tmp2]
    ]

    sigma_K = np.linalg.svd(K_kappa, compute_uv=False)
    sigma_D = np.linalg.svd(D_kappa, compute_uv=False)

    return (-1 / 2) * (np.sum(np.log(sigma_K)) - np.sum(np.log(sigma_D)))
