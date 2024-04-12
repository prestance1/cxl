import numpy as np
from numpy.typing import NDArray
from scipy import stats

from scipy.spatial.distance import cdist, pdist, squareform


def center_kernel_matrix(K: NDArray):
    """
    Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
    [Updated @Haoyue 06/24/2022]
    equivalent to:
        H = eye(n) - 1.0 / n
        return H.dot(K.dot(H))
    since n is always big, we can save time on the dot product by plugging H into dot and expand as sum.
    time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element).
    Also, consider the fact that here K (both Kx and Ky) are symmetric matrices, so K_colsums == K_rowsums
    """
    # assert np.all(K == K.T), 'K should be symmetric'
    n = np.shape(K)[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n**2)


def HSIC_V_statistic(Kx, Ky):
    """
    Compute V test statistic from kernel matrices Kx and Ky
    Parameters
    ----------
    Kx: kernel matrix for data_x (nxn)
    Ky: kernel matrix for data_y (nxn)

    Returns
    _________
    Vstat: HSIC v statistics
    Kxc: centralized kernel matrix for data_x (nxn)
    Kyc: centralized kernel matrix for data_y (nxn)
    """
    Kxc = center_kernel_matrix(Kx)
    Kyc = center_kernel_matrix(Ky)
    V_stat = np.sum(Kxc * Kyc)
    return V_stat, Kxc, Kyc


def kernel(X: np.ndarray, Y: np.ndarray | None = None):
    """
    Computes the Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)=exp(-0.5* ||x-y||**2 *self.width)
    """
    width: float = get_width_empirical_hsic(X)
    if Y is None:
        sq_dists = squareform(pdist(X, "sqeuclidean"))
    else:
        assert np.shape(X)[1] == np.shape(Y)[1]
        sq_dists = cdist(X, Y, "sqeuclidean")
    K = np.exp(-0.5 * sq_dists * width)
    return K


def get_width_empirical_hsic(X: np.ndarray):
    n = X.shape[0]
    if n < 200:
        width = 0.8
    elif n < 1200:
        width = 0.5
    else:
        width = 0.3
    theta = 1.0 / (width**2)
    width = theta * X.shape[1]
    return width


def kernel_matrix(data_x, data_y):
    data_x = stats.zscore(data_x, ddof=1, axis=0)
    data_x[np.isnan(data_x)] = 0.0  # in case some dim of data_x is constant
    data_y = stats.zscore(data_y, ddof=1, axis=0)
    data_y[np.isnan(data_y)] = 0.0
    Kx = kernel(data_x)
    Ky = kernel(data_y)
    return Kx, Ky


def get_kappa(Kx, Ky):
    """
    Get parameters for the approximated gamma distribution
    Parameters
    ----------
    Kx: kernel matrix for data_x (nxn)
    Ky: kernel matrix for data_y (nxn)

    Returns
    _________
    k_appr, theta_appr: approximated parameters of the gamma distribution

    [Updated @Haoyue 06/24/2022]
    equivalent to:
        var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / T / T
    based on the fact that:
        np.trace(K.dot(K)) == np.sum(K * K.T), where here K is symmetric
    we can save time on the dot product by only considering the diagonal entries of K.dot(K)
    time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element),
    where n is usually big (sample size).
    """
    T = Kx.shape[0]
    mean_appr = np.trace(Kx) * np.trace(Ky) / T
    var_appr = (
        2 * np.sum(Kx**2) * np.sum(Ky**2) / T / T
    )  # same as np.sum(Kx * Kx.T) ..., here Kx is symmetric
    k_appr = mean_appr**2 / var_appr
    theta_appr = var_appr / mean_appr
    return k_appr, theta_appr


def compute_pvalue(data_x, data_y):
    Kx, Ky = kernel_matrix(data_x, data_y)
    test_stat, Kxc, Kyc = HSIC_V_statistic(Kx, Ky)
    k_appr, theta_appr = get_kappa(Kxc, Kyc)
    pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
    print(pvalue)
    return pvalue


class HSICGaussian:

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha

    def is_independent(self, x, y) -> bool:
        return compute_pvalue(x.reshape(-1, 1), y.reshape(-1, 1)) > self._alpha
