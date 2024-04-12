from numpy.typing import NDArray
import itertools
import numpy as np
from cxl.graph.graph_utils import create_empty_graph
from sklearn.gaussian_process import GaussianProcessRegressor
from cxl.independence.hsic import HSICGaussian
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel


class ANMLearner:
    """
    Implements the Additive Noise Model (ANM) algorithm for pairwise causal discovery.

    Attributes:
        alpha (float): Significance level for independence testing.
    """

    def __init__(self, alpha=0.05) -> None:
        """
        Initializes the ANMLearner.

        Args:
            alpha (float, optional): Significance level for independence testing. Defaults to 0.05.
        """
        self.alpha = alpha

    def fit(self, observations: NDArray):
        """
        Fit the ANM model to the given observations.

        Args:
            observations (NDArray): Observational data.

        Returns:
            NDArray: Causal graph indicating causal relationships.
        Raises:
            ValueError: If the number of variables in the observations is not 2.
        """
        _, n = observations.shape
        if n != 2:
            raise ValueError("ANM is a pairwise method")
        indep_tester = HSICGaussian(self.alpha)
        causal_graph = create_empty_graph(n)
        for x, y in itertools.combinations(range(n), 2):
            X = observations[:, x].reshape(-1, 1)
            Y = observations[:, y].reshape(-1, 1)
            # is_independent = cat(x, y, data)
            if not indep_tester.is_independent(X, Y):
                # regress y on x i.e y = f(x) + Ny
                f_x_hat = regress(X, Y)
                y_residuals = Y - f_x_hat
                if indep_tester.is_independent(X, y_residuals):
                    causal_graph[x, y] = 1
                # regress x on y i.e x = g(y) + Nx

                g_y_hat = regress(Y, X)
                x_residuals = X - g_y_hat
                if indep_tester.is_independent(Y, x_residuals):
                    causal_graph[y, x] = 1
        return causal_graph


def regress(X, y):
    """
    Fit a Gaussian process regression model

    Parameters
    ----------
    X: input data (nx1)
    y: output data (nx1)

    Returns
    --------
    pred_y: predicted output (nx1)
    """
    regressor = GaussianProcessRegressor(
        kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        + WhiteKernel(0.1, (1e-10, 1e1))
    )

    regressor.fit(X, y)
    pred_y = regressor.predict(X).reshape(-1, 1)
    return pred_y
