import numpy as np
from numpy.typing import NDArray
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from typing import Callable
from tqdm import tqdm


NotearsLossFunction = Callable[[NDArray, NDArray], tuple[float, NDArray]]


def l2_loss(observations: NDArray, W: NDArray) -> tuple[float, NDArray]:
    """
    Computes the L2 loss and its gradient with respect to the weights matrix.

    Args:
        observations (NDArray): The input observations.
        W (NDArray): The weights matrix.

    Returns:
        tuple[float, NDArray]: A tuple containing the loss value and its gradient.
    """
    M = observations @ W
    R = observations - M
    loss = 0.5 / observations.shape[0] * (R**2).sum()
    G_loss = -1.0 / observations.shape[0] * observations.T @ R
    return loss, G_loss


def acyclicity_constraint(W: NDArray, d: int) -> tuple[float, NDArray]:
    """
    Computes the acyclicity constraint and its gradient with respect to the weights matrix.

    Args:
        W (NDArray): The weights matrix.
        d (int): The dimension.

    Returns:
        tuple[float, NDArray]: A tuple containing the constraint value and its gradient.
    """
    E = slin.expm(W * W)
    h = np.trace(E) - d
    G_h = E.T * W * 2
    return h, G_h


class NotearsLearner:
    """
        Adapted learner from : t https://github.com/xunzheng/
    notears.
        Implements the Notears algorithm for learning sparse graphs.

        Attributes:
            max_iter (int): Maximum number of iterations.
            h_tol (float): Tolerance for the acyclicity constraint.
            rho_max (float): Maximum value for the penalty parameter.
            threshold (float): Threshold for sparsity.
            loss_fn (NotearsLossFunction): Loss function to use.
            verbose (bool): Verbosity flag.
    """

    def __init__(
        self,
        max_iter=100,
        h_tol=1e-8,
        rho_max=1e16,
        threshold=0.05,
        loss_fn: NotearsLossFunction = l2_loss,
        verbose=False,
    ) -> None:
        """
        Initializes the NotearsLearner.

        Args:
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            h_tol (float, optional): Tolerance for the acyclicity constraint. Defaults to 1e-8.
            rho_max (float, optional): Maximum value for the penalty parameter. Defaults to 1e16.
            threshold (float, optional): Threshold for sparsity. Defaults to 0.05.
            loss_fn (NotearsLossFunction, optional): Loss function to use. Defaults to l2_loss.
            verbose (bool, optional): Verbosity flag. Defaults to False.
        """
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.verbose = verbose

    def fit(
        self,
        observations: NDArray,
        lambda1: float,
    ) -> NDArray:
        """
        Fits the model to the provided observations.
        https://github.com/xunzheng/notears.
        Args:
            observations (NDArray): The input observations.
            lambda1 (float): The regularization parameter.

        Returns:
            None
        """

        def _adj(w: NDArray):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[: d * d] - w[d * d :]).reshape([d, d])

        def _compute_gradient(w: NDArray) -> tuple[float, NDArray]:
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = self.loss_fn(observations, W)
            h, G_h = acyclicity_constraint(W, d)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
            return obj, g_obj

        def _dual_ascent(w_est, bounds):
            sol = sopt.minimize(
                _compute_gradient, w_est, method="L-BFGS-B", jac=True, bounds=bounds
            )
            w_new = sol.x
            h_new, _ = acyclicity_constraint(_adj(w_new), d)
            return w_new, h_new

        n, d = observations.shape
        w_est, rho, alpha, h = (
            np.zeros(2 * d * d),
            1.0,
            0.0,
            np.inf,
        )
        bounds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(d)
            for j in range(d)
        ]
        observations = observations - np.mean(observations, axis=0, keepdims=True)
        for i in (t := tqdm(range(self.max_iter), disable=not self.verbose)):
            t.set_description(f"running {i}th ascent")
            w_new, h_new = None, None
            while rho < self.rho_max:
                w_new, h_new = _dual_ascent(w_est, bounds)
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                break
        graph = _adj(w_est)
        graph[np.abs(graph) < self.threshold] = 0
        graph[np.nonzero(graph)] = 1
        return graph.astype(np.int8)
