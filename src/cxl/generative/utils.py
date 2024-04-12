import torch
import numpy as np
from numpy.typing import NDArray


def encode_onehot(labels: NDArray) -> NDArray:
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def matrix_poly(matrix: NDArray, d: int) -> NDArray:
    x = torch.eye(d).double() + torch.div(matrix, d)
    return torch.matrix_power(x, d)


def nll_gaussian(
    preds: NDArray, target: NDArray, variance: float, add_const: bool = False
) -> float:
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(
        torch.pow(mean1 - mean2, 2), 2.0 * np.exp(2.0 * variance)
    )
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def kl_gaussian_sem(preds: NDArray) -> float:
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0))) * 0.5
