import numpy as np
from numpy.typing import NDArray
import math
from scipy.stats import norm
from typing import Iterable


class FisherZTest:

    def __init__(self, observations: NDArray, alpha: float) -> None:
        self.corr_matrix = np.corrcoef(observations.T)
        self.alpha = alpha
        self.size = observations.shape[0]

    def is_conditionally_independent(self, x: int, y: int, z: Iterable[int]) -> bool:
        var = [x, y] + list(z)
        sub_corr_matrix = self.corr_matrix[np.ix_(var, var)]
        inv = np.linalg.inv(sub_corr_matrix)
        r = -inv[0, 1] / math.sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1:
            r = (1.0 - np.finfo(float).eps) * np.sign(r)
        Z = 0.5 * math.log((1 + r) / (1 - r))
        X = math.sqrt(self.size - len(z) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))

        return p > self.alpha

    def is_independent(self, x: int, y: int):
        return self.is_conditionally_independent(x, y, [])


if __name__ == "__main__":
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    data = np.vstack([x, y]).T
    tester = FisherZTest(data, 0.05)
    print(tester.is_independent(0, 1))
