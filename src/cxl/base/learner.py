from abc import ABC
from numpy.typing import NDArray


class Learner(ABC):

    def fit(self, observations: NDArray, *args, **kwargs) -> NDArray:
        pass
