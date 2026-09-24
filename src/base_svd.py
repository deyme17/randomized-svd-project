from abc import ABC, abstractmethod
import numpy as np


class ModelNotFittedException(Exception):
    """Raises when some method is called before a model is fitted."""
    pass



class BaseSVD(ABC):
    """Base abstract class for SVD implementations."""
    def __init__(self):
        self._X: np.ndarray | None = None
        self.rank: int | None = None
        self.U: np.ndarray | None = None
        self.S: np.ndarray | None = None
        self.V_T: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseSVD':
        """Fit a SVD model on input matrix X."""
        pass
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the origin matrix X using fitted U, S, V_T."""
        if any(x is None for x in (self.U, self.S, self.V_T)):
            raise ModelNotFittedException("Call `fit()` before `reconstruct()`.")
        return (self.U * self.S) @ self.V_T

    def reconstruction_error(self) -> float:
        """Relative Frobenius-norm error ||X - reconstruct()||_F / ||X||_F."""
        if self._X is None:
            raise RuntimeError("The original matrix X wasn't saved while `fit()`.")
        return float(
            np.linalg.norm(self._X - self.reconstruct(), ord="fro") /
            np.linalg.norm(self._X, ord="fro")
        )