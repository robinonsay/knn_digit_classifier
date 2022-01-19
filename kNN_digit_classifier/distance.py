from numpy.typing import ArrayLike, NDArray
import numpy as np


def squared_euclidean(a: ArrayLike, b: ArrayLike) -> float:
    if type(a) is not NDArray:
        a = np.array(a)
    if type(b) is not NDArray:
        b = np.array(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape of a {a.shape} != Dimension of b {b.shape}")
    distance_squared = np.sum((a-b)**2)
    return distance_squared
