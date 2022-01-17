import math
from numpy.typing import ArrayLike


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    if len(a) != len(b):
        raise ValueError("Dimension of a != Dimension of b")
    sum_of_squares = 0
    for i in range(len(a)):
        sum_of_squares += (a[i] - b[i])**2
    return math.sqrt(sum_of_squares)
