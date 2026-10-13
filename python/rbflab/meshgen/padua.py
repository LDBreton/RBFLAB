"""First-family Padua nodes on a reference square."""
import numpy as np


def padua_points(degree):
    """Return (n+1)(n+2)/2 Padua nodes on [-1, 1]^2 for degree n >= 1.

    Uses the first family in Koushki et al. (2020), Eq. (8),
    doi:10.1016/j.aej.2020.04.047. Ordering is by increasing cosine-grid
    index in x, then y. These are coordinates only, without connectivity.
    """
    if type(degree) is not int or degree < 1:
        raise ValueError("degree must be a positive integer")
    i, j = np.meshgrid(np.arange(degree+1), np.arange(degree+2), indexing="ij")
    keep = (i+j) % 2 == 1
    return np.column_stack((np.cos(np.pi*i[keep]/degree),
                            np.cos(np.pi*j[keep]/(degree+1))))
