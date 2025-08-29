# estimate the scaling factor of two given vectors

import numpy as np


def estimate_scaling_factor(a, b):
    """
    estimate the scaling factor of two given vectors
    """
    assert a.ndim == b.ndim == 1, "a and b must be 1D arrays"
    return np.dot(a, b) / np.dot(b, b)
