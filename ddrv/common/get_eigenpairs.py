# this function is used to get the principal eigenpairs of the koopman operator based on the residual method

import numpy as np


def get_eigenpairs(PX, PY, num_eigenpairs):
    assert PX.shape == PY.shape, "PX and PY must have the same shape"
    # Shapes
    _, n_features = PX.shape
    num_keep = max(2, min(num_eigenpairs, n_features))
