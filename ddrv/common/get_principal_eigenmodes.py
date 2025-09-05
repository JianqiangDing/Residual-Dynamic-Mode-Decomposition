# this function is used to get the principal eigenmodes of the koopman operator


import numpy as np

from ..algo.resdmd import resdmd
from .find_closet_subset_index import find_closet_subset_index


def get_principal_eigenmodes(
    X, Y, LP_ct, dt, observe_params={"basis": "poly", "degree": 2}
):
    """
    Get the principal eigenmodes of the koopman operator

    X: the input data for current state (num_samples, num_dim)
    Y: the output data for next state (num_samples, num_dim)
    LP_ct: the principal eigenvalues in the continuous time
    dt: the time step for generating the trajectory data (X,Y)
    observe_params: the parameters for the observe function for residual evaluation

    Returns:
    the principal eigenmodes,observables,corresponding residuals
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of dimensions"
    assert X.shape[1] == len(
        LP_ct
    ), "X and LP_ct must have the same number of dimensions"

    L_dt, V, residuals, observables, _, _, _ = resdmd(X, Y, observe_params)
    # compute the corresponding eigenvalues in continuous time
    L_ct = np.log(L_dt) / dt

    # get the principal eigenmodes by absolute error between the continuous eigenvalues and the principal eigenvalues
    # caution here, L_ct and LP_ct are in different sizes, so we need to evaluate all corresponding errors
    principal_idx = find_closet_subset_index(LP_ct, L_ct, lambda x, y: np.abs(x - y))
    # get the most closest eigenvalues to the given principal eigenvalues
    principal_modes = V[:, principal_idx].T

    print(L_ct[principal_idx], "L_ct[principal_idx]")

    return principal_modes, observables, residuals[principal_idx]
