"""
Principal Residual Dynamic Mode Decomposition
"""

import numpy as np
from scipy.linalg import qr

from ..common.generate_trajectory_data import generate_trajectory_data
from ..common.transform_data import transform_data


def presdmd(
    dynamic,
    k,
    domain,
    dt,
    num_samples,
    num_steps,
    observe_params={"basis": "poly", "degree": 2},
    threshold=1e-4,  # threshold for the residuals for selecting the principal eigenmodes
    random_seed=None,
):
    """
    Principal Residual Dynamic Mode Decomposition

    CAUTION: we assume that there is no constant eigenmode in the system. so in this impolmentation, we will not consider eigenmodes with small continuous eigenvalues.
    """
    trajectory = generate_trajectory_data(
        dynamic, num_samples, num_steps, dt, domain, random_seed
    )
    X = trajectory[:-1, :, :].reshape(-1, 2)
    Y = trajectory[1:, :, :].reshape(-1, 2)
    PX, observables = transform_data(X, observe_params)
    PY, observables = transform_data(Y, observe_params)
    K = np.linalg.lstsq(PX, PY, rcond=None)[0]
    LAM, V = np.linalg.eig(K)

    # compute the residuals as defined in the ResDMD paper
    PXV = PX @ V
    PYV = PY @ V
    residuals = np.linalg.norm(PYV - PXV * LAM[None, :], axis=0) / np.linalg.norm(
        PXV, axis=0
    )

    # select the principal eigenmodes based on the residuals, and the continuous eigenvalues are not zero
    LAM_ct = np.log(LAM) / dt
    idx = np.where((residuals < threshold) & (np.abs(LAM_ct) > 1e-10))[0]
    LAM = LAM[idx]

    V = V[:, idx]

    # now we using qr decomposition to get the principal eigenmodes
    # first we get all data points from the trajectory data
    D = trajectory.reshape(-1, 2)
    # transform the data with the observe function
    PD, _ = transform_data(D, observe_params)

    # compute the inner-product matrix of observables on the trajectory data
    # quadrature weights
    W = np.eye(PD.shape[0]) * 1 / PD.shape[0]
    IP = PD.T @ W @ PD + 1e-10 * np.eye(
        PD.shape[1]
    )  # add a small regularization term to avoid numerical instability
    # cholesky decomposition of the observable inner-product matrix
    L = np.linalg.cholesky(IP)

    # transform the original eigenmodes with the L matrix
    TV = L.T.conj() @ V

    # now apply qr decomposition to the transformed eigenmodes
    Q, R, P = qr(TV, pivoting=True)
    # select the first k eigenmodes based on the singular values of R
    idx = P[:k]
    # print(idx, "idx")
    # print(P, "P")
    # with the selected indices, we can get the principal eigenmodes
    V = V[:, idx]
    LAM = LAM[idx]

    return LAM, V, residuals, observables, PX, PY, K
