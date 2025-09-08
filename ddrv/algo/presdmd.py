"""
Principal Residual Dynamic Mode Decomposition
"""

import numpy as np
from scipy.linalg import qr

from ..common.transform_data import transform_data


def presdmd(
    trajectory,  # the trajectory data (num_steps, num_samples, dimension)
    k,
    dt,  # time step for generating the given trajectory data
    observe_params={"basis": "poly", "degree": 2},
    thresh_res=1e-4,  # threshold for the residuals for selecting the principal eigenmodes
    thresh_ct=1e-10,  # threshold for the continuous eigenvalues for selecting the principal eigenmodes
    eps=1e-10,  # epsilon for regulation
):
    """
    Principal Residual Dynamic Mode Decomposition

    CAUTION: we assume that there is no constant eigenmode in the system. so in this impolmentation, we will not consider eigenmodes with small continuous eigenvalues.
    """

    X = trajectory[:-1, :, :].reshape(-1, trajectory.shape[2])
    Y = trajectory[1:, :, :].reshape(-1, trajectory.shape[2])

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

    # select the principal eigenmodes based on the residuals, and the continuous eigenvalues are not quite small
    LAM_ct = np.log(LAM) / dt
    idx = np.where((residuals < thresh_res) & (np.abs(LAM_ct) > thresh_ct))[0]
    LAM = LAM[idx]

    V = V[:, idx]

    # now we using qr decomposition to get the principal eigenmodes
    # first we get all data points from the trajectory data
    D = trajectory.reshape(-1, trajectory.shape[2])
    # transform the data with the observe function
    PD, _ = transform_data(D, observe_params)

    # compute the inner-product matrix of observables on the trajectory data
    # quadrature weights
    W = np.eye(PD.shape[0]) * 1 / PD.shape[0]
    IP = PD.T @ W @ PD + eps * np.eye(
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

    # with the selected indices, we can get the principal eigenmodes
    V = V[:, idx]
    LAM = LAM[idx]

    return LAM, V, residuals[idx], observables, PX, PY, K
