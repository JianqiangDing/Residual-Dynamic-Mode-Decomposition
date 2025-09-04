"""
Residual Dynamic Mode Decomposition
Reference: Algorithm 2 in the paper "Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems"
"""

import numpy as np
import sympy as sp

from ..common.transform_data import transform_data


def resdmd(X, Y, observe_params={"basis": "poly", "degree": 2}):
    PX, PY, observables = transform_data(X, Y, observe_params)
    K = np.linalg.lstsq(PX, PY, rcond=None)[0]

    # compute the eigenvalues and eigenvectors of the koopman operator
    LAM, V = np.linalg.eig(K)

    # compute the residuals
    # residuals = || PY*v_j - PX*v_j*lam_j || / || PX*v_j ||
    print("PX.shape, V.shape", PX.shape, V.shape)
    PXV = PX @ V
    PYV = PY @ V

    residuals = np.linalg.norm(PYV - PXV * LAM[None, :], axis=0) / np.linalg.norm(
        PXV, axis=0
    )

    return LAM, V, residuals, observables, PX, PY, K
