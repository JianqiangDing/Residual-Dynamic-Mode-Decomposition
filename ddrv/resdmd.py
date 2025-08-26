"""
Residual Dynamic Mode Decomposition
Reference: Algorithm 2 in the paper "Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems"
"""

import numpy as np
import sympy as sp
from sklearn.cluster import KMeans


def resdmd(
    X,
    Y,
    observe_dict_params={"basis": "rbf", "size": 100},
):
    """
    Residual Dynamic Mode Decomposition, Algorithm 2

    This is the core ResDMD algorithm that computes eigenvalues, eigenvectors, and residuals for given trajectory data.


    ----------
    X : np.ndarray
        The input data matrix. (M, N) where M is the number of data points and N is the number of features
    Y : np.ndarray
        The output data matrix. (M, N) where M is the number of data points and N is the number of features
    observe_dict_params : dict
        The parameters for the observe function.
        "basis" : str
            The basis function for the observe function.
        "size" : int
            The size of the observe function.
    """
    # init the observe function dictionary
    if observe_dict_params["basis"] == "rbf":
        PX, PY, basis_functions, x_symbols = _rbf_observe_data(
            X, Y, observe_dict_params
        )
    else:
        raise ValueError(
            f"Observe function {observe_dict_params['basis']} not supported"
        )
    # solve the regularized least square problem with regularization parameter
    # Solve PX @ K ≈ PY in least-squares sense ⇒ K = argmin ||PX K - PY||_F
    K = np.linalg.lstsq(PX, PY, rcond=None)[0]

    # compute the eigenvalues and eigenvectors of the koopman operator
    # K V = V diag(LAM), V columns are eigenvectors
    LAM, V = np.linalg.eig(K)

    # compute residuals per Duffing_example.m:
    # res_j = || PY*v_j - PX*v_j*lam_j || / || PX*v_j ||
    PXV = PX @ V  # (n_samples, n_eigs)
    PYV = PY @ V  # (n_samples, n_eigs)
    numer = np.linalg.norm(PYV - PXV * LAM[None, :], axis=0)
    denom = np.linalg.norm(PXV, axis=0)

    residuals = numer / denom

    # Also return the dictionary describing the basis used
    return LAM, V, residuals, basis_functions, PX, PY, K, x_symbols


def _rbf_observe_data(X, Y, observe_dict_params):
    assert observe_dict_params["basis"] == "rbf"
    assert observe_dict_params["size"] > 0
    assert X.shape == Y.shape

    num_centers = observe_dict_params["size"]

    # find the RBF centers with kmeans (on input samples X of shape (n_samples, dim))
    kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(X)
    centers = kmeans.cluster_centers_  # (num_centers, dim)

    # compute a distance scale d using average point-to-center distances
    # Distances matrix: (n_samples, num_centers)
    diff = X[:, None, :] - centers[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    d = np.mean(D) + 1e-12  # avoid division by zero

    # build RBF feature matrices for all samples at once
    # PX[i, j] = exp(-||X_i - center_j|| / d)
    PX = np.exp(-D / d)

    diffY = Y[:, None, :] - centers[None, :, :]
    DY = np.linalg.norm(diffY, axis=2)
    PY = np.exp(-DY / d)

    # create a sympy symbol vector to represent the input variable
    x_dim = centers.shape[1]
    x_symbols = sp.symbols(f"x0:{x_dim}")
    x_vec = sp.Matrix(x_symbols)

    # build the basis function symbol expression list
    basis_expressions = []
    for j in range(num_centers):
        center_j = centers[j]
        # compute the symbol expression of ||x - center_j||
        diff_vec = x_vec - sp.Matrix(center_j)
        norm_expr = sp.sqrt(sum(diff_vec[i] ** 2 for i in range(x_dim)))
        # RBF basis function: exp(-||x - center_j|| / d)
        phi_j = sp.exp(-norm_expr / d)
        basis_expressions.append(phi_j)

    # return the matrix containing all the basis function symbol expressions
    basis_functions = sp.Matrix(basis_expressions)

    return PX, PY, basis_functions, x_symbols


def get_eigenpairs(PX, PY, num_eigenpairs):
    assert PX.shape == PY.shape, "PX and PY must have the same shape"
    # Shapes
    _, n_features = PX.shape
    num_keep = max(2, min(num_eigenpairs, n_features))

    # Work in feature space: X0, Y0 have shape (n_features, n_samples)
    X0 = PX.T
    Y0 = PY.T

    # Progressive feature reduction as in reference EDMD code
    Fphi = np.eye(n_features)
    Xtr = X0.copy()
    Ytr = Y0.copy()

    while Xtr.shape[0] > num_keep:
        # Koopman approximation in current subspace
        A = Ytr @ np.linalg.pinv(Xtr)
        # A = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]

        # Eigen-decomposition of A^T
        mu, phi = np.linalg.eig(A.T)

        # Residual for each eigenvector (Duffing-style in current subspace)
        # Build projections of eigenfunctions onto sample space
        PXV_cur = Xtr.T @ phi  # (n_samples, n_eigs_cur)
        PYV_cur = Ytr.T @ phi  # (n_samples, n_eigs_cur)
        numer_cur = np.linalg.norm(PYV_cur - PXV_cur * mu[None, :], axis=0)
        denom_cur = np.linalg.norm(PXV_cur, axis=0)
        res_eig = numer_cur / denom_cur

        # Sort by residual (ascending) and keep half (but not less than num_keep)
        order = np.argsort(res_eig)
        cur_dim = Xtr.shape[0]
        keep_dim = max(num_keep, cur_dim // 2)
        phi_keep = phi[:, order[:keep_dim]]

        # Update projection and transformed data
        Fphi = Fphi @ phi_keep
        Xtr = Fphi.T @ X0
        Ytr = Fphi.T @ Y0

    # Final reduced operator and eigen-decomposition
    A_final = Ytr @ np.linalg.pinv(Xtr)
    # A_final = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
    lam_final, phi_final = np.linalg.eig(A_final.T)

    # Rank eigenpairs again by residual in reduced space
    res_mat_final = phi_final.T @ (Ytr - A_final @ Xtr)
    res_eig_final = np.max(np.abs(res_mat_final), axis=1)
    order_final = np.argsort(res_eig_final)[:num_keep]

    lam_sel = lam_final[order_final]
    phi_sel = phi_final[:, order_final]

    # Map eigenfunctions to original feature space: V_full columns are eigenfunctions coefficients
    V_full = Fphi @ phi_sel  # (n_features, num_keep)

    # Compute Duffing-style residuals in the original sample space
    PXV = PX @ V_full  # (n_samples, num_keep)
    PYV = PY @ V_full  # (n_samples, num_keep)
    numer = np.linalg.norm(PYV - PXV * lam_sel[None, :], axis=0)
    denom = np.linalg.norm(PXV, axis=0)
    residuals = numer / denom

    return lam_sel, V_full, residuals
