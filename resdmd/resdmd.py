"""
Residual Dynamic Mode Decomposition
Reference: Algorithm 2 in the paper "Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems"
"""

import numpy as np
from sklearn.cluster import KMeans


def resdmd(
    X,
    Y,
    epsilon=1e-2,
    reg_param=1e-14,
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
    epsilon : float
        The threshold for the feasible residual.
    reg_param : float
        The regularization parameter for the least square problem for numerical stability.
    observe_dict_params : dict
        The parameters for the observe function.
        "basis" : str
            The basis function for the observe function.
        "size" : int
            The size of the observe function.
    """
    # init the observe function dictionary
    if observe_dict_params["basis"] == "rbf":
        PX, PY = _rbf_observe_data(X, Y, observe_dict_params)
    else:
        raise ValueError(
            f"Observe function {observe_dict_params['basis']} not supported"
        )
    # solve the regularized least square problem with regularization parameter
    K = np.linalg.lstsq(PX, PY, rcond=None)[0]

    # compute the eigenvalues and eigenvectors of the koopman operator
    LAM, V = np.linalg.eig(K)

    # compute the residuals
    residuals = np.linalg.norm(PY - K @ PX, axis=0)

    return LAM, V, residuals


def _rbf_observe_data(X, Y, observe_dict_params):
    assert observe_dict_params["basis"] == "rbf"
    assert observe_dict_params["size"] > 0
    assert X.shape == Y.shape

    num_centers = observe_dict_params["size"]

    # find the centers with kmeans
    kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    d = np.mean(np.linalg.norm(X - centers, axis=1))

    # define a vector-valued function phi(x) = [phi1(x), phi2(x), ..., phiN(x)]
    # for each phi_i(x), it is a radial basis function
    phi = lambda x: np.exp(-np.linalg.norm(x - centers, axis=1) / d)

    PX, PY = phi(X), phi(Y)

    return PX, PY
