"""
KoopPseudoSpecQR - Python implementation of Koopman operator pseudospectral analysis using QR decomposition
Based on main_routines/KoopPseudoSpecQR.m MATLAB implementation
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def koop_pseudospec_qr(PX, PY, z_pts, W, z_pts2=None, reg_param=1e-14):
    """
    Fast and accurate pseudospectrum computation using QR decomposition
    Based on MATLAB KoopPseudoSpecQR.m implementation

    Parameters:
        PX (ndarray): Current state feature matrix (n_samples × n_features)
        PY (ndarray): Next state feature matrix (n_samples × n_features)
        z_pts (ndarray): Complex points vector for pseudospectrum computation
        W (ndarray or scalar): Weight vector/scalar for data points
        z_pts2 (ndarray, optional): Complex points vector for pseudoeigenfunction computation
        reg_param (float): Regularization parameter

    Returns:
        RES: Residuals for shifts z_pts.
        RES2: Residuals for pseudoeigenfunctions corresponding to shifts z_pts2.
        V2: Pseudoeigenfunctions corresponding to shifts z_pts2.
    """

    # Convert W to vector if it's a scalar
    if np.isscalar(W):
        W = np.full(PX.shape[0], W)
    else:
        W = np.array(W).flatten()

    # Ensure W has correct length
    if len(W) != PX.shape[0]:
        raise ValueError(
            f"Weight vector length ({len(W)}) must match number of samples ({PX.shape[0]})"
        )

    # QR decomposition with weights (following MATLAB implementation)
    sqrt_W = np.sqrt(W)
    WPX = sqrt_W[:, np.newaxis] * PX  # Element-wise multiplication
    Q, R = np.linalg.qr(WPX, mode="reduced")

    # Compute C1 = (sqrt(W).*PY)/R
    WPY = sqrt_W[:, np.newaxis] * PY
    C1 = np.linalg.solve(R.T, WPY.T).T  # Equivalent to WPY @ inv(R)

    # Compute matrices
    L = C1.T @ C1
    G = np.eye(PX.shape[1])  # Identity matrix
    A = Q.T @ C1

    # Convert z_pts to column vector
    z_pts = np.array(z_pts).flatten()
    LL = len(z_pts)
    RES = np.zeros(LL)
    RES2 = None
    V2 = None

    # Compute pseudospectrum residuals
    if LL > 0:
        print(f"Computing pseudospectrum for {LL} points...")
        for jj in range(LL):
            if (jj + 1) % max(1, LL // 10) == 0:
                print(f"  Progress: {jj+1}/{LL}")

            z = z_pts[jj]

            # Build matrix L - z*A' - conj(z)*A + |z|^2*G
            matrix = L - z * A.T.conj() - np.conj(z) * A + (np.abs(z) ** 2) * G

            try:
                # Compute smallest eigenvalue
                eigenvals = np.linalg.eigvals(matrix)
                min_eigenval = np.min(np.real(eigenvals))
                RES[jj] = np.sqrt(max(0, min_eigenval))  # Ensure non-negative
            except Exception as e:
                print(f"Warning: Point {jj} computation failed: {e}")
                RES[jj] = np.inf

    # Compute pseudoeigenfunctions if z_pts2 is provided
    if z_pts2 is not None:
        z_pts2 = np.array(z_pts2).flatten()
        n_pts2 = len(z_pts2)
        RES2 = np.zeros(n_pts2)
        V2 = np.zeros((PX.shape[1], n_pts2), dtype=complex)

        print(f"Computing pseudoeigenfunctions for {n_pts2} points...")
        for jj in range(n_pts2):
            if (jj + 1) % max(1, n_pts2 // 10) == 0:
                print(f"  Progress: {jj+1}/{n_pts2}")

            z = z_pts2[jj]

            # Build matrix
            matrix = L - z * A.T.conj() - np.conj(z) * A + (np.abs(z) ** 2) * G

            try:
                # Compute smallest eigenvalue and corresponding eigenvector
                eigenvals, eigenvecs = np.linalg.eig(matrix)
                min_idx = np.argmin(np.real(eigenvals))
                V2[:, jj] = eigenvecs[:, min_idx]
                RES2[jj] = np.sqrt(max(0, np.real(eigenvals[min_idx])))
            except Exception as e:
                print(
                    f"Warning: Point {jj} pseudoeigenfunction computation failed: {e}"
                )
                V2[:, jj] = 0
                RES2[jj] = np.inf

        # Transform back using R^{-1} (following MATLAB: V2=R\V2)
        V2 = np.linalg.solve(R, V2)

    return RES, RES2, V2
