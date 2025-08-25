"""
ResDMD Algorithm 1 - Core implementation of Residual Dynamic Mode Decomposition
Based on the algorithm shown in MATLAB examples and research papers
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def resdmd_algorithm(PX, PY, reg_param=1e-14):
    """
    Residual Dynamic Mode Decomposition Algorithm 1

    This is the core ResDMD algorithm that computes eigenvalues, eigenvectors,
    and residuals for a given feature matrix representation.

    Algorithm:
    1. Solve K = PX \ PY (Koopman operator approximation)
    2. Compute eigenvalues and eigenvectors: K * V = V * Λ
    3. Calculate residuals: ||PY * V - PX * V * Λ|| / ||PX * V||

    Parameters:
        PX (ndarray): Current state feature matrix (n_samples × n_features)
        PY (ndarray): Next state feature matrix (n_samples × n_features)
        reg_param (float): Regularization parameter for numerical stability

    Returns:
        LAM (ndarray): Eigenvalues of the Koopman operator
        V (ndarray): Eigenvectors of the Koopman operator
        residuals (ndarray): Residuals for each eigenvalue/eigenvector pair
        K (ndarray): Koopman operator matrix
    """

    print("Running ResDMD Algorithm 1...")

    # Check input dimensions
    if PX.shape != PY.shape:
        raise ValueError(
            f"PX and PY must have same shape. Got PX: {PX.shape}, PY: {PY.shape}"
        )

    n_samples, n_features = PX.shape
    print(f"Feature matrices: {n_samples} samples × {n_features} features")

    # Step 1: Compute Koopman operator K = PX \ PY
    # This solves PX * K = PY in the least squares sense
    print("Computing Koopman operator approximation...")

    # Add regularization for numerical stability
    try:
        if reg_param > 0:
            # Regularized least squares: (PX^T * PX + reg * I) * K = PX^T * PY
            PX_reg = np.concatenate(
                [PX, np.sqrt(reg_param) * np.eye(n_features)], axis=0
            )
            PY_reg = np.concatenate([PY, np.zeros((n_features, n_features))], axis=0)
            K = np.linalg.lstsq(PX_reg, PY_reg, rcond=None)[0]
        else:
            # Standard least squares
            K = np.linalg.lstsq(PX, PY, rcond=None)[0]
    except np.linalg.LinAlgError:
        print("Warning: SVD did not converge, using pseudoinverse method...")
        # Fallback to Moore-Penrose pseudoinverse
        K = np.linalg.pinv(PX) @ PY

    print(f"Koopman operator shape: {K.shape}")

    # Step 2: Compute eigendecomposition of K
    print("Computing eigenvalues and eigenvectors...")
    LAM, V = np.linalg.eig(K)

    print(f"Found {len(LAM)} eigenvalues")
    print(f"Real eigenvalues: {np.sum(np.abs(np.imag(LAM)) < 1e-12)}")
    print(f"Complex eigenvalues: {np.sum(np.abs(np.imag(LAM)) >= 1e-12)}")

    # Step 3: Compute residuals
    print("Computing residuals...")
    residuals = np.zeros(len(LAM))

    for i in range(len(LAM)):
        # Compute PX * V_i and PY * V_i
        PX_Vi = PX @ V[:, i]
        PY_Vi = PY @ V[:, i]

        # Predicted next state: PX * V_i * λ_i
        PY_pred_i = PX_Vi * LAM[i]

        # Residual: ||PY * V_i - PX * V_i * λ_i|| / ||PX * V_i||
        numerator = np.linalg.norm(PY_Vi - PY_pred_i)
        denominator = np.linalg.norm(PX_Vi)

        if denominator > 1e-12:
            residuals[i] = numerator / denominator
        else:
            residuals[i] = np.inf

    # Sort eigenvalues by residuals (ascending order - best first)
    sort_indices = np.argsort(residuals)
    LAM = LAM[sort_indices]
    V = V[:, sort_indices]
    residuals = residuals[sort_indices]

    print("Residual statistics:")
    print(f"  Min residual: {np.min(residuals):.6e}")
    print(f"  Max residual: {np.max(residuals):.6e}")
    print(f"  Mean residual: {np.mean(residuals):.6e}")
    print(f"  Median residual: {np.median(residuals):.6e}")

    # Count "good" eigenvalues (low residuals)
    good_threshold = 1e-2
    good_eigs = np.sum(residuals < good_threshold)
    print(f"Eigenvalues with residual < {good_threshold}: {good_eigs}")

    return LAM, V, residuals, K


def resdmd_with_rbf_features(X, Y, n_centers=100, rbf_function=None, reg_param=1e-14):
    """
    Complete ResDMD analysis with automatic RBF feature construction

    This function combines feature matrix construction with the core ResDMD algorithm.

    Parameters:
        X (ndarray): Current state data matrix (n_dims × n_samples)
        Y (ndarray): Next state data matrix (n_dims × n_samples)
        n_centers (int): Number of RBF centers
        rbf_function (callable): RBF function, default is exp(-r)
        reg_param (float): Regularization parameter

    Returns:
        LAM (ndarray): Eigenvalues
        V (ndarray): Eigenvectors
        residuals (ndarray): Residuals
        K (ndarray): Koopman operator
        PX (ndarray): Feature matrix for current states
        PY (ndarray): Feature matrix for next states
        centers (ndarray): RBF centers
    """

    print("ResDMD with RBF feature construction")
    print("=" * 50)

    # Default RBF function
    if rbf_function is None:

        def rbf_function(r):
            return np.exp(-r)

    # Check input dimensions
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have same shape. Got X: {X.shape}, Y: {Y.shape}"
        )

    n_dims, n_samples = X.shape
    print(f"State data: {n_dims} dimensions × {n_samples} samples")

    # Scaling for RBF function
    X_mean = np.mean(X, axis=1, keepdims=True)
    d = np.mean(np.linalg.norm(X - X_mean, axis=0))

    # Handle edge case where d is zero or NaN
    if d == 0 or np.isnan(d):
        d = 1.0
        print("Warning: RBF scaling parameter d was zero or NaN, setting to 1.0")

    print(f"RBF scaling parameter d = {d:.6f}")

    # Find RBF centers using k-means clustering
    print(f"Computing k-means clustering for {n_centers} centers...")
    try:
        from sklearn.cluster import KMeans

        data_combined = np.concatenate([X.T, Y.T], axis=0)
        kmeans = KMeans(n_clusters=n_centers, random_state=1, n_init=10)
        kmeans.fit(data_combined)
        centers = kmeans.cluster_centers_.T  # Shape: (n_dims, n_centers)
    except ImportError:
        # Fallback: random sampling of data points as centers
        print("scikit-learn not available, using random centers...")
        indices = np.random.choice(n_samples, n_centers, replace=False)
        centers = X[:, indices]

    print(f"RBF centers shape: {centers.shape}")

    # Build feature matrices
    print("Building RBF feature matrices...")
    PX = np.zeros((n_samples, n_centers))
    PY = np.zeros((n_samples, n_centers))

    for j in range(n_centers):
        # Distances from j-th center
        R_X = np.linalg.norm(X - centers[:, j : j + 1], axis=0)
        R_Y = np.linalg.norm(Y - centers[:, j : j + 1], axis=0)

        # Apply RBF function
        PX[:, j] = rbf_function(R_X / d)
        PY[:, j] = rbf_function(R_Y / d)

    print(f"Feature matrices built: {PX.shape}")

    # Apply core ResDMD algorithm
    LAM, V, residuals, K = resdmd_algorithm(PX, PY, reg_param)

    return LAM, V, residuals, K, PX, PY, centers


def analyze_eigenvalues(LAM, residuals, true_eigenvalues=None):
    """
    Analyze and summarize eigenvalue results

    Parameters:
        LAM (ndarray): Computed eigenvalues
        residuals (ndarray): Corresponding residuals
        true_eigenvalues (list, optional): Known true eigenvalues for comparison

    Returns:
        dict: Analysis results
    """

    print("\nEigenvalue Analysis")
    print("=" * 30)

    analysis = {
        "total_eigenvalues": len(LAM),
        "real_eigenvalues": np.sum(np.abs(np.imag(LAM)) < 1e-12),
        "complex_eigenvalues": np.sum(np.abs(np.imag(LAM)) >= 1e-12),
        "min_residual": np.min(residuals),
        "max_residual": np.max(residuals),
        "mean_residual": np.mean(residuals),
        "median_residual": np.median(residuals),
    }

    print(f"Total eigenvalues: {analysis['total_eigenvalues']}")
    print(f"Real eigenvalues: {analysis['real_eigenvalues']}")
    print(f"Complex eigenvalues: {analysis['complex_eigenvalues']}")
    print(
        f"Residual range: [{analysis['min_residual']:.2e}, {analysis['max_residual']:.2e}]"
    )
    print(f"Mean residual: {analysis['mean_residual']:.2e}")

    # Analyze residual thresholds
    thresholds = [1e-1, 1e-2, 1e-3, 1e-4]
    for threshold in thresholds:
        count = np.sum(residuals < threshold)
        print(f"Eigenvalues with residual < {threshold:.0e}: {count}")
        analysis[f"good_eigs_1e{int(np.log10(threshold))}"] = count

    # Compare with true eigenvalues if provided
    if true_eigenvalues is not None:
        print(f"\nComparison with {len(true_eigenvalues)} true eigenvalues:")
        analysis["true_comparisons"] = []

        for i, true_eig in enumerate(true_eigenvalues):
            distances = np.abs(LAM - true_eig)
            closest_idx = np.argmin(distances)
            closest_eig = LAM[closest_idx]

            comparison = {
                "true_value": true_eig,
                "closest_computed": closest_eig,
                "distance": distances[closest_idx],
                "residual": residuals[closest_idx],
            }

            print(f"  True λ_{i+1} = {true_eig}")
            print(f"    Closest computed: {closest_eig:.6f}")
            print(f"    Distance: {comparison['distance']:.6f}")
            print(f"    Residual: {comparison['residual']:.6f}")

            analysis["true_comparisons"].append(comparison)

    return analysis
