"""
NL_EIG_reach.py - Nonlinear Dynamical System Definition (Python Version)

Defines a two-dimensional nonlinear dynamical system:
dx/dt = [∇Ψ(x)]^(-1) * diag([-1, 2.5]) * Ψ(x)

where the principal eigenfunctions are:
ψ₁(x) = x₁² + 2x₂ + x₂³
ψ₂(x) = x₁ + sin(x₂) + x₁³

and the associated eigenvalues are:
λ₁ = -1, λ₂ = 2.5
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


def define_nonlinear_system():
    """
    Define the nonlinear dynamical system

    Returns:
        dynamics: Function handle for the dynamical system
    """
    # System parameters
    lambda1 = -1.0
    lambda2 = 2.5
    Lambda = np.diag([lambda1, lambda2])

    def psi(x):
        """Principal eigenfunctions Ψ(x)"""
        return np.array(
            [
                x[0] ** 2 + 2 * x[1] + x[1] ** 3,  # ψ₁(x)
                x[0] + np.sin(x[1]) + x[0] ** 3,  # ψ₂(x)
            ]
        )

    def nabla_psi(x):
        """Gradient matrix ∇Ψ(x)"""
        return np.array(
            [
                [2 * x[0], 2 + 3 * x[1] ** 2],  # ∂ψ₁/∂x₁, ∂ψ₁/∂x₂
                [1 + 3 * x[0] ** 2, np.cos(x[1])],  # ∂ψ₂/∂x₁, ∂ψ₂/∂x₂
            ]
        )

    def dynamics(t, x):
        """Dynamical system: dx/dt = [∇Ψ(x)]^(-1) * Λ * Ψ(x)"""
        try:
            grad_psi = nabla_psi(x)
            psi_val = psi(x)
            return np.linalg.solve(grad_psi, Lambda @ psi_val)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            return np.array([0.0, 0.0])

    return dynamics


def visualize_vector_field(dynamics, domain=[-2, 2, -2, 2], step_size=0.1):
    """
    Visualize the vector field using streamlines

    Args:
        dynamics: Function handle for the dynamical system
        domain: [xmin, xmax, ymin, ymax] domain bounds
        step_size: Grid step size for vector field computation
    """
    xmin, xmax, ymin, ymax = domain

    # Create grid
    x = np.arange(xmin, xmax + step_size, step_size)
    y = np.arange(ymin, ymax + step_size, step_size)
    X, Y = np.meshgrid(x, y)

    # Compute velocity field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                dxdt = dynamics(0, [X[i, j], Y[i, j]])
                U[i, j] = dxdt[0]
                V[i, j] = dxdt[1]
            except:
                U[i, j] = np.nan
                V[i, j] = np.nan

    # Create figure
    plt.figure(figsize=(12, 9))

    # Plot streamlines
    start_points = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
    start_points = np.array([start_points[0].flatten(), start_points[1].flatten()]).T

    plt.streamplot(
        X, Y, U, V, start_points=start_points, color="blue", linewidth=0.8, density=1.5
    )

    # Add contour lines for velocity magnitude
    speed = np.sqrt(U**2 + V**2)
    plt.contour(
        X, Y, speed, levels=8, colors="gray", linestyles="--", linewidths=0.5, alpha=0.7
    )

    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Vector Field of Nonlinear Dynamical System (Streamlines)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    print(f"Vector field visualized with streamlines in domain {domain}")
    print(f"Grid step size: {step_size:.3f}")

    plt.tight_layout()
    plt.show()


def generate_trajectory_data(dynamics):
    """
    Generate trajectory data for ResDMD analysis

    Args:
        dynamics: Function handle for the dynamical system

    Returns:
        X: Current state data matrix (2 × M)
        Y: Next state data matrix (2 × M)
    """
    np.random.seed(1)  # For reproducible results

    # Parameters
    M1 = 2000  # Number of initial conditions
    M2 = 20  # Number of time steps per trajectory
    delta_t = 0.01  # Time step

    print(f"Generating {M1} trajectories with {M2} time steps each...")

    X_list = []
    Y_list = []

    for jj in range(M1):
        # Random initial condition in domain [-1.5, 1.5] × [-1.5, 1.5]
        Y0 = (np.random.rand(2) - 0.5) * 3

        try:
            # Time points
            t_span = (0, (3 + M2) * delta_t)
            t_eval = np.concatenate([[0, 0.000001], np.arange(1, 4 + M2) * delta_t])

            # Solve ODE
            sol = solve_ivp(
                dynamics,
                t_span,
                Y0,
                t_eval=t_eval,
                rtol=1e-12,
                atol=1e-12,
                method="RK45",
            )

            if sol.success and len(sol.y[0]) >= 3 + M2:
                Y1 = sol.y  # (2 × N_points)

                # Collect data pairs (current state, next state)
                # Skip index 1 (t=0.000001) for numerical stability
                X_current = Y1[:, np.concatenate([[0], np.arange(2, 2 + M2)])]
                Y_next = Y1[:, np.arange(2, 3 + M2)]

                X_list.append(X_current)
                Y_list.append(Y_next)

        except Exception as e:
            print(f"Skipping trajectory {jj + 1} due to numerical issues: {e}")

        # Progress indicator
        if (jj + 1) % 200 == 0:
            print(f"Completed {jj + 1}/{M1} trajectories")

    # Concatenate all data
    if X_list:
        X = np.concatenate(X_list, axis=1)
        Y = np.concatenate(Y_list, axis=1)
    else:
        X = np.array([[], []])
        Y = np.array([[], []])

    M = X.shape[1]
    print(f"Generated {M} data points")

    return X, Y


def analyze_with_resdmd(X, Y):
    """
    Apply ResDMD algorithm and visualize results

    Args:
        X: Current state data matrix (2 × M)
        Y: Next state data matrix (2 × M)
    """
    # Parameters
    N = 100  # Number of basis functions

    def phi(r):
        """Radial basis function"""
        return np.exp(-r)

    M = X.shape[1]

    if M == 0:
        print("No data available for ResDMD analysis")
        return

    # Scaling for radial function
    X_mean = np.mean(X, axis=1, keepdims=True)
    d = np.mean(np.linalg.norm(X - X_mean, axis=0))

    # Find centers using k-means
    print(f"Computing k-means clustering for {N} centers...")
    data_combined = np.concatenate([X.T, Y.T], axis=0)
    kmeans = KMeans(n_clusters=N, random_state=1, n_init=10)
    kmeans.fit(data_combined)
    C = kmeans.cluster_centers_

    # Build feature matrices
    print("Building feature matrices...")
    PX = np.zeros((M, N))
    PY = np.zeros((M, N))

    for j in range(N):
        # Distances from centers
        R_X = np.linalg.norm(X.T - C[j], axis=1)
        R_Y = np.linalg.norm(Y.T - C[j], axis=1)

        PX[:, j] = phi(R_X / d)
        PY[:, j] = phi(R_Y / d)

    # Apply ResDMD algorithm
    print("Computing Koopman operator approximation...")
    K = np.linalg.lstsq(PX, PY, rcond=None)[0]
    LAM, V = eig(K)

    # Compute residuals
    PY_pred = PX @ V @ np.diag(LAM)
    PX_V = PX @ V

    res = np.array(
        [
            (
                np.linalg.norm(PY_pred[:, i] - PY @ V[:, i])
                / np.linalg.norm(PX_V[:, i])
                if np.linalg.norm(PX_V[:, i]) > 1e-12
                else 0
            )
            for i in range(len(LAM))
        ]
    )

    # Visualize eigenvalues with residuals
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        np.real(LAM), np.imag(LAM), c=res, s=100, cmap="turbo", alpha=0.7
    )
    plt.colorbar(scatter, label="Residuals")
    plt.xlabel("Real(λ)")
    plt.ylabel("Imag(λ)")
    plt.title("ResDMD Eigenvalues colored by Residuals")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    # Add reference lines for true eigenvalues
    plt.axvline(x=-1, color="red", linestyle="--", alpha=0.7, label="λ₁ = -1")
    plt.axvline(x=2.5, color="red", linestyle="--", alpha=0.7, label="λ₂ = 2.5")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print analysis results
    print(f"\nResDMD Analysis Results:")
    print(f"Total eigenvalues found: {len(LAM)}")
    print(f"Real eigenvalues: {np.sum(np.abs(np.imag(LAM)) < 1e-6)}")
    print(f"Complex eigenvalues: {np.sum(np.abs(np.imag(LAM)) >= 1e-6)}")

    # Find eigenvalues closest to true values
    true_eigs = [-1, 2.5]
    for true_eig in true_eigs:
        distances = np.abs(LAM - true_eig)
        closest_idx = np.argmin(distances)
        closest_eig = LAM[closest_idx]
        print(
            f"Closest to λ = {true_eig}: {closest_eig:.4f} "
            f"(distance: {distances[closest_idx]:.4f}, residual: {res[closest_idx]:.4f})"
        )

    return LAM, V, res, K


if __name__ == "__main__":
    """Main execution function"""
    print("Nonlinear Dynamical System Analysis")
    print("Eigenvalues: λ₁ = -1, λ₂ = 2.5")
    print("System exhibits saddle-point behavior")

    # Define the nonlinear dynamical system
    dynamics = define_nonlinear_system()

    # Visualize the vector field
    print("\nVisualizing vector field...")
    visualize_vector_field(dynamics, domain=[-2, 2, -2, 2], step_size=0.1)

    # Generate trajectory data for ResDMD analysis
    print("\nGenerating trajectory data for ResDMD analysis...")
    X_data, Y_data = generate_trajectory_data(dynamics)

    # Visualize generated data
    plt.figure(figsize=(10, 8))
    plt.plot(
        X_data[0, :],
        X_data[1, :],
        "r.",
        alpha=0.5,
        markersize=2,
        label="Current states",
    )
    plt.plot(
        Y_data[0, :], Y_data[1, :], "b.", alpha=0.5, markersize=2, label="Next states"
    )
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Generated Trajectory Data")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Apply ResDMD algorithm
    print("\nApplying ResDMD algorithm...")
    LAM, V, res, K = analyze_with_resdmd(X_data, Y_data)

    print("\nAnalysis completed!")
