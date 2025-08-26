import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

import ddrv

if __name__ == "__main__":
    # define the dynamical system
    dynamical_system = ddrv.NL_EIG_System()

    np.random.seed(1)

    # visualize the vector field
    # ddrv.visualize_vector_field(dynamical_system.get_numerical_dynamics())

    DT = 0.01
    NUM_SAMPLES = 200
    NUM_STEPS = 50

    # generate the trajectory data
    traj = ddrv.generate_trajectory_data(
        dynamical_system,
        num_samples=NUM_SAMPLES,
        num_steps=NUM_STEPS,
        delta_t=DT,
        domain=[[-2, 2], [-2, 2]],
    )

    print("traj.shape", traj.shape)

    # visualize the generated trajectory data
    plt.figure(figsize=(12, 8))

    # plot all trajectories
    for i in range(traj.shape[1]):  # loop over each sample trajectory
        plt.plot(
            traj[:, i, 0],
            traj[:, i, 1],
            "-",
            alpha=0.6,
            linewidth=0.8,
            color="gray",
        )

    # mark the start and end points
    plt.scatter(
        traj[0, :, 0], traj[0, :, 1], c="green", s=20, alpha=0.7, label="start point"
    )
    plt.scatter(
        traj[-1, :, 0], traj[-1, :, 1], c="red", s=20, alpha=0.7, label="end point"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Generated Trajectory Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # from the trajectory data to construct the X and Y
    X = traj[:-1, :, :].reshape(-1, 2)
    Y = traj[1:, :, :].reshape(-1, 2)

    # visualize the X and Y
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c="blue", s=20, alpha=0.7, label="X")
    plt.scatter(Y[:, 0], Y[:, 1], c="red", s=20, alpha=0.7, label="Y")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("X and Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    # ------------------------------------------------------------
    # apply the ResDMD algorithm
    LAM, V, residuals, basis_functions, PX, PY, K, x_symbols = ddrv.resdmd(
        X,
        Y,
        observe_dict_params={
            "basis": "rbf",
            "size": 150,
        },
    )

    print("LAM", LAM.shape)
    print("V", V.shape)
    print("residuals", residuals.shape)

    print(residuals.min(), residuals.max())

    print("PX.shape", PX.shape)
    print("PY.shape", PY.shape)
    print("K.shape", K.shape)
    print("basis_functions.shape", basis_functions.shape)

    # ------------------------------------------------------------

    eig_lam, eig_v, eig_residuals = ddrv.get_eigenpairs(PX, PY, 4)

    print("eig_lam", eig_lam.shape)
    print("eig_v", eig_v.shape)
    print("eig_residuals", eig_residuals.shape)

    print("eig_lam", eig_lam)
    print("eig_residuals", eig_residuals)

    # ------------------------------------------------------------
    # from the computed discrete-time eigen values, compute the continuous-time eigen values
    eig_lam_ct = dynamical_system.get_eigenvalues()

    print("eig_lam_ct", eig_lam_ct)

    eig_lam_dt = np.exp(eig_lam_ct * DT)
    print("eig_lam_dt", eig_lam_dt)

    # now compare the real eigenfunction and the computed eigenfunction
    # now first reconstruct the computed eigenfunctions with the basis functions and the eigenvectors

    # set small values in eig_v to 0
    eig_v[np.abs(eig_v) < 1e-10] = 0

    # reconstruct the eigenfunctions
    phi_reconstructed = eig_v.T @ basis_functions

    print("phi_reconstructed", phi_reconstructed.shape)

    # ------------------------------------------------------------
    # visualize the first true eigenfunction in 3D on domain [-2,2]*[-2,2]

    def visualize_true_eigenfunction_3d(
        domain_x=(-2, 2), domain_y=(-2, 2), resolution=100
    ):
        """
        Visualize the 3D surface plot of the first true eigenfunction

        Args:
            domain_x: x-axis range
            domain_y: y-axis range
            resolution: grid resolution
        """
        # create the grid
        x_vals = np.linspace(domain_x[0], domain_x[1], resolution)
        y_vals = np.linspace(domain_y[0], domain_y[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

        # compute the first true eigenfunction psi1 = x[0]**2 + 2*x[1] + x[1]**3
        Z_grid = X_grid**2 + 2 * Y_grid + Y_grid**3

        # create the 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # plot the surface
        surf = ax.plot_surface(
            X_grid,
            Y_grid,
            Z_grid,
            cmap="viridis",
            alpha=0.8,
            linewidth=0,
            antialiased=True,
        )

        # add the contour projection to the bottom
        ax.contour(
            X_grid,
            Y_grid,
            Z_grid,
            zdir="z",
            offset=Z_grid.min() - 1,
            cmap="viridis",
            alpha=0.5,
        )

        # set the labels and title
        ax.set_xlabel("x₁", fontsize=12)
        ax.set_ylabel("x₂", fontsize=12)
        ax.set_zlabel("ψ₁(x₁,x₂)", fontsize=12)
        ax.set_title("The first true eigenfunction: ψ₁ = x₁² + 2x₂ + x₂³", fontsize=14)

        # add the color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        # set the view
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.show()

        return X_grid, Y_grid, Z_grid

    # call the visualization function
    print("Generating the 3D visualization of the first true eigenfunction...")
    X_grid, Y_grid, Z_grid = visualize_true_eigenfunction_3d()

    # also show some statistics
    print(
        f"The value range of the eigenfunction on the domain [-2,2]×[-2,2]: [{Z_grid.min():.3f}, {Z_grid.max():.3f}]"
    )

    # ------------------------------------------------------------
    # visualize specified reconstructed eigenfunction (simplified version)

    def visualize_reconstructed_eigenfunction_simple(
        eigenfunction_index=0, domain_x=(-2, 2), domain_y=(-2, 2), resolution=100
    ):
        """
        Visualize the reconstructed eigenfunction (similar to the true eigenfunction)

        Args:
            eigenfunction_index: eigenfunction index (0-based)
            domain_x: x-axis range
            domain_y: y-axis range
            resolution: grid resolution
        """
        if eigenfunction_index >= eig_v.shape[1]:
            print(
                f"Error: eigenfunction index {eigenfunction_index} out of range [0, {eig_v.shape[1]-1}]"
            )
            return

        print(
            f"Visualizing the {eigenfunction_index + 1}th reconstructed eigenfunction..."
        )

        # create the grid
        x_vals = np.linspace(domain_x[0], domain_x[1], resolution)
        y_vals = np.linspace(domain_y[0], domain_y[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

        # compute the value of the reconstructed eigenfunction on the grid
        # reshape the grid points to the sample format (n_samples, 2)
        points = np.column_stack([X_grid.flatten(), Y_grid.flatten()])

        # compute the value of all basis functions on these points (using the existing data PX)
        # note: PX is the basis function values computed on the training data, we need to recompute on the new grid points
        import sympy as sp

        phi_values = np.zeros((points.shape[0], len(basis_functions)))
        for i in range(len(basis_functions)):
            basis_expr = basis_functions[i]
            basis_func_numerical = sp.lambdify(x_symbols, basis_expr, "numpy")
            phi_values[:, i] = basis_func_numerical(points[:, 0], points[:, 1])

        # use the eigenvector weighted combination to get the reconstructed eigenfunction value
        Z_reconstructed = phi_values @ eig_v[:, eigenfunction_index]
        Z_reconstructed = Z_reconstructed.reshape(X_grid.shape)

        # create the 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # plot the surface
        surf = ax.plot_surface(
            X_grid,
            Y_grid,
            Z_reconstructed,
            cmap="plasma",
            alpha=0.8,
            linewidth=0,
            antialiased=True,
        )

        # add the contour projection to the bottom
        ax.contour(
            X_grid,
            Y_grid,
            Z_reconstructed,
            zdir="z",
            offset=Z_reconstructed.min()
            - 0.1 * abs(Z_reconstructed.max() - Z_reconstructed.min()),
            cmap="plasma",
            alpha=0.5,
        )

        # set the labels and title
        ax.set_xlabel("x₁", fontsize=12)
        ax.set_ylabel("x₂", fontsize=12)
        ax.set_zlabel(f"φ_{eigenfunction_index + 1}(x₁,x₂)", fontsize=12)

        eigenval_str = (
            f"λ = {eig_lam[eigenfunction_index]:.3f}"
            if eigenfunction_index < len(eig_lam)
            else ""
        )
        residual_str = (
            f"Residual = {eig_residuals[eigenfunction_index]:.2e}"
            if eigenfunction_index < len(eig_residuals)
            else ""
        )
        title = f"Reconstructed eigenfunction {eigenfunction_index + 1}: {eigenval_str}, {residual_str}"
        ax.set_title(title, fontsize=14)

        # add the color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        # set the view
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.show()

        # show the statistics
        print("Reconstructed eigenfunction statistics:")
        print(
            f"  Value range: [{Z_reconstructed.min():.3f}, {Z_reconstructed.max():.3f}]"
        )
        print(f"  Mean: {Z_reconstructed.mean():.3f}")
        print(f"  Standard deviation: {Z_reconstructed.std():.3f}")

        return X_grid, Y_grid, Z_reconstructed

    # visualize the reconstructed eigenfunction
    print("\n" + "=" * 60)
    print("Reconstructed eigenfunction visualization")
    print("=" * 60)

    # visualize the first reconstructed eigenfunction
    X_grid_1, Y_grid_1, Z_grid_1 = visualize_reconstructed_eigenfunction_simple(
        eigenfunction_index=0, domain_x=(-2, 2), domain_y=(-2, 2), resolution=100
    )

    # visualize the second reconstructed eigenfunction (if exists)
    if len(eig_lam) > 1:
        print(f"\n{'-'*40}")
        X_grid_2, Y_grid_2, Z_grid_2 = visualize_reconstructed_eigenfunction_simple(
            eigenfunction_index=1, domain_x=(-2, 2), domain_y=(-2, 2), resolution=100
        )

    # ------------------------------------------------------------
