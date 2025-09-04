# test the get_principal_eigenmodes function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

import ddrv

if __name__ == "__main__":
    # test the get_principal_eigenmodes function
    NL_EIG = ddrv.dynamic.NL_EIG(lambda1=-1.0, lambda2=2.5)
    equilibrium = [0, 0]
    radius = 0.001
    principal_eigenvalues_dt, principal_eigenvalues_ct = (
        ddrv.common.get_principal_eigenvalues(
            NL_EIG,
            equilibrium,
            radius,
            num_samples=1000,
            num_steps=10,
            dt=0.01,
        )
    )

    print("principal_eigenvalues", principal_eigenvalues_dt, principal_eigenvalues_ct)

    # generate the trajectory data
    DT = 0.05
    trajectory = ddrv.common.generate_trajectory_data(
        NL_EIG,
        num_samples=1000,
        num_steps=10,
        dt=DT,
        domain=[[-2, 2], [-2, 2]],
        random_seed=42,
    )

    # now split the trajectory data into X and Y arrays, X refers the current state, Y refers the next state
    X = trajectory[:-1, :, :].reshape(-1, 2)
    Y = trajectory[1:, :, :].reshape(-1, 2)
    print(X.shape, Y.shape)

    # get the principal eigenmodes
    principal_modes, observables, residuals = ddrv.common.get_principal_eigenmodes(
        X,
        Y,
        principal_eigenvalues_ct,
        DT,
        observe_params={"basis": "poly", "degree": 9},
    )

    print("principal_eigenvalues_ct (target):", principal_eigenvalues_ct)
    print("residuals", residuals)

    # now check the accuracy of the principal modes
    print(principal_modes.shape)
    print(observables.shape)
    print(residuals.shape)

    # now evaluate the principal modes on the grid samples
    domain = [[-3, 3], [-3, 3]]
    x = np.linspace(domain[0][0], domain[0][1], 100)
    y = np.linspace(domain[1][0], domain[1][1], 100)
    X, Y = np.meshgrid(x, y)
    grid_samples = np.vstack([X.flatten(), Y.flatten()]).T
    print(grid_samples.shape)

    # evaluate the principal modes on the grid samples
    Z_learned = observables.eval_mod(grid_samples, principal_modes)
    print(Z_learned.shape)

    # evaluate the ground true eigenfunctions on the grid samples
    eigF = NL_EIG.get_numerical_eigenfunctions()
    Z_true = eigF(*grid_samples.T).squeeze().T
    print(Z_true.shape)

    # estimate the scaling factor for the first principal mode
    scaling_factor_1 = ddrv.common.estimate_scaling_factor(
        Z_true[:, 1], Z_learned[:, 0]
    )
    print("scaling_factor_1", scaling_factor_1)
    # estimate the scaling factor for the second principal mode
    scaling_factor_2 = ddrv.common.estimate_scaling_factor(
        Z_true[:, 0], Z_learned[:, 1]
    )
    print("scaling_factor_2", scaling_factor_2)

    # compute the relative errors by considering the scaling factor for the first principal mode
    relative_errors_1 = np.abs(
        Z_true[:, 1] - scaling_factor_1 * Z_learned[:, 0]
    ) / np.abs(Z_true[:, 1])
    print("relative_errors_1.shape:", relative_errors_1.shape)
    print("max relative error 1:", np.max(relative_errors_1))
    # compute the relative errors by considering the scaling factor for the second principal mode
    relative_errors_2 = np.abs(
        Z_true[:, 0] - scaling_factor_2 * Z_learned[:, 1]
    ) / np.abs(Z_true[:, 0])
    print("relative_errors_2.shape:", relative_errors_2.shape)
    print("max relative error 2:", np.max(relative_errors_2))

    # visualize the distribution of relative errors in two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scatter1 = ax1.scatter(
        grid_samples[:, 0], grid_samples[:, 1], c=relative_errors_1, cmap="viridis"
    )
    ax1.set_title(
        "Relative Errors of Principal Mode  $\lambda_1 = {:.3f}$: ".format(
            principal_eigenvalues_ct[0]
        )
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(scatter1, ax=ax1)
    scatter2 = ax2.scatter(
        grid_samples[:, 0], grid_samples[:, 1], c=relative_errors_2, cmap="viridis"
    )
    ax2.set_title(
        "Relative Errors of Principal Mode  $\lambda_2 = {:.3f}$: ".format(
            principal_eigenvalues_ct[1]
        )
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(scatter2, ax=ax2)
    plt.tight_layout()
    plt.show()
