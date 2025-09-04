# test the resdmd algorithm

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    # set the seed for reproducibility
    np.random.seed(42)

    DT = 0.05
    # define the dynamical system
    NL_EIG = ddrv.dynamic.NL_EIG(lambda1=-1.0, lambda2=2.5)
    # ddrv.viz.vis_vector_field_2d(
    #     NL_EIG.get_numerical_dynamics(), domain=[-2, 2, -2, 2], step_size=0.1
    # )

    # generate the trajectory data
    traj_data = ddrv.common.generate_trajectory_data(
        NL_EIG,
        num_samples=1000,
        num_steps=10,
        dt=DT,
        domain=[[-2, 2], [-2, 2]],
    )
    print(traj_data.shape)
    # traj_data is a 3D array, shape (num_steps, num_samples, dim)

    # based on the settings, the discrete eigenvalues are:
    dc_lambdas = np.exp(np.array([-1.0, 2.5]) * DT)
    print("dc_lambdas", dc_lambdas)

    # now split the trajectory data into X and Y arrays, X refers the current state, Y refers the next state
    X = traj_data[:-1, :, :].reshape(-1, 2)
    Y = traj_data[1:, :, :].reshape(-1, 2)
    print(X.shape, Y.shape)

    # now visualize the X and Y data
    # import matplotlib.pyplot as plt

    # plt.scatter(X[:, 0], X[:, 1], s=1, c="b", alpha=0.5)
    # plt.scatter(Y[:, 0], Y[:, 1], s=1, c="r", alpha=0.5)
    # plt.show()

    # now apply the resdmd algorithm
    L, V, residuals, observables, PX, PY, K = ddrv.algo.resdmd(
        X, Y, observe_params={"basis": "poly", "degree": 9}
    )
    print(
        L.shape,
        V.shape,
        residuals.shape,
        observables.shape,
        PX.shape,
        PY.shape,
        K.shape,
    )
    # get the first 6 eigenvalues with the smallest residuals
    idx = np.argsort(residuals)[:6]
    print(L[idx])
    # output: [1.+0.j 0.95122942+0.j 0.9048374 +0.j 1.13314874+0.j 1.28387519+0.j 1.07720766+0.j]

    print(residuals[idx])
    # output: [1.40101114e-14 3.35811836e-09 3.91625954e-09 2.55907388e-07 1.68192509e-06 9.79024351e-05]

    # there are 2 eigenvalues are quite close to the actual discrete eigenvalues, the index is 1 and 3
    principal_idx = idx[[1, 3]]
    print(principal_idx)
    principal_lambdas_dt = L[principal_idx]
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("principal_lambdas_dt", principal_lambdas_dt)
    principal_lambdas_ct = np.log(principal_lambdas_dt) / DT
    print("principal_lambdas_ct", principal_lambdas_ct)
    print("dc_lambdas", dc_lambdas)
    print("residuals", residuals[principal_idx])
    # get the corresponding eigenvectors
    principal_V = V[
        :, principal_idx
    ].T  # reshape it to (num_modes, num_of_basis_functions)

    # now check the accuracy of the extracted principal modes --------------------------------
    # get the grid samples on the domain [-2,2]*[-2,2]
    domain = [[-3, 3], [-3, 3]]

    # create the grid
    x = np.linspace(domain[0][0], domain[0][1], 100)
    y = np.linspace(domain[1][0], domain[1][1], 100)
    X, Y = np.meshgrid(x, y)
    grid_samples = np.vstack([X.flatten(), Y.flatten()]).T
    print(grid_samples.shape)

    # evaluate the principal modes on the grid samples
    Z_learned = observables.eval_mod(grid_samples, principal_V)
    print(Z_learned.shape)

    # evaluate the ground true eigenfunctions on the grid samples
    eigF = NL_EIG.get_numerical_eigenfunctions()
    Z_true = eigF(*grid_samples.T).squeeze().T
    print(Z_true.shape)

    # regarding the first principal mode, we can estimate the scaling factor
    scaling_factor_1 = ddrv.common.estimate_scaling_factor(
        Z_true[:, 0], Z_learned[:, 0]
    )
    print("scaling_factor_1", scaling_factor_1)
    # compute the relative absolute difference of the first principal mode
    error_1 = np.abs(Z_true[:, 0] - scaling_factor_1 * Z_learned[:, 0]) / np.abs(
        Z_true[:, 0]
    )
    print(np.max(error_1))

    # regarding the second principal mode, we can estimate the scaling factor
    scaling_factor_2 = ddrv.common.estimate_scaling_factor(
        Z_true[:, 1], Z_learned[:, 1]
    )
    print("scaling_factor_2", scaling_factor_2)
    # compute the relative absolute difference of the second principal mode
    error_2 = np.abs(Z_true[:, 1] - scaling_factor_2 * Z_learned[:, 1]) / np.abs(
        Z_true[:, 1]
    )
    print(np.max(error_2))

    # visualize the distribution of the errors on the domain in two subplots
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # first principal mode error
    scatter1 = ax1.scatter(X, Y, c=error_1, cmap="viridis")
    ax1.set_title(
        "Absolute Error of $\lambda_1 = {:.3f}$".format(
            np.real(principal_lambdas_ct[0])
        )
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(scatter1, ax=ax1)

    # second principal mode error
    scatter2 = ax2.scatter(X, Y, c=error_2, cmap="viridis")
    ax2.set_title(
        "Absolute Error of $\lambda_2 = {:.3f}$".format(
            np.real(principal_lambdas_ct[1])
        )
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(scatter2, ax=ax2)

    plt.tight_layout()
    plt.show()
    # now these obtained eigenfunctions can be used for reachability verification
