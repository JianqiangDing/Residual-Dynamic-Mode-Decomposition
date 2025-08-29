# test the reachability verification of the NL_EIG system
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    # set the seed for reproducibility
    np.random.seed(42)

    DELTA_T = 0.05

    # --------------------------- 1. generate the trajectory data ---------------------------

    # define the dynamical system
    NL_EIG = ddrv.dynamic.NL_EIG(lambda1=-1.0, lambda2=2.5)
    # ddrv.viz.vis_vector_field_2d(
    #     NL_EIG.get_numerical_dynamics(),
    #     domain=[[-2, 2], [-2, 2]],
    #     step_size=0.1,
    # )

    # generate the trajectory data
    traj_data = ddrv.common.generate_trajectory_data(
        NL_EIG,
        num_samples=1000,
        num_steps=10,
        delta_t=DELTA_T,
        domain=[[-2, 2], [-2, 2]],
    )
    print(traj_data.shape)

    # based on the settings, the discrete eigenvalues are:
    dc_lambdas = np.exp(np.array([-1.0, 2.5]) * DELTA_T)
    print("dc_lambdas", dc_lambdas)

    # now split the trajectory data into X and Y arrays, X refers the current state, Y refers the next state
    X = traj_data[:-1, :, :].reshape(-1, 2)
    Y = traj_data[1:, :, :].reshape(-1, 2)
    print(X.shape, Y.shape)

    # --------------------------- 2. apply the resdmd algorithm ---------------------------
    L, V, residuals, observables, PX, PY, K = ddrv.algo.resdmd(
        X, Y, observe_params={"basis": "poly", "degree": 9}
    )
    idx = np.argsort(residuals)[:6]
    principal_idx = idx[[1, 3]]
    # get the principal modes based on the residuals
    principal_lambdas_dt = L[principal_idx]  # which is discrete eigenvalues
    # get the continuous principal eigenvalues
    principal_lambdas_ct = np.log(principal_lambdas_dt) / DELTA_T
    principal_modes = V[
        :, principal_idx
    ].T  # reshape it to (num_modes, num_of_basis_functions)
    print(principal_modes.shape, "principal_modes")
    print(principal_lambdas_dt, principal_lambdas_ct, dc_lambdas, "lambdas")

    # --------------------------- 3. apply the reachability verification algorithm ---------------------------
    # define the initial set, and target set for reachability verification
    X0 = [[0, 0.1], [1.1, 1.2]]
    XF = [[1.8, 1.9], [-0.8, -0.7]]

    pts_X0 = ddrv.common.sample_box_set(X0, 2000)
    pts_XF = ddrv.common.sample_box_set(XF, 2000)
    print(pts_X0.shape, pts_XF.shape)

    # evaluate the principal modes on the initial set and target set samples
    ef0_vals = observables.eval_mod(pts_X0, principal_modes)
    efF_vals = observables.eval_mod(pts_XF, principal_modes)
    print(ef0_vals.shape, efF_vals.shape)

    # evaluate the ground truth eigenfunctions on the initial set and target set samples
    eigF = NL_EIG.get_numerical_eigenfunctions()
    ef0_vals_true = eigF(*pts_X0.T).squeeze().T
    efF_vals_true = eigF(*pts_XF.T).squeeze().T
    print(ef0_vals_true.shape, efF_vals_true.shape)

    # # compute the reach time bounds
    time_bounds, status = ddrv.algo.compute_reach_time_bounds(
        ef0_vals_true, efF_vals_true, principal_lambdas_ct
    )
    print(time_bounds, status)

    T = time_bounds[0][1]
    print(T, "T")

    # get the trajectory data with computed reach time bounds
    trajs, _ = ddrv.common.simulate(
        NL_EIG.get_numerical_dynamics(),
        pts_X0,
        T_min=0,
        T_max=T,
        dt=DELTA_T,
    )
    print(trajs.shape)
    # visualize the simulated trajectories
    import matplotlib.pyplot as plt

    for traj in trajs:
        plt.plot(traj[:, 0], traj[:, 1], color="black", linewidth=0.5)
    plt.show()
    # visualize the reachability verification results
    ddrv.viz.vis_rv(
        NL_EIG.get_numerical_dynamics(),
        domain=[[-2, 2], [-2, 2]],
        bounds=time_bounds,
        initial_set=X0,
        target_set=XF,
    )
