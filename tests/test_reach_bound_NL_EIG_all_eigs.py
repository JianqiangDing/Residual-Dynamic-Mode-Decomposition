# test the reachability verification of the NL_EIG system with all the eigenpairs obtained from the data

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

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

    print(
        L.shape,
        V.shape,
        residuals.shape,
        observables.shape,
    )

    # define the threshold for the residuals
    threshold = 1e-4  # CAUTION: but for larger threshold, the results are not good, like threshold = 1e-2, which is make sense in fact
    # get the eigenvalues and eigenvectors with residuals less than the threshold
    idx = np.where(residuals < threshold)[0]
    print(idx, "idx")
    L_feasible = L[idx]  #  these are discrete eigenvalues
    # get the continuous eigenvalues
    L_feasible_ct = np.log(L_feasible) / DELTA_T
    V_feasible = V[:, idx].T
    print(L_feasible.shape, V_feasible.shape)
    print(L_feasible_ct, "L_feasible_ct")

    # --------------------------- 3. apply the reachability verification algorithm ---------------------------
    # define the initial set, and target set for reachability verification
    X0 = [[0, 0.1], [1.1, 1.2]]
    XF = [[1.8, 1.9], [-0.8, -0.7]]

    pts_X0 = ddrv.common.sample_box_set(X0, 5000)
    pts_XF = ddrv.common.sample_box_set(XF, 5000)
    print(pts_X0.shape, pts_XF.shape)

    # evaluate the eigenfunctions on the initial set and target set samples
    ef0_vals = observables.eval_mod(pts_X0, V_feasible)
    efF_vals = observables.eval_mod(pts_XF, V_feasible)
    print(ef0_vals.shape, efF_vals.shape)

    # compute the reach time bounds
    time_bounds, status = ddrv.algo.compute_reach_time_bounds(
        ef0_vals, efF_vals, L_feasible_ct
    )
    print(time_bounds, status, "time_bounds & status")
    ddrv.viz.vis_rv(
        NL_EIG.get_numerical_dynamics(),
        domain=[[-2, 2], [-2, 2]],
        bounds=time_bounds,
        dt=0.01,
        initial_set=X0,
        target_set=XF,
    )
