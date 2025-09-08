# test the presdmd algorithm

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import ddrv

if __name__ == "__main__":
    # test the presdmd algorithm
    DT = 0.01
    duffing_oscillator = ddrv.dynamic.DuffingOscillator()

    traj_data = ddrv.common.generate_trajectory_from_domain(
        duffing_oscillator,
        num_samples=1000,
        num_steps=20,
        dt=DT,
        domain=[[0.99, 1.01], [-0.01, 0.01]],
        random_seed=0,
    )

    LAM, V, residuals, observables, PX, PY, K = ddrv.algo.presdmd(
        traj_data,
        k=2,
        dt=DT,
        observe_params={"basis": "poly", "degree": 15},
    )
    # print(LAM, "LAM")
    # compute the continuous eigenvalues
    LAM_ct = np.log(LAM) / 0.01
    print(LAM_ct, "LAM_ct")

    # test for the NL-EIG system
    DT = 0.01
    NL_EIG = ddrv.dynamic.NL_EIG()
    traj_data = ddrv.common.generate_trajectory_from_domain(
        NL_EIG,
        num_samples=1000,
        num_steps=20,
        dt=DT,
        domain=[[-0.01, 0.01], [-0.01, 0.01]],
        random_seed=0,
    )

    LAM, V, residuals, _, _, _, _ = ddrv.algo.presdmd(
        traj_data,
        k=2,
        dt=DT,
        observe_params={"basis": "poly", "degree": 14},
    )
    # print(LAM, "LAM")
    # compute the continuous eigenvalues
    LAM_ct = np.log(LAM) / DT
    print(LAM_ct, "LAM_ct")
