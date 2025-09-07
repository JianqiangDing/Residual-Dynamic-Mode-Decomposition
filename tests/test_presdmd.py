# test the presdmd algorithm

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import ddrv

if __name__ == "__main__":
    # test the presdmd algorithm
    duffing_oscillator = ddrv.dynamic.DuffingOscillator()

    LAM, V, residuals, observables, PX, PY, K = ddrv.algo.presdmd(
        duffing_oscillator,
        k=2,
        domain=[[0.99, 1.01], [-0.01, 0.01]],
        dt=0.01,
        num_samples=1000,
        num_steps=10,
        observe_params={"basis": "poly", "degree": 15},
        random_seed=0,
    )
    # print(LAM, "LAM")
    # compute the continuous eigenvalues
    LAM_ct = np.log(LAM) / 0.01
    print(LAM_ct, "LAM_ct")

    # test for the NL-EIG system
    NL_EIG = ddrv.dynamic.NL_EIG()
    LAM, V, residuals, _, _, _, _ = ddrv.algo.presdmd(
        NL_EIG,
        k=2,
        domain=[[-0.01, 0.01], [-0.01, 0.01]],
        dt=0.01,
        num_samples=1000,
        num_steps=10,
        observe_params={"basis": "poly", "degree": 14},
        random_seed=0,
    )
    # print(LAM, "LAM")
    # compute the continuous eigenvalues
    LAM_ct = np.log(LAM) / 0.01
    print(LAM_ct, "LAM_ct")
