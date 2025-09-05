# test the get_principal_eigenvalues function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    # test the get_principal_eigenvalues function for the NL_EIG system
    NL_EIG = ddrv.dynamic.NL_EIG(lambda1=-1.0, lambda2=2.5)
    equilibrium = [0, 0]
    radius = 0.00001
    principal_eigenvalues_dt, principal_eigenvalues_ct = (
        ddrv.common.get_principal_eigenvalues(
            NL_EIG,
            equilibrium,
            radius,
            num_samples=5000,
            num_steps=10,
            dt=0.01,
            random_seed=0,
        )
    )
    print(principal_eigenvalues_dt, principal_eigenvalues_ct)

    # test the get_principal_eigenvalues function for the Duffing oscillator system
    Duffing_oscillator = ddrv.dynamic.Duffing_oscillator()
    equilibrium = [1, 0]
    radius = 0.000001
    principal_eigenvalues_dt, principal_eigenvalues_ct = (
        ddrv.common.get_principal_eigenvalues(
            Duffing_oscillator,
            equilibrium,
            radius,
            num_samples=1000,
            num_steps=10,
            dt=0.01,
            random_seed=0,
        )
    )
    print(principal_eigenvalues_dt, principal_eigenvalues_ct)
