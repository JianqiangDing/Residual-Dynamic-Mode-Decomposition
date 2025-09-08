# this function is used to get the principal eigenpairs of the koopman operator


import numpy as np

from ..algo.resdmd import resdmd
from .generate_trajectory_from_domain import generate_trajectory_from_domain


def get_principal_eigenvalues(
    dynamic,
    equilibrium,
    radius,
    dt=0.01,
    num_samples=1000,
    num_steps=10,
    random_seed=None,
    observe_params={"basis": "poly", "degree": 10},
):
    """
    Get the principal eigenvalues of the koopman operator
    """

    # generate the trajectory data
    domain = [
        [equilibrium[0] - radius, equilibrium[0] + radius],
        [equilibrium[1] - radius, equilibrium[1] + radius],
    ]
    trajectory = generate_trajectory_from_domain(
        dynamic,
        num_samples=num_samples,
        num_steps=num_steps,
        dt=dt,
        domain=domain,
        random_seed=random_seed,
    )

    # now split the trajectory data into X and Y arrays, X refers the current state, Y refers the next state
    X = trajectory[:-1, :, :].reshape(-1, trajectory.shape[2])
    Y = trajectory[1:, :, :].reshape(-1, trajectory.shape[2])

    # now apply the resdmd algorithm to get the approximation of the koopman operator
    K = resdmd(X, Y, observe_params)[-1]

    # get the discrete eigenvalues of the koopman operator
    LAM_dt = np.linalg.eig(K)[0]
    # get the continuous eigenvalues of the koopman operator
    LAM_ct = np.log(LAM_dt) / dt

    return LAM_dt, LAM_ct
