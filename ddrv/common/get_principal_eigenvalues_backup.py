# this function is used to get the principal eigenpairs of the koopman operator

import numpy as np

from .generate_trajectory_from_domain import generate_trajectory_from_domain


def get_principal_eigenvalues(
    dynamic,
    equilibrium,
    radius,
    dt=0.01,
    num_samples=1000,
    num_steps=10,
    random_seed=None,
):
    """
    Get the principal eigenpairs of the koopman operator

    dynamic: the dynamic system
    equilibrium: the equilibrium point
    radius: the radius of the neighborhood
    dt: the time step
    num_samples: the number of samples
    num_steps: the number of steps
    random_seed: the random seed

    Returns:
        principal_eigenvalues_dt: the principal eigenvalues in the discrete time
        principal_eigenvalues_ct: the principal eigenvalues in the continuous time


    TODO: refine this function with functionis from PyDMD
    """
    # generate the trajectory around the equilibrium point
    domain = [
        [equilibrium[0] - radius, equilibrium[0] + radius],
        [equilibrium[1] - radius, equilibrium[1] + radius],
    ]
    trajectory = generate_trajectory_from_domain(
        dynamic, num_samples, num_steps, dt, domain, random_seed
    )

    # now split the trajectory data into X and Y arrays, X refers the current state, Y refers the next state
    X = trajectory[:-1, :, :].reshape(-1, 2)
    Y = trajectory[1:, :, :].reshape(-1, 2)

    # here we assume that the behavior of the system is linear around the equilibrium point
    # so we can use the solve the linear least square problem to get the approximation of the koopman operator
    K = np.linalg.lstsq(X, Y, rcond=None)[0]

    # get the discrete eigenvalues of the koopman operator
    LAM_dt = np.linalg.eig(K)[0]
    # get the continuous eigenvalues of the koopman operator
    LAM_ct = np.log(LAM_dt) / dt
    return LAM_dt, LAM_ct
