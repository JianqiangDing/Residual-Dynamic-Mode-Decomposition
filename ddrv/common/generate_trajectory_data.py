# generate trajectory data from a dynamical system

from typing import List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.integrate import odeint


def generate_trajectory_data(
    dynamical_system,
    num_samples: int = 1000,
    num_steps: int = 10,
    dt: float = 0.05,
    domain: Optional[List[Tuple[float, float]]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate the trajectory data

    Args:
        dynamical_system: the dynamical system object
        num_samples: the number of trajectories
        num_steps: the number of time steps per trajectory
        dt: the time step
        domain: the domain of the initial conditions (min_val, max_val), if None then use (-1, 1)
        random_seed: the random seed
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, Y) data pairs, X is the current state, Y is the next state
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if domain is None:
        # domain should be an array of shape (dimension, 2), each row is the min and max of the domain for each dimension
        domain = np.array([(-1, 1)] * dynamical_system.dimension)
    else:
        domain = np.array(domain)
    assert domain.shape == (
        dynamical_system.dimension,
        2,
    ), "the domain must be a tuple of length the dimension of the system"

    # Vectorized integration for all initial points using scipy.integrate.odeint
    dim = dynamical_system.dimension

    # Draw all initial conditions at once, shape: (dim, num_samples)
    Y0_batch = (np.random.rand(dim, num_samples)) * (
        (domain[:, 1] - domain[:, 0]).reshape(dim, 1)
    ) + domain[:, 0].reshape(dim, 1)

    # Vectorized dynamics from symbolic definition
    f_symbolic = dynamical_system.get_dynamics()
    x_vars = sp.symbols(f"x:{dim}")
    f_vec = sp.lambdify(x_vars, f_symbolic, "numpy")

    # Batch ODE for odeint (flattened state of all points)
    def ode_fx(x_flat, t):
        X_pts = x_flat.reshape(num_samples, dim)  # (Npoints, dim)
        vals = f_vec(*(X_pts.T))  # returns (dim, Npoints) or similar
        V = np.array(vals)
        if V.ndim == 1:
            V = V.reshape(dim, -1)
        return V.T.flatten()

    # Small burn-in to avoid t=0
    T_min = 1e-8
    steps_init = int(np.ceil(T_min / dt)) if T_min > 0 else 0
    if steps_init > 0:
        x_init = odeint(
            ode_fx, Y0_batch.T.flatten(), np.linspace(0.0, T_min, steps_init)
        )[-1]
        X_cur = x_init.reshape(num_samples, dim)
    else:
        X_cur = Y0_batch.T  # (num_samples, dim)

    # March forward for num_steps steps with small internal solver grid
    traj = [X_cur]  # list of (num_samples, dim)
    t_small = np.linspace(0.0, dt, 5)
    for _ in range(num_steps):
        x_next = odeint(ode_fx, X_cur.flatten(), t_small)[-1].reshape(num_samples, dim)
        traj.append(x_next)
        X_cur = x_next

    print(
        f"generated {num_samples} trajectories in batch (odeint), {num_steps} steps, {dt} time step..."
    )

    return np.stack(traj)
