# this function is used to simulate the dynamical system with given initial points and specific time interval

import numpy as np
from scipy.integrate import odeint


def simulate(numerical_dynamic, pts, T_min, T_max, dt):
    """
    simulate the dynamical system with given initial points and specific time interval
    """

    if not isinstance(pts, np.ndarray):
        pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    assert pts.ndim == 2  # N*n matrix, N points in n-dimensional space
    assert T_min <= T_max and (T_max - T_min) >= dt and dt > 0

    # avoid the numerical issue when T_min is 0
    T_min = 1e-6 if T_min == 0 else T_min

    # calculate the number of steps
    steps = np.ceil((T_max - T_min) / dt).astype(int)

    # vectorized ODE right-hand side function
    def ode_fx(x, t):
        x = x.reshape(pts.shape).T  # reshape to (dim, N)
        y = (
            numerical_dynamic(*x).squeeze(axis=1).T
        )  # call the dynamics for all input points
        return y.flatten()

    # create the time grid for the entire simulation
    t = np.linspace(T_min, T_max, steps + 1)

    # integrate the entire trajectory in one call
    sol = odeint(ode_fx, pts.flatten(), t)

    # reshape the solution to proper format
    trajectories = sol.reshape(
        steps + 1, pts.shape[0], pts.shape[1]
    )  # shape: (steps+1, N, dim)

    return trajectories, t
