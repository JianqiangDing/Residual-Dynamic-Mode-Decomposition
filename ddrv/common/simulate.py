# this function is used to simulate the dynamical system with given initial points and specific time interval

import numpy as np
from scipy.integrate import odeint


def simulate(numerical_dynamic, pts, T_min, T_max, dt):
    """
    simulate the dynamical system with given initial points and specific time interval
    """

    # convert the initial points to appropriate format
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    N, dim = pts.shape

    # create the time grid
    t = np.arange(T_min, T_max + dt, dt)

    # initialize the trajectory array
    trajectories = np.zeros((N, len(t), dim))

    # perform the numerical integration for each initial point
    for i, x0 in enumerate(pts):

        def ode_rhs(x, _t):
            x = np.asarray(x)
            return numerical_dynamic(_t, x)

        traj = odeint(ode_rhs, x0, t)
        trajectories[i] = traj

    # transpose the first two dimensions of the trajectories
    trajectories = trajectories.transpose(1, 0, 2)

    return trajectories, t
