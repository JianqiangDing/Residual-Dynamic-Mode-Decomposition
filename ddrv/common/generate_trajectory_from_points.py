# this function is used to generate the trajectory data from the given points

import numpy as np
import sympy as sp
from scipy.integrate import odeint

from .simulate import simulate


def generate_trajectory_from_points(
    dynamical_system, points, num_steps, dt, forward=True
):
    """
    generate the trajectory data from the given points
    """

    if forward:
        return simulate(
            dynamical_system.get_numerical_dynamics(forward=True),
            points,
            0,
            num_steps * dt,
            dt,
        )[0]
    else:

        return simulate(
            dynamical_system.get_numerical_dynamics(forward=False),
            points,
            0,
            num_steps * dt,
            dt,
        )[0]
