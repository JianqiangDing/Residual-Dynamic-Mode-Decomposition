# generate trajectory data from a dynamical system

from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from .generate_trajectory_from_points import generate_trajectory_from_points
from .sampling import sample_box_set


def generate_trajectory_from_domain(
    dynamical_system,
    num_samples: int = 1000,
    num_steps: int = 10,
    dt: float = 0.05,
    domain: Optional[List[Tuple[float, float]]] = None,
    random_seed: Optional[int] = None,
    forward: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate the trajectory data from the domain

    Args:
        dynamical_system: the dynamical system object
        num_samples: the number of trajectories
        num_steps: the number of time steps per trajectory
        dt: the time step
        domain: the domain of the initial conditions (min_val, max_val), if None then use (-1, 1)
        random_seed: the random seed
        forward: if True, then generate the trajectory data from the initial state to the next state, otherwise generate the trajectory data from the next state to the initial state
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

    # generate the trajectory data
    points = sample_box_set(
        domain=domain,
        num_samples=num_samples,
    )

    return generate_trajectory_from_points(
        dynamical_system, points, num_steps, dt, forward
    )
