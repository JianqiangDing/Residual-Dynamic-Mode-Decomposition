# auxiliary functions for ddrv

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# defination of dynamical systems ------------------------------------------------

# define the NL_EIG system (2D nonlinear system with known eigenvalues and eigenfunctions)


class DynamicalSystem(ABC):
    """
    abstract base class for defining dynamical systems
    """

    def __init__(self, dimension: int):
        """
        initialize the dynamical system

        Args:
            dimension: the dimension of the system
        """
        self.dimension = dimension

    @abstractmethod
    def get_dynamics(self) -> sp.Matrix:
        """
        get the dynamical system definition

        Returns:
            sp.Matrix: the dynamical system equation
        """
        pass

    def get_numerical_dynamics(self) -> Callable:
        """
        get the numerical dynamics function, for numerical integration

        Returns:
            Callable: the numerical dynamics function
        """
        # convert the symbolic expression to a numerical function
        f_symbolic = self.get_dynamics()
        x_vars = sp.symbols(f"x:{self.dimension}")
        f_lambdified = sp.lambdify(x_vars, f_symbolic, "numpy")

        def dynamics(t, x):
            return np.array(f_lambdified(*x)).flatten()

        return dynamics


@dataclass
class NL_EIG_System(DynamicalSystem):
    """
    NL_EIG system class - a 2D nonlinear system with known eigenvalues and eigenfunctions
    """

    lambda1: float = -1.0
    lambda2: float = 2.5

    def __post_init__(self):
        """initialize the system"""
        super().__init__(dimension=2)
        self.Lambda = np.diag([self.lambda1, self.lambda2])

        # define the symbolic variables
        self.x = sp.symbols("x:2")

        # define the principal eigenfunctions
        self.psi1 = self.x[0] ** 2 + 2 * self.x[1] + self.x[1] ** 3
        self.psi2 = self.x[0] + sp.sin(self.x[1]) + self.x[0] ** 3

        self.Psi = sp.Matrix([self.psi1, self.psi2])

        # compute the Jacobian matrix
        self.J = self.Psi.jacobian(self.x)

        # define the dynamical system
        self.f = self.J.inv() @ self.Lambda @ self.Psi

    def get_dynamics(self):
        """
        get the dynamical system definition

        Returns:
            sp.Matrix: the dynamical system equation
        """
        return self.f

    def get_eigenfunctions(self):
        """
        get the eigenfunctions

        Returns:
            sp.Matrix: the eigenfunctions vector
        """
        return self.Psi

    def get_eigenvalues(self):
        """
        get the eigenvalues
        """
        return self.Lambda


# --------------------------------------------------------------------------------
# now some auxiliary functions for obtaining the trajectory data from the given dynamical system
def generate_trajectory_data(
    dynamical_system: DynamicalSystem,
    M1: int = 1000,
    M2: int = 10,
    delta_t: float = 0.05,
    domain: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate the trajectory data

    Args:
        dynamical_system: the dynamical system object
        M1: the number of trajectories
        M2: the number of time steps per trajectory
        delta_t: the time step
        domain: the domain of the initial conditions (min_val, max_val), if None then use (-1.5, 1.5)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, Y) data pairs, X is the current state, Y is the next state
    """
    if domain is None:
        # domain should be an array of shape (dimension, 2), each row is the min and max of the domain for each dimension
        domain = np.array([(-1.5, 1.5)] * dynamical_system.dimension)
    assert domain.shape == (
        dynamical_system.dimension,
        2,
    ), "the domain must be a tuple of length the dimension of the system"

    dynamics = dynamical_system.get_numerical_dynamics()
    X_list = []
    Y_list = []

    print(f"generating {M1} trajectories, each with {M2} time steps...")

    for jj in range(M1):
        # random initial conditions
        Y0 = (np.random.rand(dynamical_system.dimension) - 0.5) * (
            domain[:, 1] - domain[:, 0]
        ) + domain[:, 0]

        # time points
        t_span = (0, (3 + M2) * delta_t)
        t_eval = np.concatenate([[0, 0.000001], np.arange(1, 4 + M2) * delta_t])

        # solve the ODE
        sol = solve_ivp(
            dynamics,
            t_span,
            Y0,
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-12,
            method="RK45",
        )

        if sol.success and len(sol.y[0]) >= 3 + M2:
            Y1 = sol.y  # (dimension × N_points)

            # collect the data pairs (current state, next state)
            # skip the index 1 (t=0.000001) to ensure numerical stability
            X_current = Y1[:, np.concatenate([[0], np.arange(2, 2 + M2)])]
            Y_next = Y1[:, np.arange(2, 3 + M2)]

            X_list.append(X_current)
            Y_list.append(Y_next)

        # progress indicator
        if (jj + 1) % 200 == 0:
            print(f"completed {jj + 1}/{M1} trajectories")

    # connect all data
    if X_list:
        X = np.concatenate(X_list, axis=1)
        Y = np.concatenate(Y_list, axis=1)
    else:
        X = np.zeros((dynamical_system.dimension, 0))
        Y = np.zeros((dynamical_system.dimension, 0))

    M = X.shape[1]
    print(f"generated {M} data points")

    return X, Y


def visualize_vector_field(dynamics, domain=[-2, 2, -2, 2], step_size=0.1):
    """
    visualize the vector field of the dynamical system
    """
    xmin, xmax, ymin, ymax = domain

    # create grid
    x = np.arange(xmin, xmax + step_size, step_size)
    y = np.arange(ymin, ymax + step_size, step_size)
    X, Y = np.meshgrid(x, y)

    # compute velocity field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                dxdt = dynamics(0, [X[i, j], Y[i, j]])
                U[i, j] = dxdt[0]
                V[i, j] = dxdt[1]
            except Exception:
                U[i, j] = np.nan
                V[i, j] = np.nan

    # plot streamlines
    plt.figure(figsize=(12, 9))

    start_points = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
    start_points = np.array([start_points[0].flatten(), start_points[1].flatten()]).T

    plt.streamplot(
        X, Y, U, V, start_points=start_points, color="blue", linewidth=0.8, density=1.5
    )

    # add contour lines of speed for context
    speed = np.sqrt(U**2 + V**2)
    plt.contour(
        X, Y, speed, levels=8, colors="gray", linestyles="--", linewidths=0.5, alpha=0.7
    )

    # formatting
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Vector Field (Streamlines)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.tight_layout()
    plt.show()
