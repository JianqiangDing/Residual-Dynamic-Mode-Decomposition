# basic class for dynamical systems

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import sympy as sp


class DynamicalSystem(ABC):
    """
    abstract base class for dynamical systems
    """

    def __init__(self, dimension: int):
        """
        initialize the dynamical system
        """
        self.dimension = dimension

    @abstractmethod
    def get_dynamics(self) -> sp.Matrix:
        """
        get the dynamical system definition
        """
        raise NotImplementedError

    def get_numerical_dynamics(self) -> Callable:
        """
        get the numerical dynamics function, for numerical integration
        """
        # convert the symbolic expression to a numerical function
        f_symbolic = self.get_dynamics()
        x_vars = sp.symbols(f"x:{self.dimension}")
        f_lambdified = sp.lambdify(x_vars, f_symbolic, "numpy")

        def dynamics(t, x):
            return np.array(f_lambdified(*x)).flatten()

        return dynamics
