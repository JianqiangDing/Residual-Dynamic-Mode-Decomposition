# definition of the van der pol system

from dataclasses import dataclass

import numpy as np
import sympy as sp

from ._base_dynamic import DynamicalSystem


@dataclass
class Vanderpol(DynamicalSystem):
    """
    Vanderpol system class - a 2D nonlinear system with known eigenvalues and eigenfunctions
    """

    def __post_init__(self):
        """initialize the system"""
        super().__init__(dimension=2)

        # define the symbolic variables
        self.x = sp.symbols(f"x:{self.dimension}")
        self.mu = 1.0

        # define the dynamical system
        self.f = sp.Matrix(
            [self.x[1], self.mu * (1 - self.x[0] ** 2) * self.x[1] - self.x[0]]
        )

    def get_dynamics(self):
        """
        get the dynamical system definition
        """
        return self.f
