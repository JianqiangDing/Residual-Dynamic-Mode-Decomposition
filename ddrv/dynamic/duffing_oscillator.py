# definition of the duffing oscillator system

from dataclasses import dataclass

import numpy as np
import sympy as sp

from ._base_dynamic import DynamicalSystem


@dataclass
class DuffingOscillator(DynamicalSystem):
    """
    Duffing oscillator system class
    """

    def __post_init__(self):
        """initialize the system"""
        super().__init__(dimension=2)

        # define the symbolic variables
        self.x = sp.symbols(f"x:{self.dimension}")

        # define the dynamical system
        self.f = sp.Matrix(
            [self.x[1], -0.5 * self.x[1] - self.x[0] * (self.x[0] ** 2 - 1)]
        )

    def get_dynamics(self):
        """
        get the dynamical system definition
        """
        return self.f
