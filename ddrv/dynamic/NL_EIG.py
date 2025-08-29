# definition of the NL_EIG system

from dataclasses import dataclass

import numpy as np
import sympy as sp

from ._base_dynamic import DynamicalSystem


@dataclass
class NL_EIG(DynamicalSystem):
    """
    NL_EIG system class - a 2D nonlinear system with known eigenvalues and eigenfunctions
    """

    lambda1: float = -1.0
    lambda2: float = 2.5

    def __post_init__(self):
        """initialize the system"""
        super().__init__(dimension=2)
        self.Lambda = np.array([self.lambda1, self.lambda2])

        # define the symbolic variables
        self.x = sp.symbols(f"x:{self.dimension}")

        # define the principal eigenfunctions
        self.psi1 = self.x[0] ** 2 + 2 * self.x[1] + self.x[1] ** 3
        self.psi2 = self.x[0] + sp.sin(self.x[1]) + self.x[0] ** 3

        self.Psi = sp.Matrix([self.psi1, self.psi2])

        # compute the Jacobian matrix
        self.J = self.Psi.jacobian(self.x)

        # define the dynamical system
        self.f = self.J.inv() @ np.diag(self.Lambda) @ self.Psi

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

    def get_numerical_eigenfunctions(self):
        """
        get the numerical eigenfunctions
        """
        return sp.lambdify(self.x, self.Psi, "numpy")

    def get_eigenvalues(self):
        """
        get the eigenvalues

        Returns:
            np.ndarray: the eigenvalues
        """
        return self.Lambda
