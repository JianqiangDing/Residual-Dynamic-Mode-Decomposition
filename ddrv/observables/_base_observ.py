from abc import ABC, abstractmethod

import numpy as np
import sympy as sp


class BaseObservable(ABC):
    """
    Base class for all observables.

    An observable is a function f(x): R^n -> R^m.
    The structure of an observable is a vector of basis functions.
    so, this class supports some operations on the basis functions.

    """

    def __init__(self, name: str, description: str, dim_in: int, dim_out: int):
        self.name = name
        self.description = description
        self.dim_in = dim_in  # dimension of the input space
        self.dim_out = dim_out  # dimension of the output space

    def _init_symbols(self):
        """
        initialize the symbols for the variables of the observable function.
        """
        self.variables = sp.symbols(f"x:{self.dim_in}", real=True)

    @abstractmethod
    def _init_symbolic_expr(self):
        """
        initialize the symbolic expression of the observable function.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        apply the observable function to an array of state variables of shape (n_samples, n_states)
        and return an array of shape (n_samples,).
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """
        get the string representation of the observable function.
        How to use: print(obs) or str(obs)
        """
        raise NotImplementedError

    @property
    def shape(self) -> tuple:
        """
        get the shape of the observable function.
        """
        return (self.dim_in, self.dim_out)

    # override the __add__ method
    def __add__(self, other: "BaseObservable") -> "BaseObservable":
        """
        add the observable function with another observable function.
        """
        assert (
            self.dim_in == other.dim_in
        ), "the dimension of the input space must be the same"

        name = f"{self.name} + {other.name}"
        description = f"{self.description} + {other.description}"
        dim_in = self.dim_in
        dim_out = self.dim_out + other.dim_out
        # TODO: implement the addition of the observable functions

        raise NotImplementedError

    @abstractmethod
    def eval_mod(self, x: np.ndarray, mod: np.ndarray) -> np.ndarray:
        """
        evaluate the observable function with given mode on given data.

        mod: (num_modes,num_coeffs_for_each_basis_observable)
        x: (n_samples,n_states)
        """
        raise NotImplementedError
