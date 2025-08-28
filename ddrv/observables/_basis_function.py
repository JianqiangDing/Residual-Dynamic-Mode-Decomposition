import abc

import sympy as sp


# this is a private class, not intended to be used directly, but can be used within the module
class _BaseBasisFunction(abc.ABC):
    """
    Base class for all basis functions.
    A basis function is a function f(x): R^n -> R.
    """

    def __init__(self, name: str, description: str, dimension: int):
        self.name = name
        self.description = description
        self.dimension = dimension  # dimension of the input space

    def _init_symbolic_expr(self, variables: list[sp.Symbol]):
        """
        initialize the symbolic expression of the basis function.
        """
        raise NotImplementedError  # this force all children to implement this method

    def __str__(self) -> str:
        """
        get the string representation of the basis function.
        """
        raise NotImplementedError  # this force all children to implement this method
