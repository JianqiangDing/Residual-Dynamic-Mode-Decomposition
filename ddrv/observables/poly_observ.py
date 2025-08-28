import numpy as np
import sympy as sp

from ._base_observ import BaseObservable


class PolyObservable(BaseObservable):
    """
    A polynomial observable.
    """

    def __init__(self, name: str, description: str, dim_in: int, degree: int):
        # dim_out is computed from the degree and dim_in for the polynomial observable
        dim_out = int(np.sum(np.arange(degree + 1)))
        super().__init__(name, description, dim_in, dim_out)
        self._init_symbols()
        self._init_symbolic_expr()

    def _init_symbolic_expr(self):
        """
        initialize the symbolic expression of the polynomial observable.

        """
        self.symbolic_expr = sorted(
            sp.itermonomials(self._variables, self.degree),
            key=sp.monomial_key("grlex", np.flip(self._variables)),
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        apply the polynomial observable to an array of state variables of shape (n_samples, n_states)
        """
        pass
