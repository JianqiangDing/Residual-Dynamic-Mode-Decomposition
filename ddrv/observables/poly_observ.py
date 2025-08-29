import numpy as np
import sympy as sp
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

from ._base_observ import BaseObservable


class PolyObservable(BaseObservable):
    """
    A polynomial observable.
    """

    def __init__(self, dim_in: int, degree: int):
        # dim_out is computed from the degree and dim_in for the polynomial observable
        assert degree >= 1, "degree must be at least 1"
        self.degree = degree
        name = f"poly_observ_{dim_in}_{degree}"
        description = f"polynomial observable of degree {degree} for {dim_in} variables"
        self.dim_in = dim_in
        self._init_symbols()
        self._init_symbolic_expr()
        self.dim_out = self.symbolic_expr.shape[0] + 1
        super().__init__(name, description, self.dim_in, self.dim_out)

        self._f = sp.lambdify(self.variables, self.symbolic_expr, "numpy")

    def _init_symbolic_expr(self):
        """
        initialize the symbolic expression of the polynomial observable.

        """
        self.symbolic_expr = sp.Matrix(
            sorted(
                itermonomials(self.variables, max_degrees=self.degree, min_degrees=1),
                key=monomial_key("grlex", np.flip(self.variables)),
            )
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        apply the polynomial observable to an array of state variables of shape (n_samples, n_states)
        """
        assert x.ndim == 2, "x must be a 2D array"
        assert (
            x.shape[1] == self.dim_in
        ), "x must have the same number of columns as the number of variables"
        ret = np.ones((x.shape[0], self.dim_out))
        ret[:, 1:] = self._f(*x.T).squeeze().T
        return ret

    def __str__(self) -> str:
        """
        get the string representation of the polynomial observable.
        """
        return self.description

    def eval_mod(self, x: np.ndarray, mod: np.ndarray) -> np.ndarray:
        """
        evaluate the polynomial observable with given mode on given data.
        mod: (num_modes,num_coeffs_for_each_basis_observable)
        x: (n_samples,n_states)
        return: (n_samples,num_modes)
        """
        assert x.ndim == 2, "x must be a 2D array"
        assert (
            x.shape[1] == self.dim_in
        ), "x must have the same number of columns as the number of variables"
        assert mod.ndim == 2, "mod must be a 2D array"
        assert (
            mod.shape[1] == self.dim_out
        ), "mod must have the same number of columns as the number of variables"

        val_basis = self.apply(x)
        ret = mod[None, :, :] * val_basis[:, None, :]
        return ret.sum(axis=2)
