import numpy as np
import sympy as sp

from ._base_observ import BaseObservable


class RBFObservable(BaseObservable):
    """
    A radial basis function observable.
    The radial basis function is defined as:
    f(x) = sum_{i=1}^n exp(-||x - c_i||^2 / (r_i^2))
    where c_i is the i-th center and r_i is the i-th radius.
    """

    def __init__(self, c: np.ndarray, r: np.ndarray):
        assert (
            c.shape[0] == r.shape[0]
        ), "the number of centers and radii must be the same"
        assert c.ndim == 2, "the centers must be a 2D array"
        assert r.ndim == 1, "the radii must be a 1D array"
        self.dim_in = c.shape[1]
        self.dim_out = c.shape[0]
        self.n_centers = c.shape[0]
        self.c = c
        self.r = r
        super().__init__(
            name=f"RBF_{self.dim_in}_{self.n_centers}",
            description=f"RBF observable with {self.n_centers} centers",
            dim_in=self.dim_in,
            dim_out=self.dim_out,
        )
        self._init_symbols()
        self._init_symbolic_expr()

        self._f = sp.lambdify(self.variables, self.symbolic_expr, "numpy")

    def _init_symbolic_expr(self):
        self.symbolic_expr = sp.Matrix(
            [
                sp.exp(
                    -sum(
                        (self.variables[j] - self.c[i, j]) ** 2
                        for j in range(self.dim_in)
                    )
                    / self.r[i] ** 2
                )
                for i in range(self.n_centers)
            ]
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        apply the RBF observable to an array of state variables of shape (n_samples, n_states)
        """
        assert x.ndim == 2, "x must be a 2D array"
        assert (
            x.shape[1] == self.dim_in
        ), "x must have the same number of columns as the number of variables"
        return self._f(*x.T).squeeze().T

    def __str__(self) -> str:
        """
        get the string representation of the RBF observable.
        """
        return f"RBF observable with {self.n_centers} centers"

    def eval_mod(self, x: np.ndarray, mod: np.ndarray) -> np.ndarray:
        """
        evaluate the RBF observable with given mode on given data.
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
