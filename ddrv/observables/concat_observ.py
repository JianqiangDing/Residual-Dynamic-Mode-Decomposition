import numpy as np

from ._base_observ import BaseObservable


class ConcatObservable(BaseObservable):
    """
    A concatenation of multiple observables.
    """

    def __init__(self, observables: list[BaseObservable]):
        # check if each observable has the same input dimension
        for observable in observables:
            assert (
                observable.dim_in == observables[0].dim_in
            ), "each observable must have the same input dimension"

        self.dim_in = observables[0].dim_in
        self.dim_out = sum([observable.dim_out for observable in observables])
        self._observables = observables
        super().__init__(
            name=f"Concat_{len(observables)}",
            description=f"Concatenation of {len(observables)} observables",
            dim_in=self.dim_in,
            dim_out=self.dim_out,
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        apply the concatenation of the observables to an array of state variables of shape (n_samples, n_states)
        """
        assert x.ndim == 2, "x must be a 2D array"
        assert (
            x.shape[1] == self.dim_in
        ), "x must have the same number of columns as the number of variables"
        return np.concatenate(
            [observable.apply(x) for observable in self._observables], axis=1
        )

    def __str__(self) -> str:
        """
        get the string representation of the concatenation of the observables.
        """
        return f"Concatenation of {len(self._observables)} observables"

    def eval_mod(self, x: np.ndarray, mod: np.ndarray) -> np.ndarray:
        """
        evaluate the concatenation of the observables with given mode on given data.
        """
        assert x.ndim == 2, "x must be a 2D array"
        assert (
            x.shape[1] == self.dim_in
        ), "x must have the same number of columns as the number of variables"
        assert mod.ndim == 2, "mod must be a 2D array"
        assert (
            mod.shape[1] == self.dim_out
        ), "mod must have the same number of columns as the number of variables"
        return np.concatenate(
            [observable.eval_mod(x, mod) for observable in self._observables], axis=1
        )

    @property
    def observables(self) -> list[BaseObservable]:
        """
        get the list of the observables.
        """
        return self._observables

    def __add__(self, other: "ConcatObservable") -> "ConcatObservable":
        """
        add the concatenation of the observables with another concatenation of the observables.
        """
        return ConcatObservable(self._observables + other._observables)
