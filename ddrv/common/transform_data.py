from ..observables import PolyObservable, RBFObservable


def _transform_data_rbf(X, observe_params):
    rbf = RBFObservable(c=observe_params["c"], r=observe_params["r"])
    PX = rbf.apply(X)
    return PX, rbf


def _transform_data_poly(X, observe_params):
    poly = PolyObservable(dim_in=X.shape[1], degree=observe_params["degree"])
    PX = poly.apply(X)
    return PX, poly


def transform_data(X, observe_params):
    assert X.ndim == 2, "X must be a 2D array"

    if observe_params["basis"] == "rbf":
        PX, observables = _transform_data_rbf(X, observe_params)
    elif observe_params["basis"] == "poly":
        PX, observables = _transform_data_poly(X, observe_params)
    else:
        raise ValueError(f"Unsupported basis: {observe_params['basis']}")
    return PX, observables
