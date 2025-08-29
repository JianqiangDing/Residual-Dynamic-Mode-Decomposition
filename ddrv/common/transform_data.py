from ..observables import PolyObservable, RBFObservable


def _transform_data_rbf(X, Y, observe_params):
    rbf = RBFObservable(c=observe_params["c"], r=observe_params["r"])
    PX = rbf.apply(X)
    PY = rbf.apply(Y)
    return PX, PY, rbf


def _transform_data_poly(X, Y, observe_params):
    poly = PolyObservable(dim_in=X.shape[1], degree=observe_params["degree"])
    PX = poly.apply(X)
    PY = poly.apply(Y)
    return PX, PY, poly


def transform_data(X, Y, observe_params):
    assert X.ndim == 2, "X must be a 2D array"
    assert Y.ndim == 2, "Y must be a 2D array"

    if observe_params["basis"] == "rbf":
        PX, PY, observables = _transform_data_rbf(X, Y, observe_params)
    elif observe_params["basis"] == "poly":
        PX, PY, observables = _transform_data_poly(X, Y, observe_params)
    else:
        raise ValueError(f"Unsupported basis: {observe_params['basis']}")
    return PX, PY, observables
