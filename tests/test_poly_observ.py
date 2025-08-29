# test the polynomial observable

# add the ddrv package to the path
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import ddrv

if __name__ == "__main__":
    """Test the polynomial observable"""
    # create a polynomial observable
    poly = ddrv.observables.PolyObservable(dim_in=7, degree=2)
    print(poly)
    print(poly.symbolic_expr)
    print(poly.variables)

    data = np.random.rand(10000, 7)
    val = poly.apply(data)
    print(val.shape)

    mod = np.random.rand(100, poly.dim_out)
    val_mod = poly.eval_mod(data, mod)
    print(val_mod.shape)
