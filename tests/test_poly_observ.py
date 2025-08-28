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
    poly_observ = ddrv.PolyObservable(dim_in=7, degree=2)
    print(poly_observ)
    print(poly_observ.symbolic_expr)
    print(poly_observ.variables)

    data = np.random.rand(10000, 7)
    val = poly_observ.apply(data)
    print(val.shape)
