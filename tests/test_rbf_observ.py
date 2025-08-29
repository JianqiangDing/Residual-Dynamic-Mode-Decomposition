# test the RBF observable

# add the ddrv package to the path
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    """Test the RBF observable"""
    # create a RBF observable
    c = np.random.rand(100, 5)
    r = np.random.rand(100)
    rbf = ddrv.observables.RBFObservable(c, r)
    print(rbf)
    data = np.random.rand(10000, 5)
    val = rbf.apply(data)
    print(val.shape)

    mod = np.random.rand(33, rbf.dim_out)
    val_mod = rbf.eval_mod(data, mod)
    print(val_mod.shape)
