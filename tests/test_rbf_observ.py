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
    rbf_observ = ddrv.RBFObservable(c, r)
    print(rbf_observ)
    data = np.random.rand(10000, 5)
    val = rbf_observ.apply(data)
    print(val.shape)
