# add the ddrv package to the path
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    # test the transform_data function with rbf basis
    num_samples = 1000
    num_centers = 100
    X = np.random.rand(num_samples, 2)
    Y = np.random.rand(num_samples, 2)
    c = np.random.rand(num_centers, 2)
    r = np.random.rand(num_centers)
    observe_params = {"basis": "rbf", "c": c, "r": r}
    PX, PY, observables = ddrv.common.transform_data(X, Y, observe_params)
    print(PX.shape, PY.shape)
    print(observables)

    # test the transform_data function with poly basis
    observe_params = {"basis": "poly", "degree": 2}
    PX, PY, observables = ddrv.common.transform_data(X, Y, observe_params)
    print(PX.shape, PY.shape)
    print(observables)
