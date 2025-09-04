# test the find_closet_subset_index function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import ddrv

if __name__ == "__main__":
    short_array = np.array([1.0, 3.0, 5.0])
    long_array = np.array([0.5, 1.1, 2.0, 3.2, 4.8, 5.1])
    indices = ddrv.common.find_closet_subset_index(
        short_array, long_array, lambda x, y: np.abs(x - y)
    )
    print(indices)
    print(long_array[indices])

    # now test larger arrays at least 10, but seems failed, TODO: fix it later
    # short_array = np.random.rand(10)
    # long_array = np.random.rand(100)
    # indices = ddrv.common.find_closet_subset_index(
    #     short_array, long_array, lambda x, y: np.abs(x - y)
    # )
    # print(indices)
    # error = np.abs(short_array - long_array[indices])
    # print(error)
