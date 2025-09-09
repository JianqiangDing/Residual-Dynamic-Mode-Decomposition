# test for the sample_rnn_from_points function in sampling.py
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    import numpy as np

    data = np.random.rand(1000, 3)
    query = np.random.rand(10, 3)

    data_rnn, indices = ddrv.common.sample_rnn_from_points(data, query, r=0.1)

    print(indices)
