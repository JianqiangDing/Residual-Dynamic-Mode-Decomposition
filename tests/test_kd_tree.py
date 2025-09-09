# This is a test for the kd_tree module

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv


def test_knn():
    import numpy as np

    from ddrv.common import kdtree

    data = np.random.rand(1000, 3)
    query = np.random.rand(10, 3)

    kd = kdtree(data)
    distances, indices = ddrv.common.kd_tree.knn_query(kd, query, k=5)

    assert distances.shape == (10, 5)
    assert indices.shape == (10, 5)


def test_rnn():
    import numpy as np

    from ddrv.common import kdtree

    data = np.random.rand(1000, 3)
    query = np.random.rand(10, 3)

    kd = kdtree(data)
    indices = ddrv.common.kd_tree.rnn_query(kd, query, r=0.1)

    assert len(indices) == 10
    print(indices)


if __name__ == "__main__":
    test_knn()
    test_rnn()
