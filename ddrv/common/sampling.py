# some functions for sampling

import itertools

import numpy as np


def sample_rnn_from_points(data, query, r, n_jobs=-1, is_sorted=True, is_length=False):
    """
    Sample radius nearest neighbors from given points.

    :param data: data to be sampled of shape (num_data, dimension)
    :param query: data to query of shape (num_query, dimension)
    :param r: radius, float
    :param n_jobs: number of processors, default -1 means using all processors
    :param is_sorted: sorts returned indices if TRUE
    :param is_length: return the number of points inside the radius instead of the
    neighboring indices
    :return: sampled data and related indices
    """
    from ddrv.common import kdtree, rnn_query

    kd = kdtree(data)
    indices = rnn_query(
        kd, query, r, n_jobs=n_jobs, is_sorted=is_sorted, is_length=is_length
    )

    # merge the list of lists into a list
    indices = itertools.chain.from_iterable(indices)
    indices_unique = np.unique(list(indices))
    return data[indices_unique, :], indices_unique


def sample_box_set(domain, num_samples):
    """Sample a box set."""
    dimension = len(domain)
    samples = np.zeros((num_samples, dimension))

    for i in range(dimension):
        x_min, x_max = domain[i]
        samples[:, i] = np.random.uniform(x_min, x_max, num_samples)

    return samples


def sample_level_set(level_set, num_samples):
    raise NotImplementedError("Not implemented yet")
