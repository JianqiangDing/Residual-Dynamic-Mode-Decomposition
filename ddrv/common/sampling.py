# some functions for sampling

import numpy as np


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
