# this file is used to test the generate_trajectory_from_domain function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import ddrv


# test for the Vanderpol system
def test_vanderpol():
    # set the seed for reproducibility
    np.random.seed(42)

    # define the dynamical system
    Vanderpol = ddrv.dynamic.Vanderpol()

    # generate the trajectory data
    traj_data = ddrv.common.generate_trajectory_from_domain(
        Vanderpol,
        num_samples=1000,
        num_steps=100,
        dt=0.05,
        domain=[[-1, 1], [-1, 1]],
    )
    print(traj_data.shape)
    ddrv.viz.vis_trajectory_2d(traj_data)

    # generate the trajectory data backward
    traj_data_backward = ddrv.common.generate_trajectory_from_domain(
        Vanderpol,
        num_samples=1000,
        num_steps=100,
        dt=0.05,
        domain=[[-1, 1], [-1, 1]],
        forward=False,
    )
    print(traj_data_backward.shape)
    ddrv.viz.vis_trajectory_2d(traj_data_backward)


# test for the Duffing oscillator system
def test_duffing_oscillator():
    # set the seed for reproducibility
    np.random.seed(42)

    # define the dynamical system
    DuffingOscillator = ddrv.dynamic.DuffingOscillator()

    # generate the trajectory data
    traj_data = ddrv.common.generate_trajectory_from_domain(
        DuffingOscillator,
        num_samples=1000,
        num_steps=100,
        dt=0.05,
        domain=[[0.5, 1.5], [-0.5, 0.5]],
    )
    print(traj_data.shape)
    ddrv.viz.vis_trajectory_2d(traj_data)

    # generate the trajectory data backward
    traj_data_backward = ddrv.common.generate_trajectory_from_domain(
        DuffingOscillator,
        num_samples=1000,
        num_steps=100,
        dt=0.05,
        domain=[[0.5, 1.5], [-0.5, 0.5]],
        forward=False,
    )
    print(traj_data_backward.shape)
    ddrv.viz.vis_trajectory_2d(traj_data_backward)


# test for the NL_EIG system
def test_nl_eig():
    # set the seed for reproducibility
    np.random.seed(42)

    # define the dynamical system
    NL_EIG = ddrv.dynamic.NL_EIG()

    # generate the trajectory data
    traj_data = ddrv.common.generate_trajectory_from_domain(
        NL_EIG,
        num_samples=1000,
        num_steps=10,
        dt=0.05,
        domain=[[-2, 2], [-2, 2]],
    )
    print(traj_data.shape)
    ddrv.viz.vis_trajectory_2d(traj_data)

    # generate the trajectory data backward
    traj_data_backward = ddrv.common.generate_trajectory_from_domain(
        NL_EIG,
        num_samples=1000,
        num_steps=10,
        dt=0.05,
        domain=[[-2, 2], [-2, 2]],
        forward=False,
    )
    print(traj_data_backward.shape)
    ddrv.viz.vis_trajectory_2d(traj_data_backward)


if __name__ == "__main__":
    test_vanderpol()
    test_duffing_oscillator()
    test_nl_eig()
