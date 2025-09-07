# test the generate_trajectory_data function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    # test the generate_trajectory_data function
    duffing_oscillator = ddrv.dynamic.DuffingOscillator()
    traj_data = ddrv.common.generate_trajectory_data(
        duffing_oscillator,
        num_samples=1000,
        num_steps=150,
        dt=0.05,
        forward=False,
        domain=[[0.8, 1.2], [-0.2, 0.2]],
    )
    print(traj_data.shape)
    ddrv.viz.vis_trajectory_2d(traj_data)
