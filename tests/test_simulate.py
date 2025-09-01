# test the simulate function

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    # test the simulate function
    dynamical_system = ddrv.dynamic.NL_EIG(lambda1=-1.0, lambda2=2.5)
    pts = np.random.rand(100, 2)
    T_min = 0
    T_max = 1
    dt = 0.1
    trajs, t = ddrv.common.simulate(
        dynamical_system.get_numerical_dynamics(), pts, T_min, T_max, dt
    )
    print(trajs.shape, t.shape)
    # visualize the simulated trajectories
    ddrv.viz.vis_trajectory_2d(trajs)
