# test the visualization functions

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ddrv

if __name__ == "__main__":
    dynamical_system = ddrv.dynamic.NL_EIG(lambda1=-1.0, lambda2=2.5)
    ddrv.viz.vis_vector_field_2d(
        dynamical_system.get_numerical_dynamics(), domain=[-2, 2, -2, 2], step_size=0.1
    )

    traj_data = ddrv.common.generate_trajectory_data(
        dynamical_system, num_samples=1000, num_steps=10, delta_t=0.05
    )
    print(traj_data.shape)
    ddrv.viz.vis_trajectory_2d(traj_data)
