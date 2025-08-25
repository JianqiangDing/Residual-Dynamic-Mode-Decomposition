import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ddrv

if __name__ == "__main__":
    # define the dynamical system
    dynamical_system = ddrv.NL_EIG_System()

    # generate the trajectory data
    X, Y = ddrv.generate_trajectory_data(dynamical_system)

    # visualize the vector field
    ddrv.visualize_vector_field(dynamical_system.get_numerical_dynamics())
