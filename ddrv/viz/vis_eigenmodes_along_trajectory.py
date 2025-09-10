# this functions is for visualizing the evolving of values of eigenfunctions along given trajectories
import matplotlib.pyplot as plt
import numpy as np


def vis_eigenmodes_along_trajectory(trajectory, eigenfunctions, observables):
    """
    Visualize the values of eigenfunctions along given trajectories.

    :param trajectory: trajectory data of shape (num_steps, num_samples, dimension)
    :param eigenfunctions: eigenfunctions of shape (num_eigenmodes, num_observables)
    :param observables: the observables function used in the presdmd function
    """

    # got values of evaluations of eigenfunctions along the trajectory
    num_steps, num_samples, _ = trajectory.shape
    num_eigenmodes = eigenfunctions.shape[0]
    traj_data = trajectory.reshape(-1, trajectory.shape[2])
    efv = observables.eval_mod(traj_data, eigenfunctions)
    efv = efv.reshape(num_steps, num_samples, num_eigenmodes)

    # got the magnitude of the evaluations
    efv_mag = np.abs(efv)
    # got the angle of the evaluations
    efv_angle = np.angle(efv)

    # init fig with axes for the magnitude and angle of each eigenmode
    fig, axes = plt.subplots(
        num_eigenmodes, 2, figsize=(8, 2 * num_eigenmodes), sharex=True
    )
    t = np.arange(num_steps)

    for i in range(num_eigenmodes):
        ax_mag = axes[i, 0] if num_eigenmodes > 1 else axes[0]
        ax_angle = axes[i, 1] if num_eigenmodes > 1 else axes[1]
        for j in range(num_samples):
            ax_mag.plot(t, efv_mag[:, j, i], alpha=0.5)
            ax_angle.plot(t, efv_angle[:, j, i], alpha=0.5)
        ax_mag.set_ylabel(f"Mag {i+1}")
        ax_mag.grid()
        ax_angle.set_ylabel(f"Angle {i+1}")
        ax_angle.grid()
    axes[-1, 0].set_xlabel("Time step")
    axes[-1, 1].set_xlabel("Time step")
    axes[0, 0].set_title("Magnitude of Eigenmodes")
    axes[0, 1].set_title("Angle of Eigenmodes")
    plt.tight_layout()
    plt.show()
