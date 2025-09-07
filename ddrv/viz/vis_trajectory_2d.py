# visualize the trajectory in 2D

import matplotlib.pyplot as plt


# this function is just for saving the space when visualizing the trajectory
def vis_trajectory_2d(trajectory):
    """
    visualize the trajectory in 2D

    trajectory: np.ndarray, shape (num_steps,num_samples,dim)
    no return, just visualize the trajectory
    """
    _, num_samples, dim = trajectory.shape
    assert dim == 2, "the trajectory must be in 2D"
    plt.figure(figsize=(8, 6))
    for i in range(num_samples):
        plt.plot(
            trajectory[:, i, 0],
            trajectory[:, i, 1],
            "b-",
            linewidth=0.5,
            alpha=0.5,
        )
    # mark all start points
    plt.scatter(
        trajectory[0, :, 0],
        trajectory[0, :, 1],
        c="green",
        s=20,
        alpha=0.7,
        label="start point",
    )

    plt.tight_layout()
    plt.show()
