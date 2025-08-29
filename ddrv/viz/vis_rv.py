# this functions is used to visualize the reachability verification results

import matplotlib.pyplot as plt

from ddrv.common import simulate
from ddrv.common.sampling import sample_box_set

from .vis_vector_field_2d import vis_vector_field_2d


def vis_rv(dynamics, domain, bounds, initial_set, target_set):
    """
    visualize the reachability verification results

    baiscally, we need to visualize the vector field and simulated trajectories within the computed reach-time bounds,
    and color the trajectories within the bounds in red, and the trajectories outside the bounds in blue.

    all the contents are visualized in 2D in the same figure
    """

    # visualize the vector field
    fig = vis_vector_field_2d(dynamics, domain, show=False)
    # get simulated trajectories within the computed reach-time bounds

    # trajs_all, _ = simulate(
    #     dynamics, sample_box_set(initial_set, 1000), bounds[0][0], bounds[0][1], 0.05
    # )

    # add the simulated trajectories to the figure
    for traj in trajs_all:
        plt.plot(traj[:, 0], traj[:, 1], color="black", linewidth=0.5)

    # traj_within_bounds = simu

    # add the initial set and target set to the figure as two rectangles with different colors without filling the inside
    from matplotlib.patches import Rectangle

    # draw the initial set rectangle
    x_min, x_max = initial_set[0]
    y_min, y_max = initial_set[1]
    initial_rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="Initial Set",
    )
    plt.gca().add_patch(initial_rect)

    # draw the target set rectangle
    x_min, x_max = target_set[0]
    y_min, y_max = target_set[1]
    target_rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="blue",
        facecolor="none",
        label="Target Set",
    )
    plt.gca().add_patch(target_rect)
    # add the legend
    plt.legend()
    # add the title
    plt.title("Reachability Verification Results")
    # add the x and y labels
    # fig.show()
    # save the figure
    fig.savefig("vis_rv.png")
