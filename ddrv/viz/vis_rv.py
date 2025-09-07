# this functions is used to visualize the reachability verification results

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ddrv.common import simulate
from ddrv.common.sampling import sample_box_set

from .vis_vector_field_2d import vis_vector_field_2d


def vis_rv(dynamics, domain, bounds, dt, initial_set, target_set):
    """
    visualize the reachability verification results

    baiscally, we need to visualize the vector field and simulated trajectories within the computed reach-time bounds,
    and color the trajectories within the bounds in red, and the trajectories outside the bounds in blue.

    all the contents are visualized in 2D in the same figure
    """

    # visualize the vector field
    _, ax = vis_vector_field_2d(dynamics, domain, show=False)
    # get simulated trajectories within the computed reach-time bounds

    # if bounds is empty, then skip the simulation
    if len(bounds) > 0:
        trajs_all, t = simulate(
            dynamics,
            sample_box_set(initial_set, 1000),
            0,
            np.max(bounds),
            dt,
        )
        print("finshed simulation")
        # the shape of trajs_all is (num_steps, num_samples, dim)
        num_steps, num_samples, dim = trajs_all.shape

        # add the simulated trajectories to the figure
        for i in range(num_samples):
            ax.plot(trajs_all[:, i, 0], trajs_all[:, i, 1], color="gray", linewidth=0.5)

        # get the trajectories within the bounds
        for bound in bounds:
            trajs_within_bounds = trajs_all[(t >= bound[0]) & (t <= bound[1])]
            # plot the trajectories within the bounds in red
            for i in range(num_samples):
                ax.plot(
                    trajs_within_bounds[:, i, 0],
                    trajs_within_bounds[:, i, 1],
                    color="green",
                    linewidth=0.5,
                    alpha=0.1,
                )

    # add the initial set and target set to the figure as two rectangles with different colors without filling the inside

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
        zorder=10,
    )
    ax.add_patch(initial_rect)

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
        zorder=10,
    )
    ax.add_patch(target_rect)
    # add the legend
    ax.legend()
    # add the title
    plt.title("Reachability Verification Results")

    plt.tight_layout()
    plt.show()
