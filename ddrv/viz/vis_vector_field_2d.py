# visualize the vector field with 2D plot

import matplotlib.pyplot as plt
import numpy as np


def vis_vector_field_2d(dynamics, domain=[[-2, 2], [-2, 2]], step_size=0.1, show=True):
    """
    visualize the vector field of the dynamical system
    """
    [xmin, xmax], [ymin, ymax] = domain

    # create grid
    x = np.arange(xmin, xmax + step_size, step_size)
    y = np.arange(ymin, ymax + step_size, step_size)

    X, Y = np.meshgrid(x, y)

    # compute velocity field

    dxdt = dynamics(X.flatten(), Y.flatten())
    U, V = dxdt[0].reshape(X.shape), dxdt[1].reshape(Y.shape)

    # plot streamlines
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(800 * px, 800 * px), layout="constrained")
    fig.set_dpi(150)

    plt.streamplot(X, Y, U, V, color="blue", linewidth=0.8, density=1.5)

    # add contour lines of speed for context
    speed = np.sqrt(U**2 + V**2)
    plt.contour(
        X, Y, speed, levels=8, colors="gray", linestyles="--", linewidths=0.5, alpha=0.7
    )

    # formatting
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Vector Field (Streamlines)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()

    if show:
        plt.show()

    # return the current figure handle
    return fig, ax
