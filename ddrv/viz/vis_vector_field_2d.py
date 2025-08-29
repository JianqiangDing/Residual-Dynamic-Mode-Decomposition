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
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                dxdt = dynamics(0, [X[i, j], Y[i, j]])
                U[i, j] = dxdt[0]
                V[i, j] = dxdt[1]
            except Exception:
                U[i, j] = np.nan
                V[i, j] = np.nan

    # plot streamlines
    plt.figure(figsize=(12, 9))

    start_points = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
    start_points = np.array([start_points[0].flatten(), start_points[1].flatten()]).T

    plt.streamplot(
        X, Y, U, V, start_points=start_points, color="blue", linewidth=0.8, density=1.5
    )

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
    return plt.gcf()
