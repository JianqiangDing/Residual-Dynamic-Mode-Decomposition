# this function is used to visualize the eigenvalues on the complex plane

import matplotlib.pyplot as plt


def vis_eigenvalues(eigenvalues, title="Eigenvalues on Complex Plane"):

    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(800 * px, 800 * px), layout="constrained")
    fig.set_dpi(150)

    ax.scatter(eigenvalues.real, eigenvalues.imag)
    ax.set_title(title)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.axhline(0, color="black", linewidth=0.5, ls="--")
    ax.axvline(0, color="black", linewidth=0.5, ls="--")
    ax.grid()
    plt.show()
