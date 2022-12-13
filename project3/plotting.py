import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from diffusion import solve_diffusion


def task12():
    g = lambda x: np.exp(-1000*(x-0.5)**2)

    tgrid, xgrid, approx = solve_diffusion(100, 1000, 0.1, g, "implicit")
    T, X = np.meshgrid(tgrid, xgrid)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T, X, approx, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()


if __name__ == "__main__":
    task12()