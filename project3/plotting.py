import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from diffusion import solve_diffusion
from laxwen import get_nbr_timesteps, get_stepsizes, laxwen_stepper


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


def task21():
    g = lambda x: np.exp(-100*(x-0.5)**2)
    N = 100
    xgrid = np.linspace(0,1,N)
    amu = 1
    a = 0.5
    dt, dx = get_stepsizes(a, amu, N)
    dt = amu * dx / a
    u0 = g(xgrid)
    M = get_nbr_timesteps(5, dt)
    tgrid = np.linspace(0, 5, M)
    u, norms = laxwen_stepper(u0, amu, M)
    T, X = np.meshgrid(tgrid, xgrid)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T, X, u, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()
    plt.figure(2)
    plt.plot(tgrid, norms)
    plt.show()


if __name__ == "__main__":
    task21()