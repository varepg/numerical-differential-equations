import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from diffusion import solve_diffusion
from laxwen import get_nbr_timesteps, get_stepsizes, laxwen_stepper
from convdiv import convdiv_stepper
from burger import burger_stepper


def task11():
    g = lambda x: np.exp(-1000*(x-0.5)**2)
    N = 25
    M = 1000
    tgrid, xgrid, approx = solve_diffusion(N, M, 1, g)
    T, X = np.meshgrid(tgrid, xgrid)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T, X, approx, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(f"Explicit solution for N = {N}, M = {M}")
    plt.show()


def task12():
    g = lambda x: np.exp(-1000*(x-0.5)**2)
    N = 100
    M = 1000
    tgrid, xgrid, approx = solve_diffusion(N, M, 0.1, g, "implicit")
    T, X = np.meshgrid(tgrid, xgrid)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T, X, approx, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(f"Implicit solution for N = {N}, M = {M}")
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


def task31():
    g = lambda x: np.exp(-100*(x-0.5)**2)
    N = 100
    xgrid = np.linspace(0,1,N)
    a = 1
    d = 0.1
    M = 100
    u0 = g(xgrid)
    u, tgrid = convdiv_stepper(u0, a, d, M)

    T, X = np.meshgrid(tgrid, xgrid)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T, X, u, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()


def task312():
    g = lambda x: 1 if np.abs(x-0.3)<0.1 else 0
    g = np.vectorize(g)
    N = 100
    xgrid = np.linspace(0,1,N)
    a = 1
    d = 0.1
    M = 100
    u0 = g(xgrid)
    u, tgrid = convdiv_stepper(u0, a, d, M)

    T, X = np.meshgrid(tgrid, xgrid)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(T, X, u)#, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()


def task313():
    g = lambda x: np.exp(-100*(x-0.5)**2)
    N = 100
    xgrid = np.linspace(0,1,N)
    a = -10
    d = 0.1
    M = 100
    u0 = g(xgrid)
    u, tgrid = convdiv_stepper(u0, a, d, M)

    T, X = np.meshgrid(tgrid, xgrid)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T, X, u, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()


def task41():
    A = 2
    g = lambda x: A*np.exp(-100*(x-0.5)**2)
    N = 300
    xgrid = np.linspace(0,1,N)
    d = 0.005
    M = 1000
    u0 = g(xgrid)
    u, tgrid = burger_stepper(u0, d, M)

    T, X = np.meshgrid(tgrid, xgrid)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(T, X, u)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(f"A={A}, d={d}")
    plt.show()


if __name__ == "__main__":
    task21()