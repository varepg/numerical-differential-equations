import numpy as np
import scipy.linalg as linalg


def eulerstep(A, uold, dt):
    return uold + dt*A @ uold


def eulerint(A, y0, t0, tf, M, method: str = "explicit"):
    dt = (tf - t0)/M
    tgrid = np.zeros(M+1)
    tgrid[0] = t0
    approx = np.zeros((np.size(y0), M+1))
    approx[:, 0] = y0
    t = t0

    if method == "explicit":
        step = eulerstep
    elif method == "implicit":
        step = TRstep
    else:
        raise ValueError("Supported methods: 'explicit' and 'implicit'")

    for i in range(1,M+1):
        approx[:, i] = step(A, approx[:, i-1], dt)
        t = t + dt
        tgrid[i] = t
    return tgrid, approx


def TRstep(Tdx, uold, dt):
    A = (
        linalg.inv((np.eye(*np.shape(Tdx)) - dt / 2 * Tdx))
        @ (np.eye(*np.shape(Tdx)) + dt / 2 * Tdx) 
        ) 
    return A @ uold