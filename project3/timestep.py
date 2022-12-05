import numpy as np

def eulerstep(A, uold, dt):
    return uold + dt*A @ uold


def eulerint(A, y0, t0, tf, M):
    dt = (tf - t0)/M
    tgrid = np.zeros(M+1)
    tgrid[0] = t0
    approx = np.zeros((np.size(y0), M+1))
    approx[:, 0] = y0
    t = t0
    for i in range(1,M+1):
        approx[:, i] = eulerstep(A, approx[:, i-1], dt)
        t = t + dt
        tgrid[i] = t
    return tgrid, approx