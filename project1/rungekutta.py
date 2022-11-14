import numpy as np
import scipy.linalg as linalg
import numpy.typing as npt
from typing import Callable, Tuple


def RK4step(
        f: Callable[[float, npt.NDArray], npt.NDArray],
        t_old: float,
        u_old: npt.NDArray,
        h: float
    ) -> npt.NDArray:

    """Performs one step for the equation y'=f(t, y) using the RK4 schema
    
    Returns: the next step
    """

    dY1 = f(t_old, u_old)
    dY2 = f(t_old + h/2, u_old + h * dY1 / 2)
    dY3 = f(t_old + h/2, u_old + h * dY2 / 2)
    dY4 = f(t_old + h, u_old + h * dY3)
    
    return u_old + h / 6 * (dY1 + 2 * dY2 + 2 * dY3 + dY4)


def RK4_test_solver(
        f: Callable[[float, npt.NDArray], npt.NDArray],
        true_solution:  Callable[[float], npt.NDArray],
        u0: npt.NDArray,
        t0: float, tf: float, N: int
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

    """Solves the equation y'=f(t, y) for a test function f for which the
    solution true_solution is known, using the RK4 method.
    
    Returns: the time grid t_grid, the solution approx and an array of the
    errors after each step.
    """

    h = (tf - t0) / N
    tgrid = np.linspace(t0, tf, N + 1) # N RK steps => N+1 samples

    approx = np.zeros([np.size(u0), N + 1])
    approx[:, 0] = u0

    err = np.zeros([np.size(u0), N + 1]) # error idx 0 is 0

    t = t0
    for k in range(N):
        approx[:, k + 1] = RK4step(f, t, approx[:, k], h)
        
        t += h
        
        err[:, k + 1] = approx[:, k + 1] - true_solution(t)

    return tgrid, approx, err


def RK34step(
        f: Callable[[float, npt.NDArray], npt.NDArray],
        t_old: float,
        u_old: npt.NDArray,
        h: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:

    """Performs one step for the equation y'=f(t, y) using the RK4 schema and
    estimates the local error using the embedded RK3.
    
    Returns: the next step, and the local error estimate for the next step
    """
    
    dY1 = f(t_old, u_old)
    dY2 = f(t_old + h/2, u_old + h * dY1 / 2)
    dY3 = f(t_old + h/2, u_old + h * dY2 / 2)
    dZ3 = f(t_old + h, u_old - h * dY1 + 2 * h * dY2)
    dY4 = f(t_old + h, u_old + h * dY3)

    u_new = u_old + h / 6 * (dY1 + 2 * dY2 + 2 * dY3 + dY4)
    local_err_new = h / 6 * (2 * dY2 + dZ3 - 2 * dY3 - dY4)

    return u_new, local_err_new


def new_step(
        tol: float,
        err: float,
        err_old: float,
        h_old: float,
        k: int
    ) -> float:

    """Calculates the next step size for the adaptive RK34 solver using the
    previous one, keeping a constant error tolerance.

    Returns: new step size
    """

    return (
        (tol / err) ** (2 / (3 * k))
        * (tol / err_old) ** (-1 / (3 * k)) * h_old
    )

def adaptive_RK34(
        f: Callable[[float, npt.NDArray], npt.NDArray],
        t0: float,
        tf: float,
        y0: npt.NDArray,
        tol: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:

    """Solves the equation y'=f(t, y) using an adaptive RK34 scheme.
    
    Returns: the time grid t_grid and the solution y
    """
    
    h = np.abs(tf - t0) * tol ** (1 / 4) / (100 * (1 + linalg.norm(f(t0, y0))))
    y = np.reshape(y0, (np.size(y0), 1))
    t_grid = np.array([t0])
    err_old = tol
    t = t0

    while (t + h) < tf:
        y_new, err = RK34step(f, t_grid[-1], y[:, -1], h)
        y = np.column_stack((y, y_new))
        err = linalg.norm(err)
        h_old = h
        h = new_step(tol, err, err_old, h_old, 4)
        err_old = err
        t += h
        t_grid = np.append(t_grid, t)
    
    h_last = tf - t_grid[-1]
    y_new, err = RK34step(f, t_grid[-1], y[:, -1], h_last)
    t_grid = np.append(t_grid, tf)
    y = np.column_stack((y, y_new))

    return t_grid, y
