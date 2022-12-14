import numpy as np
import scipy.linalg as linalg

from numpy.typing import NDArray, ArrayLike
from typing import Tuple

def eulerstep(A: ArrayLike, uold: NDArray, dt: float) -> NDArray:
    """Takes one timestep of size dt using the explicit Euler method.

    Parameter:
    - A: a matrix.
    - uold: the previous timestep.
    - dt: the size of the step. 

    Returns: the next timestep. 
    """
    return uold + dt*A @ uold


def eulerint(
        A: ArrayLike,
        y0: NDArray,
        t0: float,
        tf: float,
        M: float,
        method: str = "explicit"
        ) -> Tuple[NDArray, NDArray]:
    """Takes M timesteps using an optional Euler method. 

    Parameters:
    - A: a matrix.
    - y0: The initial values.
    - t0: initial time.
    - tf: final time.
    - M: amount of timesteps.
    - method: can either be 'explicit' or 'implicit'. Specifies if an explicit 
      or implicit solver should be used. Default: 'explicit'.
    
    Returns:
    - tgrid: the time grid.
    - approx: the approximated N x (M+1) solution matrix.
    """
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


def TRstep(Tdx: ArrayLike, uold: NDArray, dt: float) -> NDArray:
    """Takes one timestep of size dt using the trapezoidal rule.

    Parameters:
    - Tdx: a matrix.
    - uold: the previous timestep.
    - dt: the siza of the step.

    Returns: the next timestep.
    """
    A = (
        linalg.inv((np.eye(*np.shape(Tdx)) - dt / 2 * Tdx))
        @ (np.eye(*np.shape(Tdx)) + dt / 2 * Tdx) 
        ) 
    return A @ uold