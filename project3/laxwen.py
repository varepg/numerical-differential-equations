import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg

from typing import Tuple
from numpy.typing import NDArray


def laxwen(u: NDArray, amu: float) -> NDArray:
    """Solves the advection equation for one timestep using the Lax-Wendroff 
    scheme.

    Parameters:
    u: the previous timestep. 
    amu: the prduct a * dt / dx.

    Returns: the next timestep.
    """
    N = np.size(u)
    diagonals = [[(1 - amu**2)]*N, [amu/2*(1 + amu)]*(N-1), [-amu/2*(1-amu)]*(N-1), [amu/2*(1 + amu)], [-amu/2*(1 - amu)]]
    A = sparse.diags(diagonals, [0, -1, 1, N - 1, -N + 1])
    A = A.tocsr()
    return A @ u


def laxwen_stepper(u0: NDArray, amu: float, M: int) -> Tuple[NDArray, NDArray]:
    """Solves the advection equation for M timesteps using the Lax-Wendroff 
    scheme.

    Parameters:
    - u0: the initial values. 
    - amu: the prduct a * dt / dx.
    - M: amount of timesteps.

    Returns:
    - u: the approximated N x M solution matrix.
    - norms: the RMS-norm for each timestep. 
    """
    u = np.reshape(u0, (np.size(u0),1))
    norms = np.array([linalg.norm(u0) / np.size(u0)])

    for _ in range(1, M):
        u = np.column_stack((u, laxwen(u[:,-1], amu)))
        norms = np.append(norms, linalg.norm(u[:,-1]) / np.size(u[:,-1]))

    return u, norms


def get_stepsizes(a: float, amu: float, N: float) -> Tuple[float, float]:
    """Calculates the stepsizes dt and dx.
    
    Parameters:
    - a: the advection constant.
    - amu: the product a * dt / dx.
    - N: number of values on spatial grid.

    Returns:
    dt: size of timestep.
    dx: size of spatial step
    """
    dt = amu * dx/a
    dx = 1 / N

    return dt, dx


def get_nbr_timesteps(t: float, dt: float) -> float:
    """Calculates number of timesteps needed.

    Parameters:
    - t: time span.
    - dt: size of time step.

    Returns: number of time steps needed.
    """
    return int(t / dt)
