import numpy as np
import scipy.sparse as sparse
from convdiv import create_circulant_Sdx, create_circulant_Tdx

from typing import Tuple
from numpy.typing import NDArray


def burger(u: NDArray, d: float, dt: float, dx: float) -> NDArray:
    """Performs one time step in the viscous Burgers equation.

    Parameters:
    - u: the previous timestep over a spatial grid.
    - d: constant of diffusivity.
    - dt: size of timestep.
    - dx: size of spatial step.

    Returns: the next step.
    """

    Sdx = create_circulant_Sdx(dx)
    Tdx = create_circulant_Tdx(dx)

    ux = Sdx@u
    uxx = Tdx@u
    u2 = u*u
    ux2 = ux*ux

    lw = u - dt*u*ux + dt**2/2*(2*u*ux2 +u2 + uxx)
    N = int(1/dx)

    return (
        sparse.linalg.inv(sparse.eye(N, format="csr")
        - d*dt/2*Tdx) @ (lw + d*dt/2*uxx)
    )


def burger_stepper(u0: NDArray, d: float, M: float) -> Tuple[NDArray, NDArray]:
    """Solves the viscous Burgers equation with initial condition u0.

    Parameters:
    - u0: initial condition.
    - d: constant of diffusivity.
    - M: number of timesteps.

    Returns: the solution matrix u and the timegrid tgrid.
    """
    u = np.reshape(u0, (np.size(u0),1))
    dt = 1/M
    dx = 1/np.size(u)
    tgrid = np.linspace(0, 1, M)

    for _ in range(1, M):
        u = np.column_stack((u, burger(u[:,-1], d, dt, dx)))

    return u, tgrid
