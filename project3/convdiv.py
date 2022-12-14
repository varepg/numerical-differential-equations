import numpy as np
import scipy.sparse as sparse

from typing import Tuple
from numpy.typing import NDArray


def create_circulant_Tdx(dx: float) -> sparse.csr_array:
    """Creates a sparse matrix representation of the second derivative operator
    with periodic boundary conditions.

    Parameters:
    - dx: stepsize of spatial grid.

    Returns: the sparse matrix in CSR format.
    """
    N = int(1 / dx)
    diagonals = [[-2]*N, [1]*(N-1), [1]*(N-1), [1], [1]]
    Tdx = 1/(dx**2) * sparse.diags(diagonals, [0, -1, 1, -N + 1, N - 1])
    Tdx = Tdx.tocsr()
    return Tdx


def create_circulant_Sdx(dx: float) -> sparse.csr_array:
    """Creates a sparse matrix representation of the first derivative operator
    with periodic boundary conditions.

    Parameters:
    - dx: stepsize of spatial grid.

    Returns: the sparse matrix in CSR format.
    """
    N = int(1 / dx)
    diagonals = [[-1]*(N-1), [1]*(N-1), [1], [-1]]
    Sdx = 1/(2*dx) * sparse.diags(diagonals, [ -1, 1, -N + 1, N - 1])
    Sdx = Sdx.tocsr()
    return Sdx


def convdiv(u: NDArray, a: float, d: float, dt: float, dx: float) -> NDArray:
    """Performs one timestep in the convection-diffusion equation.

    Parameters:
    - u: the previous timestep over a spatial grid.
    - a: constant of convection.
    - d: constant of diffusivity.
    - dt: size of timestep.
    - dx: size of spatial step.

    Returns: the next step.
    """
    Tdx = create_circulant_Tdx(dx)
    Sdx = create_circulant_Sdx(dx)

    A = d*Tdx - a*Sdx

    N = int(1/dx)

    B = (sparse.eye(N, format="csr") - dt/2*A)
    C = (sparse.eye(N, format="csr") + dt/2*A)

    return sparse.linalg.inv(B) @ C @ u


def convdiv_stepper(
        u0: NDArray,
        a: float,
        d: float,
        M: int
    ) -> Tuple[NDArray, NDArray]:
    """Solves the convection-diffusion equation with initial condition u0.

    Parameters:
    - u0: initial condition.
    - a: constant of convection.
    - d: constant of diffusivity.
    - M: number of timesteps.

    Returns: the solution matrix u and the timegrid tgrid.
    """

    u = np.reshape(u0, (np.size(u0),1))
    dt = 1/M
    dx = 1/np.size(u)
    tgrid = np.linspace(0, 1, M)

    for _ in range(1, M):
        u = np.column_stack((u, convdiv(u[:,-1], a, d, dt, dx)))

    return u, tgrid
