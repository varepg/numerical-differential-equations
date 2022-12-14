import numpy as np
import scipy.sparse as sparse
from timestep import eulerint

from numpy.typing import NDArray
from typing import Callable, Tuple


def create_sparse_Tdx(N: float) -> sparse.csr_array:
    """Creates a sparse N x N discretization of the second derivative operator.

    Dirichlet boundary conditions implied.
    """
    dx = 1 / (N + 1)
    diagonals = [[-2]*N, [1]*(N-1), [1]*(N-1)]
    Tdx = 1/(dx**2) * sparse.diags(diagonals, [0, -1, 1])
    Tdx = Tdx.tocsr()
    return Tdx


def solve_diffusion(
        N: float,
        M: float,
        t_end: float,
        g: Callable[[float], float],
        method: str="explicit"
    ) -> Tuple[NDArray, NDArray]:

    """Solves the diffusion problem du/dt = d2u/dx2.

    Solves the problem on the spatial interval [0, 1] and the time interval
    [0, t_end] using homogenous Dirichlet conditions.

    Parameters:
    - N: number of spatial computational points.
    - M: number of samples in time.
    - t_end: final time.
    - g: function describing the initial conditions, u(t=0,x)=g(x).
    - method: can either be 'explicit' or 'implicit'. Specifies if an explicit
      or implicit solver should be used. Default: 'explicit'.
    
    Returns:
    - tgrid: the time grid.
    - xgrid: the spatial grid, including endpoints.
    - approx: the approximated (N+2) x M solution matrix.
    """

    Tdx = create_sparse_Tdx(N)
    dt = t_end / (M + 1)
    dx = 1 / (N + 1)
    xgrid = np.linspace(dx, 1 - dx, N)
    y0 = np.vectorize(g)(xgrid)

    if method != "explicit" and method != "implicit":
        raise ValueError("Supported methods: 'explicit' and 'implicit'")

    tgrid, approx = eulerint(Tdx, y0, 0, t_end, M, method)

    zero_row = np.zeros(np.shape(approx[0,:]))
    approx = np.vstack((zero_row, approx, zero_row))
    xgrid = np.concatenate(([0], xgrid, [1]))
    return tgrid, xgrid, approx
