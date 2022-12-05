import numpy as np
import scipy.sparse as sparse
from timestep import eulerint

from numpy.typing import NDArray
from typing import Callable, Tuple


def create_sparse_Tdx(N: float) -> sparse.csr_array:
    dx = 1 / (N + 1)
    diagonals = [[-2]*N, [1]*(N-1), [1]*(N-1)]
    Tdx = 1/(dx**2) * sparse.diags(diagonals, [0, -1, 1])
    Tdx = Tdx.tocsr()
    return Tdx


def solve_diffusion(N: float, M: float, t_end: float, g: Callable) -> Tuple[NDArray, NDArray]:
    Tdx = create_sparse_Tdx(N)
    dt = t_end / (M + 1)
    dx = 1 / (N + 1)
    xgrid = np.linspace(dx, 1 - dx, N)
    y0 = np.vectorize(g)(xgrid)
    tgrid, approx = eulerint(Tdx, y0, 0, t_end, M)
    zero_row = np.zeros(np.shape(approx[0,:]))
    approx = np.vstack((zero_row, approx, zero_row))
    xgrid = np.concatenate(([0], xgrid, [1]))
    return tgrid, xgrid, approx
