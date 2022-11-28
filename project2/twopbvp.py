import numpy as np
import scipy.sparse as sparse
import numpy.typing as npt
from typing import Callable, Tuple


def two_p_BVP(
        x: npt.NDArray,
        f_vec: npt.NDArray,
        alpha: float,
        beta: float,
    ) -> Tuple[npt.NDArray, npt.NDArray]:

    """Solves the 2 point boundary value problem y''=f(x).

    Parameters:
    - x: the spatial grid.
    - f_vec: the function f evaluated on the grid.
    - alpha: the boundary condition at x=0.
    - beta: the boundary condition at x=L.

    Returns:
    - the spatial grid x.
    - the approximate solution y.
    """

    f_comp = f_vec[1:-1]
    N = np.size(x) - 2
    dx = (x[-1] - x[0]) / (N+1)

    diagonals = [[-2]*N, [1]*(N-1), [1]*(N-1)]
    D2 = 1/(dx**2) * sparse.diags(diagonals, [0, -1, 1])
    D2 = D2.tocsr()

    f_comp[0] += -alpha / (dx**2)
    f_comp[-1] += -beta / (dx**2)

    y = sparse.linalg.spsolve(D2, f_comp)
    a = np.array([alpha])
    b = np.array([beta])
    y = np.concatenate((a, y, b))

    return x, y


def get_f_vec(
        f: Callable[[float], float],
        L: float,
        N_inner: int
    ) -> Tuple[npt.NDArray, npt.NDArray]:

    """Returns the function f as a 1D array.

    Constructs an equidistant grid x over [0, L] with N_inner+2 points and
    evaluates the function f over it.

    Returns: the grid x and the array f_vec.
    """
    
    x = np.linspace(0, L, N_inner+2)
    f = np.vectorize(f)
    f_vec = f(x)
    return x, f_vec
