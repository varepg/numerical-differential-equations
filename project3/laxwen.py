import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg

from typing import Tuple
from numpy.typing import NDArray


def laxwen(u: NDArray, amu: float) -> NDArray:
    N = np.size(u)
    diagonals = [[(1 - amu**2)]*N, [amu/2*(1 + amu)]*(N-1), [-amu/2*(1-amu)]*(N-1), [amu/2*(1 + amu)], [-amu/2*(1 - amu)]]
    A = sparse.diags(diagonals, [0, -1, 1, N - 1, -N + 1])
    A = A.tocsr()
    return A @ u


def laxwen_stepper(u0: NDArray, amu: float, M: int) -> Tuple[NDArray, NDArray]:
    u = np.reshape(u0, (np.size(u0),1))
    norms = np.array([linalg.norm(u0) / np.size(u0)])

    for _ in range(1, M):
        u = np.column_stack((u, laxwen(u[:,-1], amu)))
        norms = np.append(norms, linalg.norm(u[:,-1]) / np.size(u[:,-1]))

    return u, norms


def get_stepsizes(a, amu, N):
    dx = 1 / N
    dt = amu * dx/a

    return dt, dx


def get_nbr_timesteps(t, dt):
    return int(t / dt)
