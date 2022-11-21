from typing import Callable, Tuple
import numpy.typing as npt

from twopbvp import get_f_vec, two_p_BVP


class Beam:
    """Represents the beam equation

            M'' = q(x),
            u'' = M(x)/(E*I(x)).
    
    Parameters:
    - q: load density.
    - L: length of the beam.
    - E: Young's modulus of elasticity
    - I: The beam's cross-section moment of inertia.
    """

    def __init__(
            self,
            q: Callable[[float], float],
            L: float,
            E: float,
            I: Callable[[float, float], float]
        ):

        self.q = q
        self.L = L
        self.E = E
        self.I = I
    
    def solve(self, N: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """Solves the beam equation using a 2pBVP solver.

        Parameters:
        - N: number of inner grid points.

        Returns: the spatial grid x and the approximate solution u.
        """
        x, M = two_p_BVP(*get_f_vec(self.q, self.L, N), 0, 0)
        x, u = two_p_BVP(x, M / (self.E*self.I(x, self.L)), 0, 0)
        return x, u