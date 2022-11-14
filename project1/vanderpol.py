from typing import Callable

import numpy as np
import numpy.typing as npt


class VanDerPol:
    """Represents the van der Pol problem y1'=y2, y2'=mu*(1-y1^2)*y2-y1.
    
    Parameters:
    mu: non-linear damping coefficient.
    x0: initial value of y1.
    y0: initial value of y2.
    """

    def __init__(self, mu: float, x0: float = 2, y0: float = 0):
        self._mu = mu
        self._u0 = np.array([x0, y0])

    @property
    def u0(self) -> npt.NDArray:
        """Returns the initial values as a 1d array."""
        return self._u0

    def get_f(self) -> Callable[[float, npt.NDArray], npt.NDArray]:
        """Returns a function f such that the van der Pol problem
        can be expressed as y'=f(t, y).
        """
        def f(_, u: npt.NDArray) -> npt.NDArray:
            """The function f in y'=f(t, y) for the van der Pol problem.
            
            Parameters:
            _: unused argument to fit the form f=f(t, y)
            u: a 2 element 1d array where the second element represents the
            derivative of the first.

            Returns: the function f(t, y).
            """
            return np.array([u[1], self._mu * (1 - u[0]**2) * u[1] - u[0]])
        return f
