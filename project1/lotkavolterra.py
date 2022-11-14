from typing import Callable

import numpy as np
import numpy.typing as npt


class LotkaVolterra:
    """Represents the Lotka-Volterra problem x'=a*x-b*x*y, y'=c*x*y-d*y, where
    x models the prey population and y the predator population.
    
    Parameters:
    a, b, c, d: parameters governing the dynamics of the predator-prey system.
    x0: the initial prey population.
    y0: the initial predator population.
    """
    
    def __init__(self, a, b, c, d, x0 = 1, y0 = 1) -> None:
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._u0 = np.array([x0, y0])

    @property
    def u0(self) -> npt.NDArray:
        """Returns the initial values for the prey and predator populations
        as a 1d array.
        """
        return self._u0

    def get_f(self) -> Callable[[float, npt.NDArray], npt.NDArray]:
        """Returns a function f such that the Lotka-Volterra problem
        can be expressed as y'=f(t, y).
        """
        def f(_, u: npt.NDArray) -> npt.NDArray:
            """The function f in y'=f(t, y) for the Lotka-Volterra problem.
            
            Parameters:
            _: unused argument to fit the form f=f(t, y)
            u: a 2 element 1d array where the first element models the prey
            population and the second models the predator population.

            Returns: the function f(t, y).
            """
            return np.array([
                    self._a*u[0] - self._b*u[0]*u[1],
                    self._c*u[0]*u[1] - self._d*u[1]
                ])
        return f
