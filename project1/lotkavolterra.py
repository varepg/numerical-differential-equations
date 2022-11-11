from typing import Callable

import numpy as np
import numpy.typing as npt


class LotkaVolterra:
    def __init__(self, a, b, c, d, x0 = 1, y0 = 1) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self._u0 = np.array([x0, y0])

    @property
    def u0(self):
        return self._u0
    
    def get_f(self) -> Callable[[float, npt.NDArray], npt.NDArray]:
        """ 
        Returns: the function f in y'=f for the lotka-volterra problem
        """
        def f(t: float,u: npt.NDArray):
            A = np.array([ 
                [self.a, 0],
                [0, -self.b]
            ])
            B = np.array([ 
                [-self.b, 0],
                [0, self.c]
            ])
            C = np.array([ 
                [0, 1],
                [1, 0]
            ])
            D = np.array([1, 1])
            return A @ u + (B @ (np.dot(u, C @ u) * D)/2)
        return f



