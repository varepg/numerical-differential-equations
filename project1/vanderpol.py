from typing import Callable

import numpy as np
import numpy.typing as npt

class VanDerPol:

    def __init__(self, mu, x0 = 2, y0 = 0):
        self._mu = mu
        self._u0 = np.array([x0, y0])
    

    def get_f(self):
        def f(_, u: npt.NDArray):
            return np.array([u[1], self._mu * (1 - u[0]**2) * u[1] - u[0]])
        return f
    
    @property
    def u0(self):
        return self._u0
