import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

from numpy.typing import NDArray
from typing import Callable, Tuple

import matplotlib.pyplot as plt


class TISE:
    """Represents the time independant Schrödinger equation (TISE)

            psi'' - V(x)*psi = -E * psi.
    
    Parameters:
    - V(x): one-dimensional potential.
    - L: end point of the interval, which starts at 0.
    """

    def __init__(self, V: Callable, L: float):
        self.V = V
        self.L = L

    def solve(
            self,
            N: int,
            k: int = 6
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Solves the TISE using a 2pBVP solver.

        Parameters:
        - N: number of inner grid points.
        - k: number of eigenvalues and eigenvectors to return. 

        Returns: the spatial grid x, k normalized eigenvectors psi_normed,
        the corresponding eigenvalues E and the probability densities prob.
        """
        dx = self.L / (N+1)
        x_comp = np.linspace(dx, self.L-dx, N)

        V = np.vectorize(self.V)
        Vdiag = sparse.diags(V(x_comp), 0)

        diags = [[-2]*N, [1]*(N-1), [1]*(N-1)]
        H = -1/(dx**2) * sparse.diags(diags, [0, -1, 1]) + Vdiag
        H = H.tocsr()

        E, psi = sparse.linalg.eigs(H, k=k, which="SM")

        # Dirichlet conditions u(0)=u(L)=0
        zero_row = np.zeros(np.shape(psi[0,:]))
        psi = np.vstack((zero_row, psi, zero_row))

        psi_normed = self._normalize_columns(psi)
        prob = np.power(np.abs(psi_normed), 2)

        x = np.concatenate(([0], x_comp, [self.L]))

        return x, psi_normed, E, prob

    @staticmethod
    def _normalize_columns(A: NDArray) -> NDArray:
        """Normalizes the columns in A according to the euclidean norm. 

        Parameters:
        A: a matrix.

        Returns: a matrix with normalized columns. 
        """
        norms = np.apply_along_axis(linalg.norm, 0, A)

        # norm = 0 not handled as eigenvectors are non-zero

        return A / np.power(norms[None, :], 2)

    @staticmethod
    def plot(
            x: NDArray,
            psi: NDArray,
            E: NDArray,
            prob: NDArray,
            *,
            psi_savepath: str = "",
            prob_savepath: str = "",
            scale_psi: float = 100,
            scale_prob: float = 1000,
            title: str = ""
        ) -> Tuple[plt.Axes, plt.Figure, plt.Axes, plt.Figure]:
        """Scales and plots the eigenvectors and probabilities at their respective energy levels. 

        Parameters:
        x: spatial grid.
        psi: TISE eigenvectors.
        E: corresponding eigenvalues.
        prob: corresponding probbility densities.
        psi_savepath: saves the psi-plot to path. Default: does not save.
        prob_savepath: saves the prob-plot to path. Default: does not save.
        scale_psi: scales psi by a factor of scale_psi. Default: 100.
        scale_prob: scales prob by a factor of scale_prob. Default: 1000.
        title: adds a title to the plots. Default: no title.
        """

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        # rescaling for vizualisation
        psi *= scale_psi
        prob *= scale_prob

        for i, e in np.ndenumerate(E):

            e = np.float64(e)

            psi[:, i] += e
            prob[:, i] += e

            ax1.axhline(y=e, color="black", linestyle="--", linewidth=0.8)
            ax1.plot(x, psi[:, i])
            ax2.axhline(y=e, color="black", linestyle="--", linewidth=0.8)
            ax2.plot(x, prob[:, i])
        
        ax1.set_xlabel("x")
        ax1.set_ylabel("Wavefunction, Energy")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Probability, Energy")
        if title:
            ax1.set_title(title)
            ax2.set_title(title)
        plt.show()

        if psi_savepath:
            fig1.savefig(psi_savepath)
        if prob_savepath:
            fig2.savefig(prob_savepath)
        return fig1, ax1, fig2, ax2
        
