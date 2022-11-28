import numpy as np
import scipy.sparse as sparse


def SL_solve(L: int, N: int, k: int = 3):
    """Solves the Sturm-Louiville problem -u''=w*u on the interval [0, L]

    Uses boundary conditions u(0)=0, u'(L)=0.

    Paramters:
    - L: length of interval.
    - N: number of computational points.
    - k: number of eigenvalue-eigenvector pairs to compute (must be < N+1).

    Returns:
    - x: the spatial grid.
    - v, w: eigenvectors v and eigenvalues w, ordered by smallest magnitude,
    such that v[:,i] is the vector corresponding to the value w[i].
    """

    dx = L / (N)
    x = np.linspace(0, L, N+1)

    diags = [[-2]*N, [1]*(N-1), [1]*(N-1)]
    D2 = sparse.diags(diags, [0, -1, 1])
    D2 = D2.tocsr()

    # modifying for Neumann condition at x = L
    D2[-1,-2] = 2
    
    D2 *= 1/(dx**2)
    w, v = sparse.linalg.eigs(D2, k=k, which="SM")

    ## sort by increasing size of absolute value of eigenvalue
    #idx_sort = np.argsort(np.abs(w))
    #w = w[idx_sort]
    #v = v[:, idx_sort]

    zero_row = np.zeros(np.shape(v[0,:]))
    v = np.vstack((zero_row, v))

    return x, v, w
