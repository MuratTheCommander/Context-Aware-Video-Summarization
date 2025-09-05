import numpy as np
from optimize.update_row import update_row

def cavs_sgl_solver(X,Df,Lp,alpha1,alpha2,alpha3,max_iter=100,tol=1e-4,B_init=None):
    """
    Implements Algorithm 1 

    Parameters:
        X : np.ndarray
            Feature matrix of shape (d, N), where each column is a segment descriptor
        Df : np.ndarray
            Feature dictionary of shape (d, K), each column is a dictionary atom
        Lp : np.ndarray
            Segment-level Laplacian matrix (N x N)
        alpha1, alpha2, alpha3 : float
            Regularization parameters
        max_iter : int
            Max number of iterations
        tol : float
            Convergence tolerance (L2 norm between B matrices)
        B_init : np.ndarray or None
            Optional initialization for B (K x N)

    Returns:
        B : np.ndarray
            Optimized coefficient matrix (K x N)

    """

    d,N = X.shape

    d_dF,K = Df.shape

    assert d == d_dF , "Mismatch: feature dimension in X and Df must match."

    #Initialize B
    if B_init is not None:
        B = B_init.copy()
    else:
        B = np.zeros((K,N))
    
    for iteration in range(max_iter):
        B_old = B.copy()

        #Loop over each row (i.e., atom j)
        for j in range(K):
            B[j,:] = update_row(j,X,Df,B,Lp,alpha1,alpha2,alpha3)

        #Convergence Check
        diff = np.linalg.norm(B - B_old)
        print(f"[Iter {iteration+1}] ΔB: {diff:.6f}")
        if diff < tol:
            print(f"[INFO] Converged at iteration {iteration+1}")
            break

    return B

