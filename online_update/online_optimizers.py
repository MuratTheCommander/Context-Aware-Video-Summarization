import numpy as np
import cvxpy as cp

def optimize_Df_online(P, Q, Df, max_iter=1, epsilon=1e-6):
    """
    Online dictionary update from Mairal et al. (2010), using A = P = ∑ B B^T, and B = Q = ∑ X B^T

    Args:
        P (np.ndarray): shape (K, K), accumulated B B^T
        Q (np.ndarray): shape (d, K), accumulated X B^T
        Df (np.ndarray): shape (d, K), current dictionary
        max_iter (int): number of passes over atoms (usually 1 is sufficient in online updates)
        epsilon (float): small value to avoid divide-by-zero

    Returns:
        Df (np.ndarray): updated dictionary
    """

    K = Df.shape[1]

    for _ in range(max_iter):
        for j in range(K):
            if P[j, j] < epsilon:
                continue  # Atom was not used, skip update

            # Compute update direction for atom j
            u_j = (Q[:, j] - Df @ P[:, j] + Df[:, j] * P[j, j]) / P[j, j]

            # Normalize to unit norm
            Df[:, j] = u_j / max(np.linalg.norm(u_j), epsilon)

    return Df

def sparse_code_online(X_t,Df,lambda1=0.05):
    """
    Solves: min_B ||X_t - Df B||_2^2 + lambda1 * ||B||_1
    """
    K = Df.shape[1]
    
    B = cp.Variable((K,1)) # Coefficient vector (sparse)

    objective = cp.Minimize(cp.norm2(X_t - Df@B)**2 + lambda1*cp.norm1(B))

    problem = cp.Problem(objective)

    problem.solve(cp.SCS)

    return B.value
