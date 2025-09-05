import numpy as np
from optimize.group_zero_test import is_group_zero
from scipy.optimize import minimize_scalar

def update_row(j, X, Df, B, Lp, alpha1, alpha2, alpha3):
    """
    Update row j of B (i.e., usage of dictionary atom j across all segments)
    """

    K, N = B.shape
    d = X.shape[0]

    d_j = Df[:, j]  # shape: (d,)
    B_j_new = B[j, :].copy()

    # Step 1: Compute residual matrix R_j = X - sum_{k ≠ j} Df_k * B_k
    R = X.copy()
    for k in range(K):
        if k != j:
            R -= np.outer(Df[:, k], B[k, :])

    # Step 2: Group zero check
    Z_j = d_j[:, np.newaxis]  # shape: (d, 1)
    a = Z_j.T @ R  # shape: (1, N)
    a = a.flatten()  # shape: (N,)
    
    if is_group_zero(Z_j, a, lambda1=alpha2, lambda2=alpha3):
        return np.zeros_like(B_j_new)

    # Step 3: Coordinate-wise update
    for i in range(N):
        r_i = R[:, i] + d_j * B[j, i]
        dot = d_j.T @ r_i

        if np.abs(dot) < alpha3:
            B_j_new[i] = 0.0
        else:
            def loss_fn(theta):
                recon = r_i - d_j * theta
                data_fit = 0.5 * np.sum(recon ** 2)
                l1_penalty = alpha3 * np.abs(theta)

                # Trace term - compute fresh each time
                diag_term = Lp[i, i] * theta ** 2
                cross_term = 2 * theta * (np.dot(Lp[i, :], B_j_new) - Lp[i, i] * B_j_new[i])
                trace_penalty = alpha1 * (diag_term + cross_term)

                return data_fit + l1_penalty + trace_penalty

            result = minimize_scalar(loss_fn, method='brent')
            B_j_new[i] = result.x

    return B_j_new