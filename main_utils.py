import numpy as np
from cavs_sparse_group_lasso.optimize import cavs_sgl_solver
from learn_dictionary import optimize_Df

def compute_total_loss(X, Df, B, Lp, alpha1, alpha2, alpha3):
    recon_error = 0.5 * np.sum((X - Df @ B)**2)
    trace_penalty = alpha1 * np.trace(B @ Lp @ B.T)
    group_l2 = alpha2 * np.sum(np.linalg.norm(B, axis=1))  # sum over rows
    l1 = alpha3 * np.sum(np.abs(B))
    return recon_error + trace_penalty + group_l2 + l1

def train_Df_B(X, Df_init, Lp, alpha1, alpha2, alpha3,
               max_outer_iter=10, max_inner_iter=20, tol=1e-4, verbose=True):
    """
    Alternating optimization loop for Equation (2) of CAVS:
    - Fix Df, optimize B
    - Fix B, optimize Df
    """

    d, N = X.shape
    K = Df_init.shape[1]

    Df = Df_init.copy()
    B = np.zeros((K, N))

    for outer_iter in range(max_outer_iter):
        if verbose:
            print(f"\n[Outer Iter {outer_iter+1}]")

        # Step 1: Optimize B
        B = cavs_sgl_solver(X, Df, Lp, alpha1, alpha2, alpha3,
                            max_iter=max_inner_iter, tol=tol, B_init=B)

        # Step 2: Optimize Df
        Df = optimize_Df(X, B, Df)

        # Step 3: Compute and print loss
        loss = compute_total_loss(X, Df, B, Lp, alpha1, alpha2, alpha3)
        if verbose:
            print(f"  Total loss: {loss:.6f}")
            print(f"  ||Df||_F: {np.linalg.norm(Df):.4f} | ||B||_F: {np.linalg.norm(B):.4f}")

        # Optional: early stopping (loss change threshold, etc.)

    return Df, B

def save_video_segment(segment,segment_name):
    None #will be implemented later
