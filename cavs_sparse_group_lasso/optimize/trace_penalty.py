import numpy as np

def compute_trace_penalty(B_j,Lp):
    """
    Computes the Laplacian trace penalty for a single row of B.

    Parameters:
        B_j : np.ndarray
            Row vector of B (1 x N), usage of atom j across all segments
        Lp : np.ndarray
            Segment-level Laplacian matrix (N x N)

    Returns:
        float : The scalar trace penalty B_j @ Lp @ B_j.T

    """

    return B_j @ Lp @ B_j.T #scalar