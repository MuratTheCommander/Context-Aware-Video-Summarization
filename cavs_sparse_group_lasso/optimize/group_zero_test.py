import numpy as np

def is_group_zero(x_group,r,lambda1,lambda2):

    """
    Returns True if the group should be set to zero based on the J(t̂) ≤ 1 condition
    """

    #Step 1: Compute a = Zᵀ r
    a = x_group.T @ r #shape: (K,)

    #Step 2: Compute t̂_j
    t_hat = np.clip(a/lambda2,-1,1)

    #Step 3: Compute J(t̂)
    residual = a - lambda2 * t_hat

    j = np.sum((residual/lambda1)**2)

    #Step 4: Return decision
    return j <= 1




