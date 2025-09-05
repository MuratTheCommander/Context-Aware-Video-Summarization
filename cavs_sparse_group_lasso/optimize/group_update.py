import numpy as np
from optimize.group_zero_test import is_group_zero
from optimize.feature_update import update_feature

def update_group(x_group,beta_group,r,lambda1,lambda2):

    """
    updates one group of features using sparse group lasso rules.

    Parameters:
        X_group : np.ndarray
            (n_samples, n_features_in_group) design matrix for this group
        beta_group : np.ndarray
            Current coefficient vector for this group (shape: k,)
        r : np.ndarray
            Residual vector excluding this group (shape: n,)
        lambda1 : float
            Group-level regularization strength
        lambda2 : float
            L1 feature-level sparsity penalty

    Returns:
        updated_beta : np.ndarray
            Updated coefficient vector for this group
    """

    #Step1: check if group can be skipped
    if is_group_zero(x_group,r,lambda1,lambda2):
       return np.zeros_like(beta_group)
    
    #Step2: Update each feature in the group
    updated_beta = beta_group.copy()

    for j in range(x_group.shape[1]): # loop over features in the group
        updated_beta[j] = update_feature(
            x_group,
            updated_beta,
            r,
            j,
            lambda1,
            lambda2
        )
    
    return updated_beta