import numpy as np
from scipy.optimize import minimize_scalar

def update_feature(x_group,theta,r,j,lambda1,lambda2):

 """
Updates a single feature theta_j inside a group using coordinate descent

Parameters:
        X_group : np.ndarray
            (n_samples, n_features_in_group) design matrix for one group
        theta : np.ndarray
            Current coefficient vector for the group (shape: k,)
        r : np.ndarray
            Residual vector for the group (shape: n,)
        j : int
            Index of the feature to update
        lambda1 : float
            Group-level L2 penalty
        lambda2 : float
            L1 (sparsity) penalty

    Returns:
        updated_theta_j : float
            The new value of theta_j
 """
 Z_j = x_group[:,j]

 #Step1: Compute partial residual (exclude current feature)
 r_j = r - x_group @ theta + Z_j * theta[j]

 #Step2: Soft-thresholding condition
 dot = Z_j.T @ r_j

 if np.abs(dot) < lambda2:
    return 0.0   #set theta_j = 0 if not strong enough

 #Step 3: Full objective function (1D) for scalar optimization
 def loss_fn(t_j):
  
   theta_copy = theta.copy()

   theta_copy[j] = t_j
   
   group_norm = np.linalg.norm(theta_copy)
   
   data_fit = 0.5 * np.sum((r_j - Z_j*t_j)**2)
   
   group_penalty = lambda1 * group_norm
   
   l1_penalty = lambda2 * np.abs(t_j)
   
   return data_fit + group_penalty + l1_penalty
 
 result = minimize_scalar(fun=loss_fn,method='brent')
 
 return result.x


