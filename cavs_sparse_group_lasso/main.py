import numpy as np
from optimize.solver import sparse_group_lasso_solver

#STEP 1 : Generate simple synthetic data 

#Create 3 groups with 2,3 and 2 features respecitvely 
np.random.seed(42)
n_samples = 50

x1 = np.random.randn(n_samples,2)
x2 = np.random.randn(n_samples,3)
x3 = np.random.randn(n_samples,2)

true_beta1 = np.array([1.5, 0.0])
true_beta2 = np.array([0.0, 0.0, 2.0])
true_beta3 = np.array([0.0, 0.0])

X_groups = [x1, x2, x3]
true_betas = [true_beta1, true_beta2, true_beta3]

# Create target y = Xβ + noise
y = sum(X @ b for X, b in zip(X_groups, true_betas)) + 0.1 * np.random.randn(n_samples)

# ----- STEP 2: Run Sparse Group Lasso Solver -----

lambda1 = 0.5  # Group sparsity
lambda2 = 0.3  # Feature sparsity

beta_estimates = sparse_group_lasso_solver(
    X_groups,
    y,
    lambda1=lambda1,
    lambda2=lambda2,
    max_iter=100,
    tol=1e-4
)

# ----- STEP 3: Display Results -----

print("\nEstimated Coefficients:")
for i, beta in enumerate(beta_estimates):
    print(f"Group {i+1}: {beta.round(4)}")

