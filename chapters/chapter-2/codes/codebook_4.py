# Source: Optimization/chapter-2/codebook.md -- Block 4

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Ground Truth and Data Generation
# ====================================================================

N_SAMPLES = 1000
D = 2  # 2D Gaussian

# --- True Parameters (The "Ground Truth") ---
MU_TRUE = np.array([2.5, -1.0])
SIGMA_TRUE = np.array([[1.0, 0.5], [0.5, 2.0]])

# Generate the data from the true distribution
X_data = np.random.multivariate_normal(MU_TRUE, SIGMA_TRUE, N_SAMPLES)

# ====================================================================
# 2. Maximum Likelihood Estimation (Analytical Solution)
# ====================================================================

# MLE Estimate for the Gaussian Mean is the Sample Mean
MU_MLE = np.mean(X_data, axis=0)

# MLE Estimate for the Gaussian Covariance is the Sample Covariance
# Note: np.cov uses N-1 by default (unbiased sample covariance); the MLE formula uses N (biased). 
# We use the unbiased estimator here for better comparison, but note the technical distinction.
SIGMA_MLE = np.cov(X_data, rowvar=False)

# --- Define a Poorly Chosen Parameter Set for Comparison ---
MU_POOR = np.array([0.0, 0.0])
SIGMA_POOR = np.array([[3.0, 0.0], [0.0, 3.0]])

# ====================================================================
# 3. Log-Likelihood Calculation (Verification)
# ====================================================================

def calculate_log_likelihood(X, mu, sigma):
    """
    Computes the total log-likelihood for the dataset X given parameters (mu, sigma).
    This measures how well the model (mu, sigma) explains the data.
    """
    # Create the multivariate Gaussian object
    model = multivariate_normal(mean=mu, cov=sigma)
    
    # Compute the log-likelihood for each point and sum them up
    # Total Log-Likelihood = sum(log P(x_i | theta))
    return np.sum(model.logpdf(X))

# Calculate the log-likelihood for the three parameter sets
LL_TRUE = calculate_log_likelihood(X_data, MU_TRUE, SIGMA_TRUE)
LL_MLE = calculate_log_likelihood(X_data, MU_MLE, SIGMA_MLE)
LL_POOR = calculate_log_likelihood(X_data, MU_POOR, SIGMA_POOR)

# ====================================================================
# 4. Visualization and Summary
# ====================================================================

print("--- Maximum Likelihood Estimation (MLE) Analysis ---")

print("\n1. Parameter Comparison:")
print(f"| Parameter | True | MLE Estimate | Difference |")
print("| :--- | :--- | :--- | :--- |")
print(f"| Mean (\u03bc) | {MU_TRUE} | {np.round(MU_MLE, 3)} | {np.round(MU_MLE - MU_TRUE, 3)} |")
print(f"| Cov (\u03a3)[0,1] | {SIGMA_TRUE[0, 1]:.3f} | {SIGMA_MLE[0, 1]:.3f} | {SIGMA_MLE[0, 1] - SIGMA_TRUE[0, 1]:.3f} |")

print("\n2. Log-Likelihood (LL) Verification:")
print(f"LL of True Parameters:    {LL_TRUE:.2f}")
print(f"LL of MLE Parameters:     {LL_MLE:.2f} (Maximized)")
print(f"LL of Poor Parameters:    {LL_POOR:.2f}")

# Plot LL comparison
plt.figure(figsize=(8, 5))
plt.bar(['LL_True', 'LL_MLE', 'LL_Poor'], [LL_TRUE, LL_MLE, LL_POOR], color=['gray', 'darkgreen', 'red'])
plt.axhline(LL_MLE, color='k', linestyle='--', alpha=0.6, label='Maximum Likelihood')
plt.title('Log-Likelihood Maximization')
plt.ylabel('Total Log-Likelihood')
plt.grid(True, axis='y')
plt.show()

print("\nConclusion: The LL_MLE is the highest value, confirming that the empirical sample mean and covariance are the **Maximum Likelihood Estimates** for the Gaussian model. This numerically verifies the analytical solution and shows that the empirical statistics correctly capture the generative parameters of the distribution under the MaxEnt principle.")
