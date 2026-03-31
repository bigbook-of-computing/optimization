# Source: Optimization/chapter-9/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ====================================================================
# 1. Setup Data and Parameter Scenarios
# ====================================================================

# We set up a scenario where the MLE (Sample Mean) is noisy (high std error)
# and the MAP estimate should shrink this noise using a strong prior.
N_SAMPLES = 5 # Small number of samples for high noise/variance
TRUE_MU = 10.0
DATA_STD = 2.0
PRIOR_MEAN = 5.0 # Prior is far from the true value (5.0 vs 10.0)

# Simulate Noisy Data
np.random.seed(1) 
data = np.random.normal(loc=TRUE_MU, scale=DATA_STD, size=N_SAMPLES)
MLE_estimate = np.mean(data)

# --- Scenario A: Weak Prior (Wide, low penalty) ---
SIGMA_A = 5.0 
# Scenario B: Strong Prior (Narrow, high penalty)
SIGMA_B = 0.5

# ====================================================================
# 2. MAP Calculation (Analytic Formula)
# ====================================================================

def calculate_map_mean(mu_prior, sigma_prior, sample_mean, data_std, N):
    sigma_likelihood = data_std**2 / N 
    return (mu_prior/sigma_prior**2 + sample_mean/sigma_likelihood) / \
           (1/sigma_prior**2 + 1/sigma_likelihood)

# Calculate MAP Estimates
MAP_A = calculate_map_mean(PRIOR_MEAN, SIGMA_A, MLE_estimate, DATA_STD, N_SAMPLES)
MAP_B = calculate_map_mean(PRIOR_MEAN, SIGMA_B, MLE_estimate, DATA_STD, N_SAMPLES)

# ====================================================================
# 3. Visualization and Comparison
# ====================================================================

estimates = {
    'True Mean': TRUE_MU,
    'MLE (Sample Mean)': MLE_estimate,
    'MAP (Weak Prior \u03c3=5.0)': MAP_A,
    'MAP (Strong Prior \u03c3=0.5)': MAP_B
}
names = list(estimates.keys())
values = list(estimates.values())

# Calculate the Pull towards the Prior (distance from MLE)
pull_A = abs(MLE_estimate - MAP_A)
pull_B = abs(MLE_estimate - MAP_B)


plt.figure(figsize=(9, 6))

# Plot bars
bars = plt.bar(names, values, color=['gray', 'k', 'skyblue', 'darkred'])

# Highlight Prior Mean
plt.axhline(PRIOR_MEAN, color='b', linestyle='--', label=f'Prior Mean={PRIOR_MEAN:.2f}')

# Annotate the 'pull'
plt.text(2, MAP_A + 0.3, f'Pull: {pull_A:.2f}', ha='center', color='darkred', weight='bold')
plt.text(3, MAP_B + 0.3, f'Pull: {pull_B:.2f}', ha='center', color='darkred', weight='bold')


# Labeling and Formatting
plt.title('MLE vs. MAP Estimation: Prior as Regularization')
plt.xlabel('Estimate Type')
plt.ylabel(r'Parameter Value $\hat{\theta}$')
plt.legend()
plt.grid(True, axis='y')
plt.show()

# --- Analysis Summary ---
print("\n--- MAP vs. MLE Analysis ---")
print(f"Sample Mean (MLE): {MLE_estimate:.3f}")
print(f"Prior Mean: {PRIOR_MEAN:.3f}")
print("---------------------------------------")
print(f"Weak Prior (\u03c3_0=5.0): MAP = {MAP_A:.3f} (Pulled {pull_A:.2f} units)")
print(f"Strong Prior (\u03c3_0=0.5): MAP = {MAP_B:.3f} (Pulled {pull_B:.2f} units)")

print("\nConclusion: The MAP estimate (MAP_B) calculated with the **strong prior** is pulled significantly farther away from the MLE (Sample Mean) and toward the Prior Mean (5.0). This demonstrates that the strong Prior acts as a robust **regularizer**, penalizing the noisy MLE and biasing the final estimate based on prior belief.")
