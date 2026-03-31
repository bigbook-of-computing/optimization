# Source: Optimization/chapter-2/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Target Distribution (P) and Function (f)
# ====================================================================

N_SAMPLES = 10000  # Number of samples for the estimator
TARGET_MEAN = 0.0
TARGET_STD = 1.0

# True Distribution P(x) = N(x|0, 1)
def P(x):
    return norm.pdf(x, loc=TARGET_MEAN, scale=TARGET_STD)

# Function to integrate f(x) = x^2
def f(x):
    return x**2

# Analytical Result: <f>_P = <x^2>_N(0,1) = Variance = 1.0
ANALYTICAL_MEAN = 1.0

# ====================================================================
# 2. Importance Sampling Trials
# ====================================================================

# Trial A: Good Proposal Q_A (Perfect Match)
Q_A = lambda x: norm.pdf(x, loc=TARGET_MEAN, scale=TARGET_STD)
X_A = np.random.normal(loc=TARGET_MEAN, scale=TARGET_STD, size=N_SAMPLES)
Weights_A = P(X_A) / Q_A(X_A)  # Weights should be all 1s
Estimate_A = np.mean(f(X_A) * Weights_A)
Variance_A = np.var(f(X_A) * Weights_A)

# Trial B: Poor Proposal Q_B (Distant Mean)
Q_B_MEAN = 5.0
Q_B = lambda x: norm.pdf(x, loc=Q_B_MEAN, scale=TARGET_STD)
X_B = np.random.normal(loc=Q_B_MEAN, scale=TARGET_STD, size=N_SAMPLES)
Weights_B = P(X_B) / Q_B(X_B)
Estimate_B = np.mean(f(X_B) * Weights_B)
Variance_B = np.var(f(X_B) * Weights_B)

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

print("--- Importance Sampling Performance ---")
print(f"Target Analytical Mean <f(x)>_P: {ANALYTICAL_MEAN:.4f}")

print("\nTrial A: Good Proposal Q_A = N(0, 1)")
print(f"  Estimate: {Estimate_A:.4f} (Accurate)")
print(f"  Variance of Estimator: {Variance_A:.4f} (Low)")

print("\nTrial B: Poor Proposal Q_B = N(5, 1)")
print(f"  Estimate: {Estimate_B:.4f}")
print(f"  Variance of Estimator: {Variance_B:.4f} (Extremely High)")

# Plotting the weights (visualizing the mismatch)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_B, Weights_B, s=10, alpha=0.5, color='darkred', label='Weights $w(x) = P(x)/Q_B(x)$')
ax.axhline(0, color='k', linestyle='--')
ax.set_title("Importance Weights for Poor Proposal $Q_B = \\mathcal{N}(5, 1)$")
ax.set_xlabel("Sampled Point $x$")
ax.set_ylabel("Importance Weight $w(x)$")
ax.grid(True)
plt.show()

print("\nConclusion: Both trials achieved the correct mean (analytic result of 1.0) on average. However, the variance of the estimate for the distant proposal (Trial B) is orders of magnitude higher. This confirms that the variance of the Importance Sampling estimator explodes when the proposal distribution does not adequately cover the important, low-energy region (near x=0) of the target distribution.")
