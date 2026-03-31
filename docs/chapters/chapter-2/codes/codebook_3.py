# Source: Optimization/chapter-2/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Ground Truth and Sample Data
# ====================================================================

N_SAMPLES = 100
# Ground Truth: A bimodal distribution (two Gaussians)
X_A = np.random.normal(loc=-2, scale=0.5, size=N_SAMPLES // 2)
X_B = np.random.normal(loc=2, scale=1.0, size=N_SAMPLES // 2)
X_full = np.concatenate([X_A, X_B])

# Function for the true underlying density (for plotting reference)
def true_density(x):
    return 0.5 * norm.pdf(x, loc=-2, scale=0.5) + 0.5 * norm.pdf(x, loc=2, scale=1.0)

# Grid for plotting the smooth functions
x_plot = np.linspace(-5, 5, 500)
y_true = true_density(x_plot)

# ====================================================================
# 2. KDE Trials (Varying Bandwidth)
# ====================================================================

# Trial A: Small Bandwidth (High Variance, Low Bias)
H_SMALL = 0.1
kde_small = gaussian_kde(X_full, bw_method=H_SMALL)
y_small = kde_small(x_plot)

# Trial B: Large Bandwidth (High Bias, Low Variance)
H_LARGE = 1.0
kde_large = gaussian_kde(X_full, bw_method=H_LARGE)
y_large = kde_large(x_plot)

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

plt.figure(figsize=(10, 6))
# Plot raw data (rug plot at the bottom)
plt.plot(X_full, np.full_like(X_full, -0.01), '|k', markeredgewidth=1, alpha=0.5, label='Raw Data Samples')

# Plot true density
plt.plot(x_plot, y_true, 'k--', label='True Density (Reference)', lw=2)

# Plot KDE estimates
plt.plot(x_plot, y_small, 'r-', label=f'KDE (h={H_SMALL}): High Variance', lw=1.5)
plt.plot(x_plot, y_large, 'b-', label=f'KDE (h={H_LARGE}): High Bias', lw=1.5)

# Labeling and Formatting
plt.title('Kernel Density Estimation: Bandwidth and Bias-Variance Trade-off')
plt.xlabel('x')
plt.ylabel('Probability Density $\\hat{p}(x)$')
plt.ylim(-0.05, 0.45)
plt.legend()
plt.grid(True)
plt.show()

print("\n--- KDE Bandwidth Analysis ---")
print(f"Reference Structure: Bimodal (peaks at x=-2 and x=2)")
print("-------------------------------------------------")
print(f"KDE with h={H_SMALL} (Small): Estimate is spiky (high variance) but accurately resolves the bimodal structure (low bias).")
print(f"KDE with h={H_LARGE} (Large): Estimate is smooth (low variance) but fails to resolve the two peaks, becoming an inaccurate single-mode blob (high bias).")

print("\nConclusion: The bandwidth h controls the bias-variance trade-off. A proper choice is critical for accurately inferring the underlying multi-modal energy landscape.")
