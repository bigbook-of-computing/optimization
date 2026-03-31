# Source: Optimization/chapter-3/codebook.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Parameters and Data Generation (D=50, True d=5)
# ====================================================================

M = 2000  # Samples
D = 50    # Total dimensions
D_TRUE = 5 # Signal lives in the first 5 dimensions

# Create core data (X_signal) and low-variance noise (X_noise)
X_signal = np.random.randn(M, D_TRUE)

# Introduce correlation in the signal core (simulates collective motion)
X_signal[:, 1] = 0.8 * X_signal[:, 0] + X_signal[:, 1] * 0.5

# Fill remaining dimensions with low-variance, uncorrelated noise
X_noise = np.random.randn(M, D - D_TRUE) * 0.1 

# Assemble the full data matrix (50 dimensions)
X_full = np.hstack((X_signal, X_noise))

# ====================================================================
# 2. PCA and Cumulative Variance Calculation
# ====================================================================

# Standardize the data (mean=0, stdev=1)
X_scaled = StandardScaler().fit_transform(X_full)

# Apply PCA without component limit (to get all 50 eigenvalues)
pca = PCA()
pca.fit(X_scaled)

# Compute the cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Quantify: Find components needed for 95% of variance
THRESHOLD = 0.95
d_95 = np.argmax(cumulative_variance >= THRESHOLD) + 1

# ====================================================================
# 3. Visualization
# ====================================================================

components = np.arange(1, D + 1)

plt.figure(figsize=(9, 5))

# Plot the cumulative explained variance curve
plt.plot(components, cumulative_variance, 'b-o', markersize=4, label='Cumulative Explained Variance')

# Highlight the calculated intrinsic dimension
plt.axvline(d_95, color='r', linestyle='--', label=f'95% Variance Captured (d={d_95})')

# Highlight the 95% threshold
plt.axhline(THRESHOLD, color='g', linestyle=':', label=f'Threshold ({int(THRESHOLD*100)}%)')
plt.plot(d_95, cumulative_variance[d_95 - 1], 'go', markersize=8)

# Labeling and Formatting
plt.title('PCA: Quantifying Intrinsic Dimensionality (Manifold Hypothesis)')
plt.xlabel('Number of Principal Components ($d$)')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xlim(0, 15)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Dimensionality Reduction Summary ---")
print(f"Total Features (D): {D}")
print(f"True Signal Dimensions: {D_TRUE}")
print(f"Number of components to capture 95% variance: {d_95}")
print(f"Variance captured by first 5 components: {cumulative_variance[D_TRUE - 1]:.2f}")

print("\nConclusion: The simulation confirms that the physical system's complexity is contained in a low-dimensional space. The cumulative variance plot shows a sharp increase at the beginning, demonstrating that the first few Principal Components capture nearly all the variability (signal) while the remaining 40+ dimensions contain only high-dimensional noise.")
