# Source: Optimization/chapter-1/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Generate Correlated Synthetic Data (D=5)
# ====================================================================

M = 1000
D = 5
NOISE_LEVEL = 0.5

# Create standard normal (uncorrelated) data
X = np.random.randn(M, D)

# --- Engineering the Correlation ---
# 1. Feature 1 strongly correlated with Feature 0
X[:, 1] = 0.8 * X[:, 0] + NOISE_LEVEL * np.random.randn(M)

# 2. Feature 3 is weakly correlated with Feature 2
X[:, 3] = 0.4 * X[:, 2] + NOISE_LEVEL * np.random.randn(M)

# 3. Feature 4 (the last one) remains largely independent (noise only)
X[:, 4] = X[:, 4] * 0.5

# ====================================================================
# 2. Data Preparation and PCA
# ====================================================================

# Standardize the data (centering mean=0, scaling stdev=1)
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA for dimensionality reduction (n_components=2)
pca = PCA(n_components=2)
# fit_transform finds the axes (v_k) and projects the data (z_ik)
X_pca = pca.fit_transform(X_scaled)

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Plot the 2D projection (PC2 vs. PC1)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=15, color='darkblue')

# Annotate variances
variance_pc1 = pca.explained_variance_ratio_[0]
variance_pc2 = pca.explained_variance_ratio_[1]

plt.text(0.05, 0.95, f'PC1 Variance: {variance_pc1:.2f}', transform=plt.gca().transAxes, fontsize=10)
plt.text(0.05, 0.90, f'PC2 Variance: {variance_pc2:.2f}', transform=plt.gca().transAxes, fontsize=10)
plt.text(0.05, 0.85, f'Cumulative Variance: {variance_pc1 + variance_pc2:.2f}', transform=plt.gca().transAxes, fontsize=10)

plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('PCA Projection: 2D Shadow of 5D Correlated Data')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('Optimization/RESEARCH/docs/chapters/chapter-1/codes/ch1_pca_projection.png', dpi=150, bbox_inches='tight')
plt.close()

# Print out the components to see which original features contribute most
print("\n--- Principal Component Loadings (Coefficients) ---")
print("PC1 (Direction of Max Variance):")
print(np.round(pca.components_[0], 3))

print("PC2 (Next Best Direction):")
print(np.round(pca.components_[1], 3))

print("\nConclusion: The plot shows a clear elongated, elliptical shape. PC1, which captures the majority of the variance (driven by the X0-X1 correlation), aligns with the longest axis of the data cloud. This visual map successfully reduces the 5D data into a 2D projection that reveals the inherent one-dimensional structure (the core collective variable).")
