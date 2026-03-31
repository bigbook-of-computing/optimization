# Source: Optimization/chapter-1/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Generate correlated synthetic data (analogy to molecular modes)
N, D = 1000, 5  # 1000 samples (snapshots), 5 features (dimensions)

# Create standard normal (uncorrelated) data
rng = np.random.default_rng(seed=42)
X = rng.standard_normal((N, D))

# Introduce a strong correlation:
# Make feature 1 (index 1) strongly dependent on feature 0 (index 0)
# This simulates a physical "collective variable" or constraint.
X[:, 1] += 0.8 * X[:, 0]

# 2. PCA projection
# We 'standardize' by centering, though scikit-learn's PCA does this.
# For clarity, we'd typically use:
# X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
# But for this simple demo, we proceed directly.

pca = PCA(n_components=2)
# 'fit_transform' computes the mean, finds the eigenvectors (v_k),
# and projects X onto the first two (v_1, v_2).
X_pca = pca.fit_transform(X)

# 3. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=10)
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('2D Projection of 5D Correlated Data')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Optional: Check the variance captured
print(f"Variance captured by PC1: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Variance captured by PC2: {pca.explained_variance_ratio_[1]:.2f}")
