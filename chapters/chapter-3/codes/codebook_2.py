# Source: Optimization/chapter-3/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Generate Synthetic Curved Manifold (3D S-Curve)
# ====================================================================

N_SAMPLES = 2000
# True low dimension is 1 (the length of the curve)
t = np.linspace(0, 4 * np.pi, N_SAMPLES)
noise = np.random.normal(0, 0.1, N_SAMPLES)

# Create the 3D S-shaped data (Curved manifold in 3D space)
X_3D = np.zeros((N_SAMPLES, 3))
X_3D[:, 0] = t + noise       # Feature 1 (mostly linear)
X_3D[:, 1] = np.sin(t) + noise  # Feature 2 (introduces curvature)
X_3D[:, 2] = np.cos(t) * 0.5 + noise # Feature 3 (orthogonal curvature)

# Color the data based on the true underlying 1D progression (t)
colors = t

# ====================================================================
# 2. Linear Projection (PCA)
# ====================================================================
X_scaled = StandardScaler().fit_transform(X_3D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ====================================================================
# 3. Nonlinear Projection (t-SNE)
# ====================================================================
# Note: UMAP is often faster/better, but t-SNE is standard for illustration.
# The random_state is essential for reproducibility in t-SNE.
tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# ====================================================================
# 4. Visualization and Comparison
# ====================================================================

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: PCA (Linear Projection)
sc1 = ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis', s=10)
ax[0].set_title('PCA (Linear) Projection: Distortion')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
ax[0].grid(True)

# Plot 2: t-SNE (Nonlinear Projection)
sc2 = ax[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', s=10)
ax[1].set_title('t-SNE (Nonlinear) Projection: Manifold Unrolling')
ax[1].set_xlabel('t-SNE Component 1')
ax[1].set_ylabel('t-SNE Component 2')
ax[1].grid(True)

fig.colorbar(sc2, ax=ax, orientation='vertical', label='True 1D Progression (t)')
plt.tight_layout()
plt.show()

# --- Analysis Summary ---
print("\n--- Manifold Projection Comparison ---")
print("PCA Result (Linear): The curve is folded and distorted onto a flat plane. States from opposite ends of the S-curve (different 't' values) are projected close together, misrepresenting the physical distance.")
print("t-SNE Result (Nonlinear): The algorithm successfully 'unrolls' the 3D curve into a continuous 2D path, preserving the true sequential structure (color gradient).")
print("\nConclusion: This demonstrates the failure of the linear assumption for curved physical manifolds. Nonlinear methods are required to preserve the system's topological relationships accurately.")
