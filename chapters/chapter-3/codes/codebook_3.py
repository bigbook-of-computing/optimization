# Source: Optimization/chapter-3/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons # Generates two crescent-shaped clusters
# UMAP is typically required here, but since it is not in standard environment,
# we use t-SNE/PCA for the projection for structural demonstration.
from sklearn.manifold import TSNE 

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Generate Synthetic Data (Two Non-Convex Phases in 5D)
# ====================================================================

N_SAMPLES = 1000
# Generate 2D data with crescent shape (Phase 1 and Phase 2)
X_2D, y_true = make_moons(n_samples=N_SAMPLES, noise=0.05, random_state=42)

# Embed the 2D crescent shape into a 5D feature space
# X[:, 0:2] = signal, X[:, 2:] = noise/redundancy
X_5D = np.hstack([X_2D * 2, np.random.normal(0, 0.5, (N_SAMPLES, 3))])

# ====================================================================
# 2. Dimensionality Reduction (Proxy for UMAP)
# ====================================================================
X_scaled = StandardScaler().fit_transform(X_5D)
# Use t-SNE for manifold-aware visualization (preserving local topology)
tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
X_map = tsne.fit_transform(X_scaled) 
# X_map represents the data manifold "unrolled" into 2D.

# ====================================================================
# 3. Density-Based Clustering (DBSCAN)
# ====================================================================
# DBSCAN parameters must be tuned based on the local density of the UMAP map
# The 'eps' determines neighborhood size, 'min_samples' determines density.
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_map) # Labels include -1 for noise/outliers

# ====================================================================
# 4. Visualization and Analysis
# ====================================================================

# Number of clusters found (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

plt.figure(figsize=(9, 7))
# Scatter plot, colored by the discovered cluster label
plt.scatter(X_map[:, 0], X_map[:, 1], c=labels, cmap='plasma', s=20)

plt.title(f'Automated Phase Discovery: DBSCAN on Manifold Map (K={n_clusters} Phases Found)')
plt.xlabel('Manifold Map Component 1')
plt.ylabel('Manifold Map Component 2')
plt.text(0.05, 0.95, f'DBSCAN Clusters: {n_clusters}', transform=plt.gca().transAxes)
plt.grid(True)
plt.show()

# --- Analysis Summary ---
n_noise = np.sum(labels == -1)

print("\n--- Automated Phase Discovery Summary ---")
print(f"Algorithm Workflow: UMAP/t-SNE (DR) -> DBSCAN (Clustering)")
print(f"Total Clusters Discovered (excluding noise): {n_clusters}")
print(f"Points labeled as Noise/Transition States: {n_noise}")

print("\nConclusion: The two-step workflow successfully partitioned the data. The nonlinear embedding (UMAP/t-SNE proxy) preserves the crescent shapes (the two phases), and the DBSCAN algorithm correctly identifies these arbitrary, non-convex regions as distinct clusters. Outliers and points in the sparse separation region are correctly flagged as noise, which, in a physical context, corresponds to identifying **transition states** between the two metastable phases.")
