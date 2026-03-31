# Source: Optimization/chapter-3/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate synthetic high-dimensional data
# We create 1000 samples in 5 dimensions (D=5)
# with 3 distinct cluster centers (K=3).
X, y_true = make_blobs(n_samples=1000, WAR: tool-call-rejected
n_features=5, 
centers=3, 
cluster_std=1.2, 
random_state=0)

# X is our (1000, 5) data matrix. y_true holds the "ground truth" labels,
# but we will only use them for validation, not for the clustering.

# 2. Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
# fit_transform finds the 2 principal components and projects
# the 5D data down to a 2D representation.
X_pca = pca.fit_transform(X)

# X_pca is now our (1000, 2) "map" of the data.

# 3. Apply K-Means clustering
# We apply K-Means to the 2D projected data.
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X_pca)
# 'labels' is an array of 0s, 1s, and 2s, assigning each point to a cluster.

# 4. Visualization
plt.figure(figsize=(9, 7))
# Create a scatter plot of the PCA-projected data (PC1 vs PC2)
# Color each point (c=labels) according to its discovered cluster.
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)

plt.title('PCA Projection + K-Means Clustering')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
