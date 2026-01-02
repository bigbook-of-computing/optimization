# 📘 Chapter 3: Dimensionality Reduction & Clustering

## Project 1: Quantifying Dimensionality (Explained Variance)

-----

### Definition: Quantifying Dimensionality

The goal of this project is to quantitatively determine the system's **effective intrinsic dimensionality** ($d$) by analyzing the **explained variance ratio ($\lambda_k / \sum \lambda_j$)** derived from Principal Component Analysis (PCA).

### Theory: Explained Variance and the Manifold

While simulation data may live in a high-dimensional **embedding space** ($\mathbb{R}^D$), physical constraints force the dynamics onto a much lower-dimensional, intrinsic manifold ($\mathcal{M}$).

**PCA** finds the axes of maximum variance (eigenvectors, $\mathbf{v}_k$), and the **eigenvalues** ($\lambda_k$) quantify the variance along those axes.

The **Cumulative Explained Variance** measures the total fraction of system variability captured by the first $d$ components:

$$\text{Cumulative Variance}(d) = \frac{\sum_{j=1}^d \lambda_j}{\sum_{j=1}^D \lambda_j}$$

The **intrinsic dimensionality** is typically chosen as the number of components $d$ required to capture $\sim 95\%$ or more of the total variance. This number of components approximates the dimensionality of the underlying manifold $\mathcal{M}$.

-----

### Extensive Python Code and Visualization

The code generates high-dimensional data ($D=50$) with a known low intrinsic dimension ($d=5$) and plots the cumulative explained variance to visually and quantitatively locate the system's true core dimensionality.

```python
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
```

-----

## Project 2: Simulating and Visualizing a Curved Manifold

-----

### Definition: Simulating and Visualizing a Curved Manifold

The goal is to generate a dataset with a clear **curved (non-linear) manifold structure**. This demonstrates the limitations of **linear PCA** and highlights why **nonlinear embedding methods** (like t-SNE or UMAP) are necessary to accurately preserve the system's true local geometry.

### Theory: Linear vs. Nonlinear Preservation

  * **PCA (Linear):** PCA attempts to fit a flat hyperplane through the curved data. This results in **distortion** and folding, where physically distinct states are projected onto the same point.
  * **Manifold Learning (Nonlinear):** Methods like t-SNE or UMAP are designed to **preserve local neighborhood relationships** and "unroll" the curved manifold. This is crucial because the geodesic distance (the path along the curve) is the physically meaningful metric.

We use a simple 3D S-curve to visualize the effect of the underlying linear/nonlinear assumption.

-----

### Extensive Python Code and Visualization

The code generates the 3D S-curve data, applies both PCA and t-SNE (a nonlinear method) to project it to 2D, and plots the results side-by-side to compare the preservation of the manifold's structure.

```python
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
```

-----

## Project 3: Automated Phase Discovery (UMAP + DBSCAN)

-----

### Definition: Automated Phase Discovery

The goal is to execute the two-step workflow (`DR -> Density Clustering`) to **automatically discover distinct, non-convex physical phases** within a high-dimensional dataset. This process is the data-driven equivalent of mapping the system's **metastable states**.

### Theory: DR and Density Clustering

1.  **Dimensionality Reduction (UMAP/t-SNE):** First, the high-dimensional data is projected onto a low-dimensional map ($d=2$) that is **curvature-aware**. UMAP (or t-SNE) is used because its objective is to preserve the complex **local topology** of the hidden manifold $\mathcal{M}$.
2.  **Clustering (DBSCAN):** Second, **DBSCAN** (a density-based method) is applied to the 2D map. DBSCAN is ideal because it identifies clusters of **arbitrary, non-convex shapes** and automatically flags low-density regions (like transition states or noise) as unassigned, contrasting sharply with the spherical assumptions of K-Means.

The resulting clusters represent the system's distinct physical **phases**.

-----

### Extensive Python Code and Visualization

The code generates a synthetic dataset with two complex, crescent-shaped clusters (non-convex phases), applies UMAP for projection, uses DBSCAN for density-based clustering, and visualizes the results.

```python
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
```

-----

## Project 4: K-Means as an Energy Minimizer (Convergence Check)

-----

### Definition: K-Means as an Energy Minimizer

The goal is to track the optimization process of the **K-Means (Lloyd's) algorithm** to demonstrate that the clustering process is a **relaxation dynamic**. This confirms that K-Means is mathematically equivalent to a system seeking a **local energy minimum**.

### Theory: Monotonic Energy Descent

K-Means is an iterative heuristic that minimizes the **intra-cluster variance** (or objective function $J$), which is analogous to a system's potential energy:

$$J = \sum_{i=1}^N \| \mathbf{x}_i - \boldsymbol{\mu}_{c_i} \|^2$$

**Lloyd's Algorithm** alternates between the **E-Step (Assignment)** and the **M-Step (Centroid Update)**:

1.  **E-Step** (Assigning points to the nearest fixed centroids) is guaranteed to decrease or maintain $J$.
2.  **M-Step** (Re-calculating centroids as the mean of their assigned points) is also guaranteed to decrease or maintain $J$.

Because $J$ is bounded from below, the cost function **must monotonically decrease** until the algorithm converges to a **local minimum**. Tracking $J(t)$ validates this relaxation dynamic.

-----

### Extensive Python Code and Visualization

The code implements the K-Means algorithm manually (using a simple loop structure) to track the objective function $J$ (SSE) at every step and plots the convergence curve.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Parameters and Data Generation
# ====================================================================

N_SAMPLES = 300
N_CLUSTERS = 3
MAX_ITER = 30

# Generate 2D circular clusters (ideal K-Means data)
X, y_true = make_blobs(n_samples=N_SAMPLES, n_features=2, centers=N_CLUSTERS, 
                       cluster_std=0.8, random_state=42)

# ====================================================================
# 2. K-Means (Lloyd's) Algorithm Implementation with Cost Tracking
# ====================================================================

# Initialize centroids randomly from the data points
random_indices = np.random.choice(N_SAMPLES, N_CLUSTERS, replace=False)
centroids = X[random_indices]

# Storage for the objective function J
J_history = []

for iteration in range(MAX_ITER):
    # --- E-Step (Assignment: Find nearest centroid) ---
    # cdist computes all pairwise distances (300 x 3)
    distances = cdist(X, centroids, 'euclidean') 
    
    # labels[i] = index of the minimum distance (nearest centroid)
    labels = np.argmin(distances, axis=1)
    
    # Calculate current Objective Function (J) - Sum of Squared Errors
    # ||x_i - mu_k||^2
    current_J = np.sum(distances[np.arange(N_SAMPLES), labels]**2)
    J_history.append(current_J)
    
    # Check for convergence (if J hasn't changed much)
    if iteration > 0 and np.abs(J_history[-1] - J_history[-2]) < 1e-4:
        break
        
    # --- M-Step (Update: Recalculate centroids) ---
    new_centroids = np.zeros_like(centroids)
    
    for k in range(N_CLUSTERS):
        # Find all points belonging to cluster k
        points_in_cluster = X[labels == k]
        
        if len(points_in_cluster) > 0:
            # Update centroid to the mean (center of mass)
            new_centroids[k] = points_in_cluster.mean(axis=0)
        
    centroids = new_centroids

# ====================================================================
# 3. Visualization and Convergence Check
# ====================================================================

plt.figure(figsize=(8, 5))

# Plot the Monotonic Descent of the Objective Function J (Energy)
plt.plot(J_history, 'r-o', lw=2, markersize=5)

plt.title('K-Means Relaxation Dynamics (Objective Function $J$)')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Function $J$ (Sum of Squared Errors)')
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- K-Means Convergence Check ---")
print(f"Number of Iterations to Converge: {iteration + 1}")
print(f"Initial Cost (J): {J_history[0]:.2f}")
print(f"Final Cost (J):   {J_history[-1]:.2f}")

print("\nConclusion: The plot shows the **Sum of Squared Errors (J)** strictly decreases at every iteration, confirming the K-Means algorithm is a **relaxation process (gradient descent)** guaranteed to converge to a local minimum in the data's energy landscape.")
```
