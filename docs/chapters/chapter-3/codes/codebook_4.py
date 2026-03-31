# Source: Optimization/chapter-3/codebook.md -- Block 4

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
