# Source: Optimization/chapter-2/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde

# 1. Define the "ground truth" target distribution
# A 2D Gaussian with strong correlation (non-diagonal covariance)
mean = np.zeros(2)
cov = np.array([[1.0, 0.8], [0.8, 1.5]])
# This covariance matrix defines the "true" energy landscape.

# 2. Draw N samples (our "simulation data")
samples = np.random.multivariate_normal(mean, cov, 5000)

# 3. Perform Kernel Density Estimation (KDE)
# This is the practical implementation of the formula from Sec 2.6
# 'gaussian_kde' automatically determines a good bandwidth (h).
kde = gaussian_kde(samples.T)

# 4. Create a 2D grid to evaluate the learned density
x, y = np.mgrid[-3:3:100j, -3:3:100j]
grid_points = np.vstack([x.ravel(), y.ravel()])

# Evaluate the KDE on the grid to get the smooth density
z = kde(grid_points).reshape(100, 100)

# 5. Visualization
plt.figure(figsize=(9, 7))
# Plot the learned, smooth density as a filled contour map
plt.contourf(x, y, z, levels=30, cmap='viridis')
plt.colorbar(label='Estimated Density $\hat{p}(\mathbf{x})$')

# Overlay a few raw samples to show the "empirical" data
plt.scatter(samples[:200, 0], samples[:200, 1], s=5, color='white', alpha=0.5, label='Raw Samples')

plt.title('Monte Carlo + KDE Density Estimation')
plt.xlabel('Feature $x_1$')
plt.ylabel('Feature $x_2$')
plt.legend()
plt.show()
