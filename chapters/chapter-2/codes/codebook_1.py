# Source: Optimization/chapter-2/codebook.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# ====================================================================
# 1. Setup Datasets (Low D vs. High D)
# ====================================================================

N = 500  # Number of samples
# Run 1: Low Dimension (D=2)
D_LOW = 2
X_low = np.random.randn(N, D_LOW)

# Run 2: High Dimension (D=1000)
D_HIGH = 1000
X_high = np.random.randn(N, D_HIGH)

# ====================================================================
# 2. Distance Calculation and Concentration Metric
# ====================================================================

def calculate_concentration(X):
    """
    Calculates the relative spread of pairwise Euclidean distances.
    The ratio approaches 0 for high dimensions.
    """
    # Calculate all pairwise Euclidean distances
    distances = pdist(X, metric='euclidean')
    
    # Compute the required metrics
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    avg_dist = np.mean(distances)
    
    # Relative Spread Ratio
    spread_ratio = (max_dist - min_dist) / avg_dist
    return spread_ratio, avg_dist

spread_low, avg_low = calculate_concentration(X_low)
spread_high, avg_high = calculate_concentration(X_high)

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

print("--- Distance Concentration Test (Curse of Dimensionality) ---")
print(f"1. Low Dimension (D={D_LOW}):")
print(f"   Average Distance: {avg_low:.3f}")
print(f"   Relative Spread Ratio: {spread_low:.3f} (Large variance in distances)")

print(f"\n2. High Dimension (D={D_HIGH}):")
print(f"   Average Distance: {avg_high:.3f}")
print(f"   Relative Spread Ratio: {spread_high:.3f} (Distances are tightly concentrated)")

print("\nConclusion: The Spread Ratio drops significantly in high dimensions, confirming that most of the space is empty and all data points become equidistant from one another. This illustrates why distance-based algorithms struggle in high-D feature space.")
