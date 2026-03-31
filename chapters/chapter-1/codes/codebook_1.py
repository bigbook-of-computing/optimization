# Source: Optimization/chapter-1/codebook.md -- Block 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Parameters and Synthetic Data Generation
# ====================================================================

M = 1000  # Number of samples (snapshots)
D = 5     # Number of features (dimensions)

# Generate synthetic data with specific biases as required by the setup:
X = np.random.randn(M, D)

# 1. Feature 0: Large Mean (e.g., molecular size in Angstroms)
MU_LARGE = 100.0
X[:, 0] += MU_LARGE 

# 2. Feature 2: Large Variance (e.g., energy fluctuation)
SIGMA_LARGE = np.sqrt(50) # sigma^2 approx 50
X[:, 2] *= SIGMA_LARGE

# Feature labels for clarity
feature_names = [
    'Position_X (\u03bc=100)', 
    'Velocity_Y', 
    'Energy (\u03c3^2=50)', 
    'Spin_Z', 
    'Temp_T'
]

# ====================================================================
# 2. Standardization (Z-score normalization)
# ====================================================================

# Step 1: Compute moments of the raw data (X)
mu_raw = X.mean(axis=0)
sigma_raw = X.std(axis=0, ddof=1) # Use ddof=1 for sample stdev

# Step 2: Apply the standardization formula manually
X_standardized = (X - mu_raw) / sigma_raw

# Step 3: Compute moments of the standardized data (X')
mu_std = X_standardized.mean(axis=0)
sigma_std = X_standardized.std(axis=0, ddof=1)

# ====================================================================
# 3. Visualization and Verification
# ====================================================================

# Create a DataFrame for clean output
data = {
    'Raw Mean (\u03bc)': mu_raw,
    'Raw StDev (\u03c3)': sigma_raw,
    'Std Mean (\u03bc\')': mu_std,
    'Std StDev (\u03c3\')': sigma_std
}
df_stats = pd.DataFrame(data, index=feature_names)

print("--- Data Standardization Verification ---")
print(df_stats.to_string())

# Plot the standard deviations before and after
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(D)
width = 0.35

rects1 = ax.bar(x - width/2, sigma_raw, width, label='Raw StDev', color='skyblue')
rects2 = ax.bar(x + width/2, sigma_std, width, label='Standardized StDev', color='darkgreen')

ax.set_ylabel('Standard Deviation')
ax.set_title('Feature Scaling: Raw vs. Standardized StDev')
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45, ha="right")
ax.axhline(1.0, color='r', linestyle='--', label='Target $\u03c3 = 1$')
ax.legend()
plt.tight_layout()
fig.savefig('Optimization/RESEARCH/docs/chapters/chapter-1/codes/ch1_standardization.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print("\nConclusion: Standardization successfully transformed the data. The features with wildly different raw means (e.g., 100.0) and standard deviations (e.g., 7.0) now all have a mean of approximately 0.0 and a standard deviation of 1.0, ensuring that all features contribute equally to the final geometric analysis.")
