# Source: Optimization/chapter-1/codebook.md -- Block 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Parameters and Correlated Data Generation
# ====================================================================

M = 1000  # Number of samples
D = 3     # Number of features (X1, X2, X3)
NOISE_LEVEL = 0.5

# Create core data (Features X1, X2, X3)
X = np.random.randn(M, D)

# --- Engineering the Correlation (Physical Dependency) ---
# X_col2 = 0.8 * X_col1 + noise (Strongly correlated)
X[:, 1] = 0.8 * X[:, 0] + NOISE_LEVEL * np.random.randn(M)

# X_col3 remains uncorrelated with X_col1 and X_col2 (Independent)
X[:, 2] = 2.0 * X[:, 2] # Scale X3 up to show its variance explicitly

feature_names = ['X1 (Input)', 'X2 (Correlated)', 'X3 (Independent)']

# ====================================================================
# 2. Covariance Matrix Calculation
# (Standardization is implicitly applied here by using the correlation logic)
# ====================================================================

# Calculate the Covariance Matrix (uses unbiased estimator M-1)
# Note: For accurate PCA, data should be centered first, but np.cov does this automatically.
Cov_Matrix = np.cov(X, rowvar=False) 

# ====================================================================
# 3. Analysis and Visualization
# ====================================================================

df_cov = pd.DataFrame(Cov_Matrix, index=feature_names, columns=feature_names)

print("--- Computed Covariance Matrix (\u03a3) ---")
print(df_cov.to_string())

# --- Interpretation ---
print("\n--- Geometric and Physical Interpretation ---")
print("Interpretation of Off-Diagonal Elements:")
print(f"1. \u03a3[X1, X2] = {Cov_Matrix[0, 1]:.3f}: Large positive value. Confirms the strong, engineered correlation: as X1 increases, X2 tends to increase.")
print(f"2. \u03a3[X1, X3] = {Cov_Matrix[0, 2]:.3f}: Value near zero. Confirms that the input feature X1 is independent (uncorrelated) with feature X3.")
print(f"3. \u03a3[X2, X3] = {Cov_Matrix[1, 2]:.3f}: Value near zero. Confirms that the correlated feature X2 is also largely independent of X3.")

print("\nInterpretation of Diagonal Elements (\u03a3[i,i] = Variance):")
print(f"Variance X1: {Cov_Matrix[0, 0]:.3f} | Variance X2: {Cov_Matrix[1, 1]:.3f} | Variance X3: {Cov_Matrix[2, 2]:.3f}")
print("The variances (diagonal elements) show the features' individual spread. X3 has the largest variance due to its scale factor (2.0).")

# Visualization: Heatmap of the matrix
plt.figure(figsize=(6, 5))
plt.imshow(Cov_Matrix, cmap='coolwarm', origin='upper', interpolation='none')
plt.colorbar(label='Covariance Value')
plt.title('Covariance Matrix Heatmap (\u03a3)')
plt.xticks(np.arange(D), feature_names, rotation=45, ha="right")
plt.yticks(np.arange(D), feature_names)
plt.tight_layout()
plt.savefig('Optimization/RESEARCH/docs/chapters/chapter-1/codes/ch1_covariance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
