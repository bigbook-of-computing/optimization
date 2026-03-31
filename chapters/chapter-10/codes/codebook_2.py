# Source: Optimization/chapter-10/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup High-Dimensional Data (Prone to Overfitting)
# ====================================================================

N = 50   # Samples
D = 10   # High dimensionality (features)
TRUE_W_SMALL = np.array([3.0] + [0.0]*(D-1)) # Only one feature is truly important
NOISE_STD = 1.0

# Generate random features
X_input = np.random.randn(N, D)
# Add bias column
X_design = np.hstack([X_input, np.ones((N, 1))])
W_TRUE = np.concatenate([TRUE_W_SMALL, [5.0]]) # [w1, w2...w10, b]

# Target Vector y: y = X @ W_TRUE + noise
y = X_design @ W_TRUE + np.random.normal(0, NOISE_STD, N)

# Define the identity matrix for the regularization term (D+1 dimensions)
I_reg = np.eye(D + 1)
I_reg[-1, -1] = 0 # Do not regularize the bias term (b)

# ====================================================================
# 2. Closed-Form Solutions (OLS vs. Ridge)
# ====================================================================

def solve_ridge(X_design, y, lambda_reg, I_reg):
    """
    Solves the Ridge/OLS problem analytically.
    w* = (X^T X + lambda*I)^{-1} X^T y
    """
    XTX = X_design.T @ X_design
    XTy = X_design.T @ y
    
    # Add regularization term to XTX
    reg_term = lambda_reg * I_reg
    w_star = np.linalg.solve(XTX + reg_term, XTy)
    return w_star

# --- Scenario A: OLS (No Regularization, lambda = 0) ---
W_OLS = solve_ridge(X_design, y, lambda_reg=0.0, I_reg=I_reg)

# --- Scenario B: Ridge (Strong Regularization, lambda = 10) ---
W_RIDGE = solve_ridge(X_design, y, lambda_reg=10.0, I_reg=I_reg)

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Prepare data for plotting weights (excluding bias term at index D)
w_names = [f'$w_{i+1}$' for i in range(D)]
w_true_val = W_TRUE[:D]
w_ols_val = W_OLS[:D]
w_ridge_val = W_RIDGE[:D]

df_weights = pd.DataFrame({
    'True': w_true_val,
    'OLS (\u03bb=0)': w_ols_val,
    'Ridge (\u03bb=10)': w_ridge_val
}, index=w_names)

print("--- Weight Comparison: OLS vs. Ridge Regression ---")
print(df_weights.to_markdown(floatfmt=".3f"))

# Plot 1: Weight Magnitudes
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(D)
width = 0.25

ax.bar(x - width, w_true_val, width, label='True Weight', color='gray')
ax.bar(x, w_ols_val, width, label='OLS (Unregularized)', color='skyblue')
ax.bar(x + width, w_ridge_val, width, label='Ridge (Regularized, $\\lambda=10$)', color='darkred')

# Labeling and Formatting
ax.set_title('Regularization Shrinks Weight Magnitude')
ax.set_xlabel('Feature Weight')
ax.set_ylabel('Weight Magnitude')
ax.set_xticks(x)
ax.set_xticklabels(w_names)
ax.axhline(0, color='k', lw=0.8)
ax.legend()
plt.tight_layout()
plt.show()

print("\nConclusion: The OLS weights for irrelevant features (w2-w10) are large and noisy. Ridge Regression successfully applies the L2 penalty, shrinking the magnitude of these irrelevant weights toward zero. This demonstrates that regularization is computationally equivalent to introducing a strong **Gaussian Prior** that favors simpler models.")
