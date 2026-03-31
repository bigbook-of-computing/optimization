# Source: Optimization/chapter-10/codebook.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Data Generation
# ====================================================================

N = 100  # Number of samples
TRUE_W = np.array([3.0, 5.0])  # True weights (w1, bias b)
NOISE_STD = 2.0

# Generate synthetic linear data: y = w1*x + b + noise
X_input = np.linspace(0, 10, N)
X_bias = np.ones(N)

# Design Matrix X: [X_input, X_bias]
X_design = np.vstack([X_input, X_bias]).T 

# Target Vector y: y = 3.0*x + 5.0 + noise
y = X_design @ TRUE_W + np.random.normal(0, NOISE_STD, N)

# ====================================================================
# 2. Closed-Form Solution (Normal Equation)
# ====================================================================

# Formula: w* = (X^T X)^{-1} X^T y
XTX = X_design.T @ X_design
XTy = X_design.T @ y

# Compute inverse and solve for w* (using np.linalg.solve for stability)
w_mle = np.linalg.solve(XTX, XTy)

# Separate weight and bias for interpretation
w1_mle, b_mle = w_mle

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Prediction line
y_pred = X_design @ w_mle
y_true_line = X_design @ TRUE_W

plt.figure(figsize=(9, 6))

# Plot raw data
plt.scatter(X_input, y, label='Simulated Data', alpha=0.6)

# Plot the true line
plt.plot(X_input, y_true_line, 'k--', label=f'True Model ($y={TRUE_W[0]:.2f}x + {TRUE_W[1]:.2f}$)')

# Plot the fitted line (MLE solution)
plt.plot(X_input, y_pred, 'r-', lw=2, label=f'MLE Fit ($y={w1_mle:.2f}x + {b_mle:.2f}$)')

# Labeling and Formatting
plt.title('Linear Regression: Closed-Form Maximum Likelihood Estimate (MLE)')
plt.xlabel('Input Feature $x$')
plt.ylabel('Target $y$')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Linear Regression (OLS/MLE) Summary ---")
print(f"True Weights (w, b): {TRUE_W}")
print(f"MLE Estimates (w*, b*): {np.round(w_mle, 3)}")

print("\nConclusion: The closed-form solution successfully calculated the optimal weights (w*, b*) in a single matrix operation. The fitted line closely matches the true underlying function, demonstrating the equivalence between minimizing the Sum of Squared Errors (SSE) and finding the Maximum Likelihood Estimate under Gaussian noise assumptions.")
