# Source: Optimization/chapter-4/codebook.md -- Block 4

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup and Analytic Hessian Calculation
# ====================================================================

# Loss Function: L(theta1, theta2) = theta1^2 + 4*theta2^2

# First Derivatives:
# dL/d(theta1) = 2*theta1
# dL/d(theta2) = 8*theta2

# Second Derivatives (Analytic Hessian Components):
# H[0, 0] = d^2L/d(theta1)^2 = 2
# H[1, 1] = d^2L/d(theta2)^2 = 8
# H[0, 1] = H[1, 0] = d^2L/d(theta1)d(theta2) = 0

H_analytic = np.array([
    [2.0, 0.0],
    [0.0, 8.0]
])

# ====================================================================
# 2. Eigenvalue Analysis
# ====================================================================

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(H_analytic)

# The eigenvalues represent the curvature along the principal axes
lambda_min = np.min(eigenvalues)
lambda_max = np.max(eigenvalues)

# Calculate the condition number (Anisotropy ratio)
condition_number = lambda_max / lambda_min

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

print("--- Hessian Eigenvalue Analysis (Curvature) ---")
print("Analytic Hessian Matrix H:")
print(H_analytic)
print("\n--- Results ---")
print(f"Eigenvalue 1 (\u03bb_min, Sloppy Direction): {lambda_min:.2f}")
print(f"Eigenvalue 2 (\u03bb_max, Stiff Direction):  {lambda_max:.2f}")
print(f"Condition Number (\u03bb_max / \u03bb_min): {condition_number:.2f}x")

# Plot the eigenvalues
plt.figure(figsize=(6, 4))
plt.bar(['$\lambda_1$ (Sloppy)', '$\lambda_2$ (Stiff)'], eigenvalues, color=['skyblue', 'darkred'])
plt.title('Hessian Eigenvalues: Quantifying Anisotropy')
plt.ylabel('Curvature ($\lambda$)')
plt.grid(True, axis='y')
plt.show()

print("\nConclusion: The condition number of 4.0x confirms the high **anisotropy** of the loss landscape: it is four times steeper (stiffer) in the $\u03b8_2$ direction than in the $\u03b8_1$ direction. This extreme difference in curvature explains why simple gradient descent struggles, as it must use a tiny learning rate to avoid oscillating across the steep direction, resulting in slow progress along the wide, flat direction.")
