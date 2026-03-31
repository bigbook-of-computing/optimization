# Source: Optimization/chapter-4/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Define the Loss Function and Analytic Gradient
# ====================================================================

# Loss Function (L1): L = theta1^2 + 4*theta2^2
def L1(t1, t2):
    return t1**2 + 4*t2**2

# Analytic Gradient: dL/d(theta) = (2*theta1, 8*theta2)
def grad_L1(t1, t2):
    dL_dt1 = 2 * t1
    dL_dt2 = 8 * t2
    # Return the negative gradient for the force field visualization
    return -dL_dt1, -dL_dt2

# ====================================================================
# 2. Setup Grid and Vector Field
# ====================================================================

# Parameter space grid
theta_range = np.linspace(-3, 3, 50)
theta1, theta2 = np.meshgrid(theta_range, theta_range)

# Calculate Loss Surface (for contours)
L_surface = L1(theta1, theta2)

# Calculate Negative Gradient Field (Force Vectors)
U, V = grad_L1(theta1, theta2)

# ====================================================================
# 3. Visualization
# ====================================================================

plt.figure(figsize=(8, 6))

# Plot 1: Contour Map of the Loss
plt.contourf(theta1, theta2, L_surface, levels=40, cmap='viridis')
plt.colorbar(label='Loss Value $L(\mathbf{\theta})$')

# Plot 2: Negative Gradient Field (Quiver Plot)
# The arrows show the direction of the optimization force (steepest descent)
plt.quiver(theta1, theta2, U, V, color='white', alpha=0.7, scale=60, headwidth=4)

# Highlight the minimum
plt.plot(0, 0, 'r*', markersize=15, label='Global Minimum')

# Labeling and Formatting
plt.title('Loss Landscape with Negative Gradient Field $(-\\nabla L)$')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Gradient Field Interpretation ---")
print("Observation: The quiver arrows (negative gradient/force) are visually perpendicular to the loss contours (lines of constant loss) and point directly toward the minimum at (0, 0).")
print("Interpretation: This confirms that the force acting on the model parameters always drives the system along the path of steepest descent, which is the foundational principle of gradient descent optimization.")
