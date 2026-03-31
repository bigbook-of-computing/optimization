# Source: Optimization/chapter-4/codebook.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Define the Parameter Grid
# ====================================================================

# Range for the two parameters
theta_range = np.linspace(-3, 3, 200)
theta1, theta2 = np.meshgrid(theta_range, theta_range)

# ====================================================================
# 2. Define Loss Surfaces (L1 and L2)
# ====================================================================

# L1: Convex Quadratic Bowl (Anisotropic)
L_quad = theta1**2 + 4*theta2**2

# L2: Non-Convex Rugged Landscape (Quadratic + Oscillation)
L_rugged = L_quad + 0.3 * np.sin(5 * theta1) * np.cos(5 * theta2)

# ====================================================================
# 3. Visualization
# ====================================================================

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

# Plot 1: The Convex Landscape (Quadratic)
cs1 = axs[0].contourf(theta1, theta2, L_quad, levels=40, cmap='viridis')
axs[0].set_title('Convex Landscape ($L_1$: Quadratic Bowl)')
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\theta_2$')
axs[0].set_aspect('equal')

# Plot 2: The Rugged (Non-Convex) Landscape
cs2 = axs[1].contourf(theta1, theta2, L_rugged, levels=40, cmap='viridis')
axs[1].set_title('Non-Convex Landscape ($L_2$: Rugged)')
axs[1].set_xlabel(r'$\theta_1$')
axs[1].set_ylabel(r'$\theta_2$')
axs[1].set_aspect('equal')

fig.suptitle('Comparison of Optimization Landscapes')
fig.colorbar(cs2, ax=axs[1], label='Loss Value $L(\mathbf{\theta})$')
plt.tight_layout()
plt.show()

# --- Analysis Summary ---
print("\n--- Landscape Comparison Summary ---")
print("Convex ($L_1$): Exhibits a single, clear elliptical minimum, guaranteeing convergence to the global optimum.")
print("Non-Convex ($L_2$): Shows a corrugated surface with numerous local minima and saddle points, making global optimization difficult but necessary for complex model fitting.")
