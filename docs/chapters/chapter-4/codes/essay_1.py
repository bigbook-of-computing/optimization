# Source: Optimization/chapter-4/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

# Define the 2D parameter grid
theta1, theta2 = np.meshgrid(np.linspace(-3, 3, 200), 
                             np.linspace(-3, 3, 200))

# --- Define Loss Surfaces ---

# 1. The Convex Landscape (Quadratic Bowl)
# L = theta1^2 + 4*theta2^2
L_quad = theta1**2 + 4*theta2**2

# 2. The Non-Convex Landscape (Rugged Surface)
# L = (theta1^2 + 4*theta2^2) + 0.3*sin(5*theta1)*cos(5*theta2)
L_rugged = L_quad + 0.3 * np.sin(5 * theta1) * np.cos(5 * theta2)

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

# Plot the Quadratic (Convex) Landscape
axs[0].contourf(theta1, theta2, L_quad, levels=40, cmap='viridis')
axs[0].set_title('Convex Landscape (Quadratic)')
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\theta_2$')
axs[0].set_aspect('equal') # Ensure aspect ratio is equal

# Plot the Rugged (Non-Convex) Landscape
cs = axs[1].contourf(theta1, theta2, L_rugged, levels=40, cmap='viridis')
axs[1].set_title('Non-Convex Landscape (Rugged)')
axs[1].set_xlabel(r'$\theta_1$')
axs[1].set_ylabel(r'$\theta_2$')
axs[1].set_aspect('equal')

fig.suptitle('Optimization Landscapes')
fig.colorbar(cs, ax=axs[1], label='Loss Value $L(\mathbf{\\theta})$')
plt.tight_layout()
plt.show()
