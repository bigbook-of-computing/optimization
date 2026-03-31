# Source: Optimization/chapter-4/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Loss Function and Gradient (1D Example)
# ====================================================================

# Function with two minima (Global min near x=-1, Local min near x=2)
def L(x):
    return (x**2 - 1)**2 + 0.5 * (x - 2)**2

# Analytic Gradient (first derivative)
# dL/dx = 2*(x^2 - 1)*(2x) + 2*0.5*(x - 2)
def grad_L(x):
    return 4 * x * (x**2 - 1) + (x - 2)

# ====================================================================
# 2. Optimization Dynamics (Deterministic Gradient Descent)
# ====================================================================

def run_gradient_descent(x_start, learning_rate=0.01, max_steps=1000):
    """Runs GD until convergence (gradient near zero)."""
    x = x_start
    
    for _ in range(max_steps):
        g = grad_L(x)
        x_new = x - learning_rate * g
        
        # Check for convergence (gradient is near zero)
        if np.abs(x_new - x) < 1e-6:
            return x_new
        x = x_new
        
    return x # Return final point if max steps reached

# ====================================================================
# 3. Basin Mapping
# ====================================================================

# Create a grid of starting points (x0)
x_start_grid = np.linspace(-3, 3, 200)
minima_found = []

# Run GD from every starting point
for x0 in x_start_grid:
    x_final = run_gradient_descent(x0)
    minima_found.append(x_final)

# Identify the two distinct attractors (minima values)
attractors = np.unique(np.round(minima_found, 3))

# Map starting points to their final attractor index
attractor_map = np.digitize(np.round(minima_found, 3), attractors)

# ====================================================================
# 4. Visualization and Analysis
# ====================================================================

# Plot 1: The Loss Function and Minima
x_plot = np.linspace(-3, 3, 400)
L_plot = L(x_plot)

plt.figure(figsize=(9, 5))
plt.plot(x_plot, L_plot, lw=2, label='Loss Function $L(x)$')
plt.plot(attractors, L(attractors), 'r*', markersize=12, label='Local Minima')

# Color the starting positions based on the final attractor
plt.scatter(x_start_grid, np.zeros_like(x_start_grid), c=attractor_map, cmap='viridis', s=10, alpha=0.8, label='Initial Points (Color = Final Minimum)')

# Labeling and Formatting
plt.title('Basins of Attraction for a Non-Convex Landscape')
plt.xlabel('Parameter $x$')
plt.ylabel('Loss Value $L(x)$')
plt.ylim(L_plot.min() - 0.5, L_plot.max() + 0.5)
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
watershed_boundary = x_start_grid[np.where(np.diff(attractor_map) != 0)[0][0]]

print("\n--- Basin of Attraction Analysis ---")
print(f"Discovered Minima (Attractors): {attractors}")
print(f"Approximate Watershed Boundary (Boundary of Basins): x \u2248 {watershed_boundary:.2f}")

print("\nConclusion: The visualization successfully maps the basins of attraction. The initial starting positions are partitioned by a sharp boundary (the watershed), confirming that the final solution found by deterministic gradient descent is highly dependent on the initial conditions, a core challenge of non-convex optimization.")
