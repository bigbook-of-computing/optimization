# Chapter 4: The Optimization Landscape


-----


## Project 1: Visualizing Convex vs. Rugged Landscapes (Replication)


-----


### Definition: Visualizing Convex vs. Rugged Landscapes

The goal is to replicate the visualization of two contrasting two-dimensional loss surfaces: a **convex** landscape ($L_1$) and a **non-convex (rugged)** landscape ($L_2$). This visually demonstrates the fundamental difference between simple, guaranteed-solvable optimization problems and the complex, trap-filled terrains found in deep learning.

### Theory: Convexity and Landscape Complexity

The **Loss Function $L(\boldsymbol{\theta})$** is the analogue of the system's **Potential Energy $E[\mathbf{s}]$**.

1.  **Convex Landscape ($L_1$):** This perfect "bowl" (paraboloid) is defined by the **anisotropic quadratic loss**:
    $$L_1(\theta_1, \theta_2) = \theta_1^2 + 4\theta_2^2$$
    Any local minimum is guaranteed to be the **global minimum**. The term $4\theta_2^2$ makes the surface **anisotropic** (steeper in the $\theta_2$ direction).

2.  **Non-Convex Landscape ($L_2$):** This surface is created by adding a high-frequency, oscillating perturbation term to $L_1$:
    $$L_2(\theta_1, \theta_2) = (\theta_1^2 + 4\theta_2^2) + 0.3 \sin(5\theta_1) \cos(5\theta_2)$$
    The perturbation creates numerous **local minima** and **saddle points** (analogous to a **spin glass** energy surface), which can trap a simple optimizer. The ruggedness is a model for the complexity introduced by non-linear model architectures and finite sampling noise.

-----

### Extensive Python Code and Visualization

```python
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
```

-----

## Project 2: Calculating and Interpreting the Gradient Field

-----

### Definition: Calculating and Interpreting the Gradient Field

The goal is to compute the **analytic gradient ($\boldsymbol{\nabla L}$)** for the convex landscape $L_1(\theta_1, \theta_2) = \theta_1^2 + 4\theta_2^2$ and visualize the resulting negative gradient field ($-\boldsymbol{\nabla L}$).

### Theory: Gradient as Force

The **Gradient ($\boldsymbol{\nabla L}$)** is the first derivative vector, representing the direction of **steepest ascent** on the loss surface.

The core analogy of optimization is that the process is driven by the **negative gradient (force)**:

$$\mathbf{F}_{\text{optim}} = - \nabla L(\boldsymbol{\theta})$$

The analytic gradient for the convex landscape is calculated as:

$$\nabla L_1 = \left( \frac{\partial L_1}{\partial \theta_1}, \frac{\partial L_1}{\partial \theta_2} \right) = (2\theta_1, 8\theta_2)$$

Visualizing the negative gradient field confirms that the arrows representing the optimization force are always **perpendicular to the constant-loss contour lines** and point directly toward the minimum, following the **path of steepest descent**.

-----

### Extensive Python Code and Visualization

```python
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
```

-----

## Project 3: Visualizing the Basin of Attraction (Conceptual)

-----

### Definition: Visualizing the Basin of Attraction

The goal is to create a conceptual visual map of the **basins of attraction** for a simple one-dimensional function with two distinct local minima. This demonstrates how the initial starting position ($\theta_0$) determines the final converged solution ($\theta_L$).

### Theory: Basins and the Watershed

The **Basin of Attraction** for a local minimum $\boldsymbol{\theta}_L$ is the region of parameter space ($\boldsymbol{\theta}_0$) from which a deterministic optimizer will converge to $\boldsymbol{\theta}_L$.

For a non-convex landscape, the parameter space is partitioned by a sharp boundary called the **watershed** or **ridge**.

We use a function with two competing minima:

$$L(x) = (x^2-1)^2 + 0.5(x-2)^2$$

This landscape has a global minimum near $x=-1$ and a local minimum near $x=2$. The simulation runs a simple **deterministic gradient descent** from many initial points ($x_0$) and colors the starting point based on which minimum it finds, visually mapping the basins.

-----

### Extensive Python Code and Visualization

```python
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
```

-----

## Project 4: Hessian Eigenvalues and Stiffness

-----

### Definition: Hessian Eigenvalues and Stiffness

The goal is to compute the **analytic Hessian matrix ($\boldsymbol{H}$)** and its **eigenvalues ($\boldsymbol{\lambda}$)** for the convex landscape $L_1(\theta_1, \theta_2) = \theta_1^2 + 4\theta_2^2$. This quantifies the **anisotropy (stiffness/sloppiness)** of the landscape, which is a major factor determining the speed of optimization.

### Theory: Curvature and Anisotropy

The **Hessian matrix ($H$)** is the matrix of second derivatives, encoding the local curvature of the loss surface.

$$H = \begin{pmatrix} \frac{\partial^2 L}{\partial \theta_1^2} & \frac{\partial^2 L}{\partial \theta_1 \partial \theta_2} \\ \frac{\partial^2 L}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 L}{\partial \theta_2^2} \end{pmatrix}$$

For the function $L_1(\theta_1, \theta_2) = \theta_1^2 + 4\theta_2^2$, the analytic Hessian is:

$$H = \begin{pmatrix} 2 & 0 \\ 0 & 8 \end{pmatrix}$$

The **eigenvalues ($\lambda$)** of $H$ are the diagonal entries (since the matrix is diagonal): $\lambda_1 = 2$ and $\lambda_2 = 8$ (or vice-versa).

  * The **Condition Number** ($\kappa = \lambda_{\max} / \lambda_{\min}$) measures the ratio of stiffness (anisotropy).
  * A large difference in eigenvalues ($\kappa \gg 1$) confirms the landscape is **stiff** (narrow) in one direction and **sloppy** (wide) in the other, causing simple gradient descent to oscillate inefficiently.

-----

### Extensive Python Code and Visualization

```python
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
```
