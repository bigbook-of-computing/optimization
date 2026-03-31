# Source: Optimization/chapter-7/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Rugged Loss Function (Energy Landscape) ---
def L(x, y):
    """The non-convex loss function with four major wells and high-frequency ripples."""
    # Global structure: four deep wells at (±1, ±1)
    L_global = (x**2 - 1)**2 + (y**2 - 1)**2
    # Local roughness: high-frequency perturbation
    L_perturb = 0.3 * np.sin(5*x) * np.cos(5*y)
    return L_global + L_perturb

# --- 2. Simulated Annealing Implementation ---
np.random.seed(0)

theta = np.array([2.5, 2.5]) # Initial parameter vector (starting position)
T = 1.0                      # Initial high temperature (T0 = 1.0)
trajectory = [theta.copy()]  # List to store the history of theta

for t in range(2000):
    # Propose a random move (Gaussian random walk)
    # The magnitude of the random step (0.2) controls local exploration scale
    proposal = theta + 0.2 * np.random.randn(2)
    
    # Calculate the change in Loss (dL = E' - E)
    dL = L(*proposal) - L(*theta)
    
    # Acceptance probability P_acc = min(1, exp(-dL / T))
    # This is the core thermodynamic rule (Metropolis criterion)
    if np.random.rand() < np.exp(-dL / T):
        theta = proposal # Accept the move (either downhill or lucky uphill)
        
    trajectory.append(theta.copy())
    
    # Annealing/Cooling Schedule (T gradually decays)
    T *= 0.995  # Slow cooling rate

trajectory = np.array(trajectory)

# --- 3. Visualization ---
# Define the contour space for the loss function L(x,y)
x = np.linspace(-3.5, 3.5, 100)
y = np.linspace(-3.5, 3.5, 100)
X, Y = np.meshgrid(x, y)
Z = L(X, Y)

plt.figure(figsize=(9, 7))
# Plot the loss landscape contours
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Loss Value L(x,y)')

# Plot the SA trajectory
plt.plot(trajectory[:,0], trajectory[:,1], lw=1.5, color='white', alpha=0.8, label=f'SA Trajectory (T$_0$={1.0})')
plt.scatter(trajectory[0,0], trajectory[0,1], s=100, color='red', marker='o', label='Start (2.5, 2.5)')
plt.scatter(trajectory[-1,0], trajectory[-1,1], s=100, color='gold', marker='*', label='End/Converged State')

plt.title('Simulated Annealing Trajectory on Rugged Loss Landscape')
plt.xlabel(r'Parameter $x$')
plt.ylabel(r'Parameter $y$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.2)
plt.show()
