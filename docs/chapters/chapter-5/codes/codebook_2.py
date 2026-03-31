# Source: Optimization/chapter-5/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Loss and Gradient Functions
# ====================================================================

# Loss Function (Anisotropic Bowl): L = 0.5*theta1^2 + 5*theta2^2
# Note: The gradient for L = 0.5*theta1^2 + 5*theta2^2 is (theta1, 10*theta2)
def L_aniso(t1, t2):
    return 0.5 * t1**2 + 5 * t2**2

# Gradient: dL/d(theta) = (theta1, 10*theta2)
def grad_L_aniso(t1, t2):
    dL_dt1 = t1
    dL_dt2 = 10 * t2
    return np.array([dL_dt1, dL_dt2])

# ====================================================================
# 2. Gradient Descent Simulation (with Zigzagging)
# ====================================================================

MAX_STEPS = 20
ETA = 0.09 # High enough to cause visible zigzagging but not divergence

# Starting position (off-axis to demonstrate anisotropy)
THETA_START = np.array([3.0, 0.8])

# Store trajectory
theta_history = np.zeros((MAX_STEPS, 2))
theta = THETA_START.copy()

for t in range(MAX_STEPS):
    theta_history[t] = theta
    
    # Calculate gradient
    grad = grad_L_aniso(theta[0], theta[1])
    
    # Update rule
    theta_new = theta - ETA * grad
    theta = theta_new
    
# Final minimum point
theta_min = np.array([0.0, 0.0])

# ====================================================================
# 3. Visualization
# ====================================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Plot 1: Contour Map of the Anisotropic Loss
t1_plot, t2_plot = np.meshgrid(np.linspace(-3.5, 3.5, 100), np.linspace(-1, 1, 100))
L_surface = L_aniso(t1_plot, t2_plot)
levels = np.logspace(0, np.log10(L_surface.max()), 15) # Log-spaced contours for better visibility
plt.contour(t1_plot, t2_plot, L_surface, levels=levels, colors='gray', alpha=0.6)

# Plot 2: The Gradient Descent Trajectory
plt.plot(theta_history[:, 0], theta_history[:, 1], 'r-', lw=2, label='GD Trajectory')
plt.plot(theta_history[:, 0], theta_history[:, 1], 'bo', markersize=5, label='GD Steps')

# Highlight the start and end
plt.plot(THETA_START[0], THETA_START[1], 'go', markersize=8, label='Start')
plt.plot(theta_min[0], theta_min[1], 'r*', markersize=12, label='Minimum')

# Labeling and Formatting
ax.set_title(f'Gradient Descent Path in an Anisotropic Loss Landscape ($\\eta={ETA}$)')
ax.set_xlabel(r'$\theta_1$ (Sloppy Direction)')
ax.set_ylabel(r'$\theta_2$ (Stiff Direction)')
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Anisotropic Dynamics Summary ---")
print(f"Starting Point: ({THETA_START[0]}, {THETA_START[1]})")
print(f"Final Point: ({theta_history[-1, 0]:.4f}, {theta_history[-1, 1]:.4f})")
print("Observation: The GD path exhibits characteristic **zigzagging** (oscillation) across the steep (\u03b8_2) direction while making slow, steady progress along the shallow (\u03b8_1) direction. This confirms that GD's local, first-order rule is highly inefficient on anisotropic surfaces, motivating the use of advanced techniques that incorporate second-order information (curvature).")
