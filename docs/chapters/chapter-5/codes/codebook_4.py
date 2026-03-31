# Source: Optimization/chapter-5/codebook.md -- Block 4

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Loss and Gradient Functions (Anisotropic Loss)
# ====================================================================

# Use the anisotropic loss from Project 2
def L_aniso(t1, t2):
    # L = 0.5*theta1^2 + 5*theta2^2
    return 0.5 * t1**2 + 5 * t2**2

def grad_L_aniso(t1, t2):
    dL_dt1 = t1
    dL_dt2 = 10 * t2
    return np.array([dL_dt1, dL_dt2])

# ====================================================================
# 2. Gradient Descent Simulation with Loss Tracking
# ====================================================================

MAX_ITER = 50
ETA = 0.05 # Stable and low learning rate
THETA_START = np.array([3.0, 0.8])

# Store cost (Energy)
J_history = []
theta = THETA_START.copy()

for t in range(MAX_ITER):
    # Calculate current Loss (Energy)
    J_history.append(L_aniso(theta[0], theta[1]))
    
    # Calculate gradient
    grad = grad_L_aniso(theta[0], theta[1])
    
    # Update rule
    theta_new = theta - ETA * grad
    theta = theta_new
    
    # Stop condition
    if np.linalg.norm(grad) < 1e-4:
        break

# ====================================================================
# 3. Visualization and Convergence Check
# ====================================================================

plt.figure(figsize=(8, 5))

# Plot the Monotonic Descent of the Objective Function J (Energy)
plt.plot(J_history, 'r-', lw=2, markersize=5)

plt.title(f'Energy Dissipation in Deterministic Gradient Descent ($\\eta={ETA}$)')
plt.xlabel('Iteration Number')
plt.ylabel('Loss $L_t$ (Energy)')
plt.grid(True)
plt.show()

# --- Analysis Summary ---
# Check for monotonicity (loss never increases)
is_monotonic = np.all(np.diff(J_history) <= 1e-9)

print("\n--- Energy Dissipation Check ---")
print(f"Initial Loss (Energy): L0 = {J_history[0]:.4f}")
print(f"Final Loss (Energy): L_final = {J_history[-1]:.4f}")
print(f"Loss Monotonically Decreasing? {is_monotonic}")

print("\nConclusion: The plot shows a smooth, monotonically decreasing loss function, confirming the **Lyapunov stability** of Gradient Descent. This behavior is the direct numerical evidence of **energy dissipation**—the system constantly sheds energy (loss) as it follows the negative gradient to find the stable equilibrium state (the minimum).")
