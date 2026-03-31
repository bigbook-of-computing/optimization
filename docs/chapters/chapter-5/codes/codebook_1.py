# Source: Optimization/chapter-5/codebook.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt

# ====================================================================
# 1. Setup Loss and Gradient Functions
# ====================================================================

# L(theta) = 0.5 * theta^2 (Convex Bowl)
def loss_func(theta):
    return 0.5 * theta**2

# Gradient: dL/d(theta) = theta
def gradient(theta):
    return theta

# ====================================================================
# 2. Gradient Descent Simulation
# ====================================================================

def run_gd(theta_start, learning_rate, max_steps=50):
    """Runs deterministic GD for the simple quadratic loss."""
    theta_history = np.zeros(max_steps)
    theta = theta_start
    
    for t in range(max_steps):
        # Law of Motion: theta_new = theta_old - eta * gradient(theta_old)
        grad = gradient(theta)
        theta_new = theta - learning_rate * grad
        
        theta_history[t] = theta
        theta = theta_new
        
        # Stop early if converged
        if np.abs(theta) < 1e-6:
            theta_history[t+1:] = theta
            break
            
    return theta_history

# --- Simulation Scenarios ---
THETA_START = 4.0
MAX_STEPS = 10

# Scenario A: Optimal Learning Rate (Fast Convergence)
ETA_A = 0.5
THETA_A = run_gd(THETA_START, ETA_A, MAX_STEPS)

# Scenario B: High Learning Rate (Divergence)
ETA_B = 2.1 
THETA_B = run_gd(THETA_START, ETA_B, MAX_STEPS)

# Scenario C: Critical Learning Rate (Oscillation)
ETA_C = 2.0
THETA_C = run_gd(THETA_START, ETA_C, MAX_STEPS)

# ====================================================================
# 3. Visualization
# ====================================================================

t_steps = np.arange(MAX_STEPS)

fig, ax = plt.subplots(figsize=(8, 5))

# Plot the three scenarios
ax.plot(t_steps, THETA_A, 'o-', color='darkgreen', label=f'Optimal $(\\eta={ETA_A})$: Convergence')
ax.plot(t_steps, THETA_B, 's--', color='darkred', label=f'Too High $(\\eta={ETA_B})$: Divergence')
ax.plot(t_steps, THETA_C, '^:', color='purple', label=f'Critical $(\\eta={ETA_C})$: Oscillation')

# Annotate the minimum
ax.axhline(0, color='k', linestyle='-', lw=0.8)

# Labeling and Formatting
ax.set_title(r'Gradient Descent Dynamics: Effect of Learning Rate $\eta$ on $L(\theta) = \frac{1}{2}\theta^2$')
ax.set_xlabel('Iteration Step $t$')
ax.set_ylabel(r'Parameter Value $\theta_t$')
ax.set_ylim(-10, 10)
ax.legend()
ax.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Gradient Descent Dynamics Summary ---")
print("Scenario A (\u03b7=0.5): The parameter decays exponentially to the minimum (\u03b8=0).")
print("Scenario B (\u03b7=2.1): The parameter overshoots the minimum and diverges (numerical instability).")
print("Scenario C (\u03b7=2.0): The parameter oscillates between +4.0 and -4.0 (critical stability limit).")
