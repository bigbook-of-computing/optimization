# Source: Optimization/chapter-5/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ====================================================================
# 1. Setup Loss and Gradient (Quadratic Loss with Stochastic Noise)
# ====================================================================

# True Loss: L_true = 0.5 * theta^2 (Minimum at theta_true = 0)
THETA_TRUE = 0.0

# SGD Gradient: g_SGD = theta + xi (where xi is the sampling noise)
def gradient_sgd(theta, sigma_noise):
    # True gradient: theta
    grad_true = theta
    # Sampling noise: xi_t ~ N(0, sigma_noise^2)
    noise = np.random.normal(0, sigma_noise)
    return grad_true + noise

# ====================================================================
# 2. SGD Simulation (Tracking the Ensemble)
# ====================================================================

MAX_STEPS = 1000
ETA = 0.05 # Learning rate
SIGMA_NOISE = 1.0 # Standard deviation of the stochastic gradient noise

# Start point
THETA_START = 4.0

# Store trajectory
theta_history = np.zeros(MAX_STEPS)
theta = THETA_START

for t in range(MAX_STEPS):
    theta_history[t] = theta
    
    # Calculate noisy gradient
    grad = gradient_sgd(theta, SIGMA_NOISE)
    
    # SGD Update Rule
    theta_new = theta - ETA * grad
    theta = theta_new

# Use the last 500 steps as the stationary ensemble
ENSEMBLE_SIZE = 500
theta_ensemble = theta_history[-ENSEMBLE_SIZE:]

# Ensemble Statistics
MU_ENSEMBLE = np.mean(theta_ensemble)
VAR_ENSEMBLE = np.var(theta_ensemble)

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Plot 1: Ensemble Distribution (Histogram)
plt.figure(figsize=(8, 5))

# Plot histogram of final states (the thermal ensemble)
plt.hist(theta_ensemble, bins=30, density=True, color='purple', alpha=0.6, 
         label='Final Parameter Ensemble')

# Plot the theoretical distribution center
plt.axvline(THETA_TRUE, color='red', linestyle='--', label='True Minimum $(\\theta^*=0)$')

# Overlay a Gaussian with the ensemble's calculated mean and variance
x_plot = np.linspace(theta_ensemble.min(), theta_ensemble.max(), 100)
pdf_ensemble = norm.pdf(x_plot, MU_ENSEMBLE, np.sqrt(VAR_ENSEMBLE))
plt.plot(x_plot, pdf_ensemble, 'k-', lw=2, label='Fitted Thermal Distribution')

# Labeling and Formatting
plt.title('SGD as a Thermal Ensemble: Parameter Distribution at Steady State')
plt.xlabel(r'Parameter Value $\theta$')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- SGD Thermal Ensemble Analysis ---")
print(f"True Minimum (Analytic): \u03b8* = {THETA_TRUE}")
print(f"Ensemble Mean (Simulated): \u03bc \u2248 {MU_ENSEMBLE:.4f}")
print(f"Ensemble Variance (\u03c3\u00b2): Var(\u03b8) \u2248 {VAR_ENSEMBLE:.4f}")

print("\nConclusion: The simulation shows that the parameters under SGD do not converge to a single point but form a **statistical ensemble** (a distribution) centered near the true minimum. This is the result of the thermal noise (\u03be_t) preventing the system from reaching absolute zero temperature, confirming that SGD operates as a high-dimensional physical system in a **non-equilibrium steady state**.")
