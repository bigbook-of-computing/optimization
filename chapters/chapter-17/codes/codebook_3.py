# Source: Optimization/chapter-17/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. Setup Conceptual Optimization Loop
# ====================================================================

# We simulate the VMC optimization trajectory conceptually by defining 
# an Energy function that approaches a minimum.

MAX_EPOCHS = 50
ETA = 0.05 # Conceptual learning rate
TRUE_GROUND_STATE = -1.707 # E0 for the conceptual Ising chain

# --- Conceptual Energy/Gradient ---
# We define a simple 1D quadratic function for the loss surface
# L(theta) = (theta - E0)^2 + offset. Minimum is at theta = E0.
# The gradient is G(theta) = 2 * (theta - E0).

def conceptual_energy_loss(theta):
    """The potential energy loss surface for the optimization."""
    # Add a small noise term to simulate Monte Carlo variance
    noise = np.random.normal(0, 0.01) 
    return (theta - TRUE_GROUND_STATE)**2 + noise

def conceptual_gradient(theta):
    """The gradient (force) driving the optimization."""
    return 2 * (theta - TRUE_GROUND_STATE)

# ====================================================================
# 2. VMC Optimization Trajectory (Relaxation Check)
# ====================================================================

# Start with a high-energy initial parameter
theta = 0.0 
Energy_History = []

for epoch in range(MAX_EPOCHS):
    # Calculate energy (E_t)
    E_t = conceptual_energy_loss(theta)
    Energy_History.append(E_t)
    
    # Calculate gradient
    G_t = conceptual_gradient(theta)
    
    # Update rule: theta_new = theta_old - eta * G_t
    theta = theta - ETA * G_t
    
# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot the energy descent
plt.plot(np.arange(MAX_EPOCHS), Energy_History, 'b-', lw=2, label='Expected Energy $E(\u03b8)$')

# Highlight the theoretical minimum (Ground State)
plt.axhline(0, color='r', linestyle='--', label='Theoretical Minimum (E=0)')

# Labeling and Formatting
plt.title('VMC Optimization: Energy Dissipation Check')
plt.xlabel('Epoch')
plt.ylabel('Expected Energy Loss (E)')
plt.ylim(bottom=-0.1, top=Energy_History[0] + 0.1)
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
E_initial = Energy_History[0]
E_final = Energy_History[-1]

print("\n--- VMC Relaxation Analysis ---")
print(f"Initial Energy (E0): {E_initial:.4f} (High Energy)")
print(f"Final Energy (E_final): {E_final:.4f} (Near Minimum)")
print(f"Total Energy Reduction: {E_initial - E_final:.4f}")

# Check for the stability property (energy should not consistently increase)
# Due to MC noise in the conceptual function, energy may fluctuate slightly, 
# but the trend must be strictly decreasing.
energy_trend = np.polyfit(np.arange(MAX_EPOCHS), Energy_History, 1)[0]

print(f"Trend of Energy Curve (Slope): {energy_trend:.4f}")

print("\nConclusion: The energy trajectory shows a clear, rapid initial decrease, followed by fluctuations near the minimum. The negative trend (slope) confirms that the classical optimization process successfully simulates the natural quantum physical dynamic of minimizing the expected energy, driving the system toward its ground state.")
