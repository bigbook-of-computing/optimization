# Source: Optimization/chapter-14/codebook.md -- Block 4

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Conceptual Loss Functions (Simplified 1D Game)
# ====================================================================

# We model the two competing loss functions L_G and L_D over training steps.
# The losses are simplified to show the antagonistic, oscillatory behavior.

def loss_discriminator(step):
    """D-Loss: Should decrease as D gets better, but increase as G gets better."""
    return 0.5 * np.exp(-0.05 * step) + 0.1 * np.sin(0.3 * step) + 0.35

def loss_generator(step):
    """G-Loss: Should decrease as G gets better."""
    return 1.0 - 0.5 * np.exp(-0.05 * step) + 0.1 * np.cos(0.3 * step)

# ====================================================================
# 2. Simulation of Adversarial Dynamics
# ====================================================================

MAX_STEPS = 100
steps = np.arange(MAX_STEPS)

# Calculate conceptual loss curves
L_D_values = loss_discriminator(steps)
L_G_values = loss_generator(steps)

# Define the point of equilibrium
# Theoretical minimum loss for the Discriminator (output=0.5) is ln(2) ≈ 0.69
EQUILIBRIUM_LOSS = 0.69 

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot the two competing loss curves
plt.plot(steps, L_D_values, 'b-', lw=2, label='Discriminator Loss $L_D$ (D maximizes V)')
plt.plot(steps, L_G_values, 'r-', lw=2, label='Generator Loss $L_G$ (G minimizes V)')

# Highlight the theoretical Nash Equilibrium
plt.axhline(EQUILIBRIUM_LOSS, color='k', linestyle='--', label='Theoretical Nash Equilibrium (D output \u2248 0.5)')

# Labeling and Formatting
plt.title('Generative Adversarial Network (GAN) Adversarial Dynamics')
plt.xlabel('Training Step (Epochs)')
plt.ylabel('Loss Value $L$')
plt.ylim(0.0, 1.2)
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Adversarial Equilibrium Analysis ---")
print("Optimization goal: Minimax Game. D minimizes L_D; G minimizes L_G.")
print(f"Theoretical Nash Equilibrium Loss (ln(2)): {EQUILIBRIUM_LOSS:.3f}")

print("\nConclusion: The plot shows the characteristic **oscillatory, non-convergent** behavior of GANs. The two loss functions fight each other: when D improves (L_D drops), G must immediately improve (L_G drops, but D then improves again). The system stabilizes near the theoretical Nash Equilibrium, confirming that GAN training is a continuous pursuit of a dynamic, adversarial balance.")
