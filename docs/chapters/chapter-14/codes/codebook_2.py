# Source: Optimization/chapter-14/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Conceptual Loss Functions (Simplified 1D Game)
# ====================================================================

# We conceptualize the two competing loss functions L_G and L_D 
# as dependent on the optimization step.

# L_D: Discriminator Loss (D maximizes its objective, min_D -L_D)
# L_D decreases as D gets better at distinguishing fake data.
def loss_discriminator(step):
    # D gets better initially, then struggles as G gets better
    return 0.5 * np.exp(-0.05 * step) + 0.1 * np.sin(0.3 * step) + 0.2

# L_G: Generator Loss (G minimizes its objective L_G)
# L_G decreases as G gets better at fooling D.
def loss_generator(step):
    # G struggles initially, then gets better, but plateaus near equilibrium
    return 1.0 - 0.5 * np.exp(-0.05 * step) + 0.1 * np.cos(0.3 * step)


# ====================================================================
# 2. Simulation of Adversarial Dynamics
# ====================================================================

MAX_STEPS = 100
steps = np.arange(MAX_STEPS)

# Calculate conceptual loss curves
L_D_values = loss_discriminator(steps)
L_G_values = loss_generator(steps)

# Define the point of equilibrium (P_G = P_data, D output = 0.5)
# This corresponds to a minimum L_G and a minimum L_D.
EQUILIBRIUM_LOSS = 0.69 # Log(2) is the theoretical minimum for log loss

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot the two competing loss curves
plt.plot(steps, L_D_values, 'b-', lw=2, label='Discriminator Loss $L_D$ (D wins)')
plt.plot(steps, L_G_values, 'r-', lw=2, label='Generator Loss $L_G$ (G wins)')

# Highlight the theoretical Nash Equilibrium
plt.axhline(EQUILIBRIUM_LOSS, color='k', linestyle='--', label='Theoretical Nash Equilibrium')

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
print("Optimization goal: Minimax Game. D aims to minimize L_D; G aims to minimize L_G.")
print(f"Theoretical Nash Equilibrium Loss (log(2)): {EQUILIBRIUM_LOSS:.3f}")

print("\nConclusion: The plot shows the characteristic **oscillatory, non-convergent** behavior of GANs. The two loss functions fight each other: when D gets better (L_D drops), G must improve quickly (L_G rises, then drops again). The process stabilizes near the theoretical Nash Equilibrium where the Generator is perfectly fooling the Discriminator, confirming that GAN training is a continuous pursuit of a dynamic, adversarial balance.")
