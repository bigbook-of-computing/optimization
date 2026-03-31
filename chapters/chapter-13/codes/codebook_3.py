# Source: Optimization/chapter-13/codebook.md -- Block 3

import numpy as np

# ====================================================================
# 1. Setup Conceptual Loss Components
# ====================================================================

# Conceptual values for the three losses from a single training step
# Assume the optimization is running, and these values are calculated.

# Cost 1: Reconstruction Error (Energy Term)
# Lower is better (we maximize -L_reconstruction)
L_RECONSTRUCTION = 5.0 

# Cost 2: KL Divergence (Entropy Cost)
# Lower is better (we maximize -D_KL)
D_KL = 2.0 

# Total Loss (Objective) is L_RECONSTRUCTION + \beta * D_KL
# The ELBO is ELBO = -Total Loss

# --- Scenario A: Standard VAE (\beta = 1.0) ---
BETA_A = 1.0
ELBO_A = - (L_RECONSTRUCTION + BETA_A * D_KL)

# --- Scenario B: High Entropy Cost (\beta = 5.0, forcing simpler latent space) ---
BETA_B = 5.0
# Assume that forcing the KL term higher (BETA_B=5) leads to higher reconstruction error 
# because the model is more constrained.
L_RECONSTRUCTION_B = 8.0 
ELBO_B = - (L_RECONSTRUCTION_B + BETA_B * D_KL)

# ====================================================================
# 2. Analysis and Summary
# ====================================================================

print("--- VAE: Energy-Entropy Trade-Off Analysis ---")
print(r"Objective: Maximize ELBO = - (L_Recon + \beta * D_KL)")

print("\n--- Scenario A: Standard VAE (\u03b2=1.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECONSTRUCTION:.1f}")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_A * D_KL:.1f}")
print(f"Total Loss: {L_RECONSTRUCTION + BETA_A * D_KL:.1f} | ELBO: {ELBO_A:.1f}")

print("\n--- Scenario B: Beta-VAE (\u03b2=5.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECONSTRUCTION_B:.1f} (Increased)")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_B * D_KL:.1f} (Penalty is 5x stronger)")
print(f"Total Loss: {L_RECONSTRUCTION_B + BETA_B * D_KL:.1f} | ELBO: {ELBO_B:.1f}")
print("----------------------------------------------------------")

print("\nObservation: The high \u03b2 (5.0) in Scenario B drastically increased the penalty for the complexity of the latent representation (Entropy Cost = 10.0), even though it resulted in a worse image quality (Energy Cost = 8.0).")
print("\nConclusion: This demonstrates the **Energy-Entropy Trade-Off**: the \u03b2 parameter allows the user to balance the desire for **high fidelity** (low Energy Cost) against the desire for a **simple, structured latent space** (low Entropy Cost). This control is crucial for disentangling the generative factors of the data.")
