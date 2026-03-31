# Source: Optimization/chapter-14/codebook.md -- Block 1

import numpy as np

# ====================================================================
# 1. Setup Conceptual Loss Components
# ====================================================================

# We model the ELBO components conceptually, as calculated from a single step
# of a VAE where Q is a Gaussian Q ~ N(mu_Q, sigma_Q).

# Define a function to calculate the negative entropy term: -E_Q[ln Q(\theta)]
# For a Gaussian, this is proportional to the variance (log(sigma^2)).
def entropy(sigma_q):
    # Entropy H(Q) = 0.5 * log(2*pi*e*sigma_q^2)
    return 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_q**2)

# Define the ELBO function based on conceptual loss values (since we don't run the VAE)
def calculate_elbo(L_recon, D_kl, beta):
    """
    Conceptual ELBO calculation: ELBO = -L_Recon - beta * D_KL
    Note: L_Recon is the Mean Squared Error (MSE), so we negate it for the ELBO.
    """
    return -(L_recon + beta * D_kl)

# --- Define fixed cost values for two comparison scenarios ---
# Scenario A: Standard VAE (\beta = 1.0)
L_RECON_A = 5.0 # MSE
D_KL_A = 2.0    # KL Divergence

# Scenario B: Beta-VAE (\beta = 5.0)
# Assume \beta=5.0 forces a simpler latent space (D_KL shrinks) but increases recon error.
L_RECON_B = 8.0 # Higher reconstruction loss
D_KL_B = 1.0    # Lower KL Divergence (closer to prior)

# ====================================================================
# 2. ELBO Calculation and Comparison
# ====================================================================

BETA_A = 1.0
BETA_B = 5.0

# Calculate ELBO for both scenarios
ELBO_A = calculate_elbo(L_RECON_A, D_KL_A, BETA_A)
ELBO_B = calculate_elbo(L_RECON_B, D_KL_B, BETA_B)

# Calculate the contribution of each term to the *total loss* (L_Recon + \beta*D_KL)
Loss_Total_A = L_RECON_A + BETA_A * D_KL_A
Loss_Total_B = L_RECON_B + BETA_B * D_KL_B

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- VAE: Energy-Entropy Trade-Off Analysis (\u03b2-VAE) ---")
print(r"Objective: Minimize Total Loss = L_Recon + \beta * D_KL")

print("\n--- Scenario A: Standard VAE (\u03b2=1.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECON_A:.1f}")
print(f"2. Entropy Cost (D_KL): {D_KL_A:.1f}")
print(f"Total Loss: {Loss_Total_A:.1f} | ELBO: {ELBO_A:.1f}")

print("\n--- Scenario B: Beta-VAE (\u03b2=5.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECON_B:.1f} (Sacrificed Fidelity)")
print(f"2. Entropy Cost (D_KL): {D_KL_B:.1f}")
print(f"Total Loss: {Loss_Total_B:.1f} | ELBO: {ELBO_B:.1f} (Increased Total Loss)")
print("----------------------------------------------------------")

print("\nObservation: By increasing the Entropy Penalty (\u03b2 from 1.0 to 5.0), the VAE found a latent distribution that is simpler (D_KL dropped from 2.0 to 1.0) but resulted in worse image quality (Recon Loss increased from 5.0 to 8.0).")
print("\nConclusion: This demonstrates the energy-entropy trade-off. The model must balance **high fidelity** (low Energy/Recon Loss) against **structural regularity** (low Entropy/KL Cost). The \u03b2 factor allows explicit control over the relative importance of these two physical principles.")
