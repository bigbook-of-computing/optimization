# Source: Optimization/chapter-14/codebook.md -- Block 3

import numpy as np

# ====================================================================
# 1. Setup Conceptual Functions and Parameters
# ====================================================================

# Conceptual VAE loss components (approximated from a single step)

# Define a function to calculate the negative entropy term: -E_Q[ln Q(\theta)]
def entropy(sigma_q):
    """Conceptual measure of the Entropy/Spread of the latent distribution Q."""
    # Entropy H(Q) = 0.5 * log(2*pi*e*sigma_q^2)
    return 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_q**2)

# Define the ELBO function based on conceptual loss values 
def calculate_elbo(L_recon, D_kl, beta):
    """
    Conceptual ELBO calculation: ELBO = -L_Recon - beta * D_KL
    (L_Recon is the Mean Squared Error, so it's a negative component of ELBO)
    """
    return -(L_recon + beta * D_kl)

# --- Define fixed cost values for two comparison scenarios ---
# Scenario A: Standard VAE (low regularization)
L_RECON_A = 5.0 # Low reconstruction error
D_KL_A = 2.0    # High KL Divergence (complex latent space)

# Scenario B: Beta-VAE (high regularization)
# Assume Beta-VAE forces D_KL down but L_RECON up.
L_RECON_B = 8.0 # Higher reconstruction loss (sacrificed fidelity)
D_KL_B = 1.0    # Lower KL Divergence (closer to prior/simpler)

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
print(f"1. Energy Cost (Recon Loss): {L_RECON_A:.1f} (High Fidelity)")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_A * D_KL_A:.1f}")
print(f"Total Loss: {Loss_Total_A:.1f} | ELBO: {ELBO_A:.1f}")

print("\n--- Scenario B: Beta-VAE (\u03b2=5.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECON_B:.1f} (Sacrificed Fidelity)")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_B * D_KL_B:.1f} (Penalty is 5x stronger)")
print(f"Total Loss: {Loss_Total_B:.1f} | ELBO: {ELBO_B:.1f}")

print("\nConclusion: The trade-off is clear: by increasing the regularization weight (\u03b2 from 1.0 to 5.0), the model is forced to prioritize simplicity (D_KL drops from 2.0 to 1.0), even though this results in a less accurate reconstruction (Recon Loss increases). This explicitly demonstrates the **energy-entropy trade-off** required to learn a structured latent space.")
