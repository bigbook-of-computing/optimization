# Source: Optimization/chapter-9/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Parameters and KL Divergence Function
# ====================================================================

# Analytical KL Divergence for D_KL(Q || P) where both are 1D Gaussians
def kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    """Calculates D_KL(Q || P) for two 1D Gaussians."""
    # D_KL(Q || P) = 0.5 * [log(s_p^2 / s_q^2) + (s_q^2 + (m_q - m_p)^2) / s_p^2 - 1]
    
    sigma_q_sq = sigma_q**2
    sigma_p_sq = sigma_p**2
    
    kl = 0.5 * (
        np.log(sigma_p_sq / sigma_q_sq) 
        + (sigma_q_sq + (mu_q - mu_p)**2) / sigma_p_sq 
        - 1
    )
    return kl

# --- Define fixed Prior (P) ---
MU_PRIOR = 1.0
SIGMA_PRIOR = 2.0 

# --- Define Posterior Scenarios (Q) ---
# Simulates the result of observing more data (mu moves, sigma shrinks)

# Scenario A: Initial Posterior (Q_A) - After a small amount of data
MU_A = 1.5 
SIGMA_A = 1.8 # Still wide

# Scenario B: Final Posterior (Q_B) - After much data (final convergence)
MU_B = 4.5 
SIGMA_B = 0.5 # Much narrower and closer to the true mean

# ====================================================================
# 2. Calculation of Information Gain
# ====================================================================

# Information Gain A: D_KL(Q_A || P)
KL_A = kl_divergence_gaussian(MU_A, SIGMA_A, MU_PRIOR, SIGMA_PRIOR)

# Information Gain B: D_KL(Q_B || P)
KL_B = kl_divergence_gaussian(MU_B, SIGMA_B, MU_PRIOR, SIGMA_PRIOR)

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

kl_values = [KL_A, KL_B]
names = ['Initial Gain (Q_A || P)', 'Final Gain (Q_B || P)']

plt.figure(figsize=(8, 5))

# Plot KL comparison
plt.bar(names, kl_values, color=['skyblue', 'darkred'])

plt.text(0, KL_A + 0.05, f'{KL_A:.3f} bits', ha='center', color='k', weight='bold')
plt.text(1, KL_B + 0.05, f'{KL_B:.3f} bits', ha='center', color='k', weight='bold')

# Labeling and Formatting
plt.title(r'Quantifying Information Gain using KL Divergence $D_{\text{KL}}(Q \mid\mid P)$')
plt.xlabel('Posterior Stage')
plt.ylabel('Information Gain (KL Divergence in Bits)')
plt.grid(True, axis='y')
plt.show()

# --- Analysis Summary ---
print("\n--- KL Divergence (Information Gain) Analysis ---")
print(f"Prior P: \u03bc={MU_PRIOR}, \u03c3={SIGMA_PRIOR}")
print(f"Final Posterior Q_B: \u03bc={MU_B}, \u03c3={SIGMA_B}")
print("-------------------------------------------------")
print(f"KL Divergence D_KL(Q_A || P): {KL_A:.3f} (Low gain)")
print(f"KL Divergence D_KL(Q_B || P): {KL_B:.3f} (High gain)")

print("\nConclusion: The final posterior (Q_B) is statistically much farther from the initial prior (P) than the early posterior (Q_A). This quantitative increase in KL divergence confirms the **information gain** provided by additional data: the final, more confident, and centered belief represents a major shift away from the initial, uninformed belief.")
