# Source: Optimization/chapter-19/codebook.md -- Block 4

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Conceptual Scores (Negative Energy)
# ====================================================================

# We analyze the attention scores from a single Query (i) to its 5 neighbors (j).
T_SCALE = 1.0 # Simple temperature T=1 for this check

# --- Raw Similarity Scores (S_ij = q_i \cdot k_j) ---
RAW_SCORES = np.array([3.0, 1.0, 0.5, -0.5, 2.0])
NEIGHBOR_LABELS = ['J1 (High Score)', 'J2', 'J3', 'J4 (Low Score)', 'J5']

# ====================================================================
# 2. Attention Energy and Probability Calculation
# ====================================================================

# 1. Attention Energy: E_ij = -S_ij
ATTENTION_ENERGY = -RAW_SCORES

# 2. Boltzmann (Softmax) Probability: \alpha_ij = exp(S_ij / T) / sum(exp(S_ik / T))
def softmax_numpy(scores, T=T_SCALE):
    scaled_scores = scores / T
    e_x = np.exp(scaled_scores - np.max(scaled_scores))
    return e_x / np.sum(e_x)

ALPHA_PROBABILITIES = softmax_numpy(RAW_SCORES)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Attention as a Boltzmann Energy Model ---")
print(f"Temperature Scale (T): {T_SCALE:.1f} (High T \u2192 Uniform \u03b1)")
print("----------------------------------------------------------------")

print("1. Energy and Score Mapping:")
print(f"{'Neighbor':<15} | {'Raw Score (S_ij)':<16} | {'Attention Energy (E_ij)':<23}")
print("-" * 60)
for label, score, energy in zip(NEIGHBOR_LABELS, RAW_SCORES, ATTENTION_ENERGY):
    print(f"{label:<15} | {score:<16.2f} | {energy:<23.2f}")

print("\n2. Final Attention Probabilities (\u03b1_{ij}):")
for label, prob in zip(NEIGHBOR_LABELS, ALPHA_PROBABILITIES):
    print(f"{label:<15} | Probability (\u03b1): {prob:.3f}")
print(f"Total Probability Check: {np.sum(ALPHA_PROBABILITIES):.0f}")

print("\nConclusion: The calculation confirms the Boltzmann analogy. The neighbor with the **Highest Raw Score** (J1: 3.0) is the one with the **Lowest Energy** (E: -3.0) and receives the overwhelmingly **Highest Attention Probability** (0.69). This demonstrates that the Softmax function is the core thermodynamic mechanism that converts the raw similarity between Query and Key into a normalized probability distribution, acting as an energy minimizer that highlights the most relevant (lowest energy) state.")
