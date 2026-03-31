# Source: Optimization/chapter-10/codebook.md -- Block 4

import numpy as np

# ====================================================================
# 1. Softmax Implementation
# ====================================================================

def softmax(z):
    """
    Implements the numerically stable Multiclass Softmax function.
    p_k = exp(z_k) / sum(exp(z_j))
    """
    # Numerical stability trick: subtract max(z) from all logits before exponentiating
    z_max = np.max(z)
    e_z = np.exp(z - z_max)
    
    # Partition function Z is the sum of exponentiated logits
    Z = np.sum(e_z) 
    
    # Probability p_k
    p = e_z / Z
    return p, Z

# ====================================================================
# 2. Scenario Testing
# ====================================================================

# Define three classes (K=3)
CLASSES = ['Class 1', 'Class 2', 'Class 3']

# --- Scenario A: Strong bias toward Class 3 ---
Z_A = np.array([1.0, 2.0, 3.0])
P_A, Z_A_part = softmax(Z_A)

# --- Scenario B: Strong bias toward Class 1 ---
Z_B = np.array([3.0, 2.0, 1.0])
P_B, Z_B_part = softmax(Z_B)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Multiclass Softmax as Boltzmann Distribution ---")
print(f"Logits are analogous to Negative Energy: z_k \u223d -E_k")

print("\n1. Scenario A: Logits Z_A = [1.0, 2.0, 3.0] (Bias to Class 3)")
print(f"   Partition Function (Z): {Z_A_part:.3f}")
print(f"   Probabilities (P): {np.round(P_A, 3)}")
print(f"   Sum(P): {np.sum(P_A):.0f} (Verifies constraint)")

print("\n2. Scenario B: Logits Z_B = [3.0, 2.0, 1.0] (Bias to Class 1)")
print(f"   Partition Function (Z): {Z_B_part:.3f}")
print(f"   Probabilities (P): {np.round(P_B, 3)}")
print(f"   Sum(P): {np.sum(P_B):.0f} (Verifies constraint)")

print("\nConclusion: The Softmax function successfully transforms linear scores (negative energies) into a valid probability distribution. The largest logit (highest negative energy) receives the highest probability. The denominator acts as the **Partition Function (Z)**, normalizing the distribution and completing the direct computational analogy to the physical Boltzmann distribution.")
