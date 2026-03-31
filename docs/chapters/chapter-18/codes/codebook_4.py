# Source: Optimization/chapter-18/codebook.md -- Block 4

import numpy as np

# ====================================================================
# 1. Setup Conceptual Parameters and Input Features
# ====================================================================

# Node of interest: Node i
# Neighbors: Node j1, Node j2, Node j3

# Conceptual features of the node (hi) and its neighbors (h_j)
# Features are assumed to be pre-transformed: h' = W*h
H_PRIME_I = np.array([1.0, 0.5])  # Features of Node i
H_PRIME_J1 = np.array([2.0, 0.0]) # Neighbor 1 (Highly relevant/Compatible)
H_PRIME_J2 = np.array([0.0, 0.0]) # Neighbor 2 (Neutral)
H_PRIME_J3 = np.array([1.0, 1.0]) # Neighbor 3 (Moderately relevant)

NEIGHBOR_FEATURES = np.array([H_PRIME_J1, H_PRIME_J2, H_PRIME_J3])

# Trainable Attention Vector (a)
# This vector is learned during training and dictates importance
A_VEC = np.array([0.3, 0.7, 0.5, 0.5]) # a^T has dimension 2*F' (here 4)

# ====================================================================
# 2. GAT Attention Coefficient Calculation
# ====================================================================

# Step 1: Calculate the Compatibility Score (e_ij)
compatibility_scores = []

for h_j in NEIGHBOR_FEATURES:
    # Concatenate features: [h'_i || h'_j] (Dimension 4)
    concatenated = np.concatenate([H_PRIME_I, h_j])
    
    # Compatibility Score: e_ij = a^T * [h'_i || h'_j]
    e_ij = np.dot(A_VEC, concatenated)
    compatibility_scores.append(e_ij)

E_SCORES = np.array(compatibility_scores)

# Step 2: Normalize with Softmax to get Attention Coefficients (\alpha_ij)
def softmax_numpy(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

ALPHA_COEFFICIENTS = softmax_numpy(E_SCORES)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- GAT Attention Coefficient (\u03b1_{ij}) Analysis ---")
print(f"Node i Features: {H_PRIME_I}")
print(f"Trainable Attention Vector (a): {A_VEC}")
print("----------------------------------------------------------------")

print("1. Compatibility Scores (e_ij) before Softmax:")
print(f"  e_i,j1 (Relevant): {E_SCORES[0]:.3f}")
print(f"  e_i,j2 (Neutral): {E_SCORES[1]:.3f}")
print(f"  e_i,j3 (Moderate): {E_SCORES[2]:.3f}")

print("\n2. Final Attention Coefficients (\u03b1_{ij}):")
print(f"  \u03b1_i,j1: {ALPHA_COEFFICIENTS[0]:.3f}")
print(f"  \u03b1_i,j2: {ALPHA_COEFFICIENTS[1]:.3f}")
print(f"  \u03b1_i,j3: {ALPHA_COEFFICIENTS[2]:.3f}")

print("\nConclusion: The GAT successfully assigns a variable weighting to its neighbors. The most compatible neighbor (Node j1) receives the highest attention coefficient (\u03b1_i,j1), while the least compatible (Node j2) receives the lowest. This demonstrates that GAT overcomes the limitation of fixed averaging by allowing the network to dynamically **prioritize messages** based on their content, leading to more expressive and robust relational modeling.")
