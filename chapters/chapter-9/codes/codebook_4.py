# Source: Optimization/chapter-9/codebook.md -- Block 4

import numpy as np

# ====================================================================
# 1. Setup Network and Conditional Probability Tables (CPTs)
# ====================================================================

# Dependency: A -> B -> C (A is root, C is leaf)
# Variables are binary: 0 or 1

# P(A) - Prior for the root node
# Index [0] is P(A=0), Index [1] is P(A=1)
P_A = np.array([0.4, 0.6]) 

# P(B | A) - Conditional Probability Table (CPT)
# Rows: P(B | A=0), P(B | A=1)
# Columns: P(B=0), P(B=1)
# If A=0 (e.g., False), B is likely 0. If A=1 (e.g., True), B is likely 1.
P_B_given_A = np.array([
    [0.8, 0.2],  # P(B=0|A=0), P(B=1|A=0)
    [0.1, 0.9]   # P(B=0|A=1), P(B=1|A=1)
])

# P(C | B) - Conditional Probability Table (CPT)
# Rows: P(C | B=0), P(C | B=1)
# Columns: P(C=0), P(C=1)
P_C_given_B = np.array([
    [0.9, 0.1],  # P(C=0|B=0), P(C=1|B=0)
    [0.2, 0.8]   # P(C=0|B=1), P(C=1|B=1)
])

# ====================================================================
# 2. Joint Probability Calculation (Factoring Rule)
# ====================================================================

# Goal: Compute P(A=1, B=0, C=1)

# Factoring Rule: P(A, B, C) = P(A) * P(B | A) * P(C | B)

# Define the state indices: A=1 (index 1), B=0 (index 0), C=1 (index 1)
A_idx = 1
B_idx = 0
C_idx = 1

# 1. Term P(A=1)
Term_A = P_A[A_idx]

# 2. Term P(B=0 | A=1)
# B's probability (0) given A's state (1)
Term_B_given_A = P_B_given_A[A_idx, B_idx]

# 3. Term P(C=1 | B=0)
# C's probability (1) given B's state (0)
Term_C_given_B = P_C_given_B[B_idx, C_idx]

# Total Joint Probability
P_joint = Term_A * Term_B_given_A * Term_C_given_B

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Joint Probability Calculation using Bayesian Network ---")
print(f"Network Structure: A \u2192 B \u2192 C")
print(f"Factored Joint Probability: P(A, B, C) = P(A) * P(B|A) * P(C|B)")
print("---------------------------------------------------------------")
print(f"Target State: P(A={A_idx}, B={B_idx}, C={C_idx})")
print(f"Term 1: P(A=1) = {Term_A:.2f}")
print(f"Term 2: P(B=0 | A=1) = {Term_B_given_A:.2f}")
print(f"Term 3: P(C=1 | B=0) = {Term_C_given_B:.2f}")

print(f"\nFinal Joint Probability P(1, 0, 1): {P_joint:.4f}")

print("\nConclusion: The Bayesian Network framework allows a complex joint distribution to be computed efficiently by factoring it into a product of local conditional probabilities defined by the graph's topology.")
