# Source: Optimization/chapter-11/codebook.md -- Block 1

import numpy as np
import pandas as pd

# ====================================================================
# 1. Setup Network and Conditional Probability Tables (CPTs)
# ====================================================================

# Dependency: A -> B -> C (A is root, C is leaf)
# Variables are binary: 0 (False) or 1 (True)

# P(A) - Prior for the root node
# Index [0] is P(A=0), Index [1] is P(A=1)
P_A = np.array([0.4, 0.6]) 

# P(B | A) - Conditional Probability Table (CPT)
# Rows: P(B | Parent)
# P_B_given_A[A_state, B_state]
P_B_given_A = np.array([
    [0.8, 0.2],  # P(B=0|A=0), P(B=1|A=0)
    [0.1, 0.9]   # P(B=0|A=1), P(B=1|A=1)
])

# P(C | B) - Conditional Probability Table (CPT)
# P_C_given_B[B_state, C_state]
P_C_given_B = np.array([
    [0.9, 0.1],  # P(C=0|B=0), P(C=1|B=0)
    [0.2, 0.8]   # P(C=0|B=1), P(C=1|B=1)
])

# ====================================================================
# 2. Joint Probability Calculation (Factoring Rule)
# ====================================================================

# Goal: Compute P(A=1, B=0, C=1)
# State Indices: A_idx=1, B_idx=0, C_idx=1

A_idx = 1
B_idx = 0
C_idx = 1

# 1. Term P(A=1)
Term_A = P_A[A_idx]

# 2. Term P(B=0 | A=1)
Term_B_given_A = P_B_given_A[A_idx, B_idx]

# 3. Term P(C=1 | B=0)
Term_C_given_B = P_C_given_B[B_idx, C_idx]

# Total Joint Probability P(A, B, C) = P(A) * P(B|A) * P(C|B)
P_joint = Term_A * Term_B_given_A * Term_C_given_B

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Joint Probability Calculation using Bayesian Network ---")
print(f"Network Structure: A \u2192 B \u2192 C")
print(f"Target State: P(A={A_idx}, B={B_idx}, C={C_idx})")
print("---------------------------------------------------------------")
print(f"Term 1: P(A=1) = {Term_A:.2f}")
print(f"Term 2: P(B=0 | A=1) = {Term_B_given_A:.2f}")
print(f"Term 3: P(C=1 | B=0) = {Term_C_given_B:.2f}")

print(f"\nFinal Joint Probability P(1, 0, 1): {P_joint:.4f}")

print("\nConclusion: The Bayesian Network framework allows the complex joint probability of the state (A=1, B=0, C=1) to be computed efficiently by factoring it into a product of local conditional probabilities defined by the graph's topology.")
