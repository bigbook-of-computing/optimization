# Source: Optimization/chapter-11/codebook.md -- Block 2

import numpy as np

# ====================================================================
# 1. Setup Network and Initial Data
# ====================================================================

# Network: A - B - C (Node B is calculating the message to C)
# Variables are binary: x_A, x_B, x_C \in {0, 1}

# --- Node B's Local Evidence (Factor \psi_{B,C}) ---
# This is the CPT-like factor \psi(x_B, x_C) or the edge potential
# We simplify by using an edge potential that favors x_B == x_C
# Rows (x_B), Columns (x_C)
FACTOR_B_C = np.array([
    [0.9, 0.1],  # x_B=0 favors x_C=0 (90%)
    [0.1, 0.9]   # x_B=1 favors x_C=1 (90%)
])

# --- Incoming Message to B from A (\mu_{A \to B}) ---
# This message is B's current belief about A's state
# P(x_B=0), P(x_B=1) - Uniform prior for the next iteration
MU_A_TO_B = np.array([0.5, 0.5]) 

# ====================================================================
# 2. Belief Propagation Update
# ====================================================================

# Goal: Calculate the outgoing message from B to C: \mu_{B \to C}(x_C)
# Message formula: \mu_{B \to C}(x_C) \propto \sum_{x_B} \psi(x_B, x_C) * \mu_{A \to B}(x_B)

# The outgoing message \mu_{B \to C} will be a vector of size 2 (for x_C=0 and x_C=1)
MU_B_TO_C = np.zeros(2)

# Loop over the target variable x_C (index 0 and 1)
for x_C in range(2):
    # The sum is over x_B (index 0 and 1)
    sum_term = 0.0
    for x_B in range(2):
        # 1. Local Factor: \psi(x_B, x_C)
        factor_term = FACTOR_B_C[x_B, x_C]
        
        # 2. Product of Incoming Messages: \mu_{A \to B}(x_B)
        # Note: B only has one other neighbor (A)
        incoming_message = MU_A_TO_B[x_B]
        
        sum_term += factor_term * incoming_message
        
    MU_B_TO_C[x_C] = sum_term

# Normalize the final message (since it's only proportional)
MU_B_TO_C /= np.sum(MU_B_TO_C)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Belief Propagation Message Calculation (\u03bc_{B \u2192 C}) ---")
print(f"Incoming Message from A (\u03bc_{A \u2192 B}): {np.round(MU_A_TO_B, 3)}")
print("---------------------------------------------------------------")
print("Factor \u03c8(x_B, x_C): Favors x_B = x_C")

print(f"\nOutgoing Message \u03bc_{B \u2192 C}: {np.round(MU_B_TO_C, 3)}")
print(f"P(x_C=0): {MU_B_TO_C[0]:.3f}")
print(f"P(x_C=1): {MU_B_TO_C[1]:.3f}")

print("\nConclusion: Node B successfully processed its local evidence (\u03c8_{B,C}) and the incoming message (\u03bc_{A \u2192 B}) to compute a new outgoing message (\u03bc_{B \u2192 C}). The message is nearly uniform (0.5, 0.5) because the incoming message from A was uniform, demonstrating that global inference is achieved by iteratively passing and combining local beliefs.")
