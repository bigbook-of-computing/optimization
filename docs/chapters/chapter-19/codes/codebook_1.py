# Source: Optimization/chapter-19/codebook.md -- Block 1

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Conceptual Matrices (Q, K, V)
# ====================================================================

# Sequence Length L=4 (e.g., 4 tokens/particles)
L = 4
D_MODEL = 2 # Embedding dimension
D_K = 2     # Key dimension (same as D_MODEL for simplicity)

# Conceptual Query, Key, and Value Matrices (L x D_MODEL)
Q = np.array([
    [1.0, 0.0],  # Q_1 (Focuses on F1)
    [0.0, 1.0],  # Q_2 (Focuses on F2)
    [0.8, 0.2],
    [0.1, 0.9]
])

K = np.array([
    [1.0, 0.1],  # K_1 (F1 high)
    [-1.0, 1.0], # K_2 (F1 low, F2 high)
    [0.9, 0.2],
    [0.0, 0.9]
])

V = np.array([
    [5.0, 5.0],  # V_1 (High value)
    [1.0, 10.0], # V_2 (Very high value in F2)
    [3.0, 3.0],
    [10.0, 1.0]
])

# ====================================================================
# 2. Self-Attention Calculation
# ====================================================================

# 1. Calculate Raw Scores: S = Q K^T
S_raw = Q @ K.T

# 2. Scale: S / sqrt(d_k)
S_scaled = S_raw / np.sqrt(D_K)

# 3. Softmax Normalization: A_attn = Softmax(S_scaled)
def softmax_numpy(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

A_attn = softmax_numpy(S_scaled)

# 4. Final Output: V_out = A_attn V
V_out = A_attn @ V

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

df_v_out = pd.DataFrame(V_out, index=[f'Output {i}' for i in range(L)],
                        columns=['Feature 1', 'Feature 2'])

print("--- Self-Attention Mechanism (Dynamic Output) ---")

print("\n1. Attention Matrix (\u03b1_{ij}): The Dynamic Coupling Kernel")
print(pd.DataFrame(A_attn, index=[f'Query {i}' for i in range(L)],
                   columns=[f'Key {j}' for j in range(L)]).to_string())

print("\n2. Output V_out: Weighted Sum of V")
print(df_v_out.to_string())

print("\nConclusion: The output vector for each element (row) is a dynamically calculated blend of all Value vectors (\u03bb_ij V_j). The Softmax matrix \u03bb_attn successfully encodes the relevance, creating a global, dynamic coupling that is the foundation of the Transformer's non-local information propagation.")
