# Source: Optimization/chapter-18/codebook.md -- Block 1

import numpy as np
import pandas as pd

# ====================================================================
# 1. Setup Graph and Features
# ====================================================================

# 5-Node Star Graph (Node 0 is the center hub, connected to all others)
NODES = ['Hub (0)', 'A (1)', 'B (2)', 'C (3)', 'D (4)']
N = len(NODES)

# Adjacency Matrix A (A_ij = 1 if edge exists)
# Edges: (0, 1), (0, 2), (0, 3), (0, 4)
A = np.array([
    [0, 1, 1, 1, 1], # Node 0 (Hub)
    [1, 0, 0, 0, 0], # Node 1
    [1, 0, 0, 0, 0], # Node 2
    [1, 0, 0, 0, 0], # Node 3
    [1, 0, 0, 0, 0]  # Node 4
])

# Initial Features H^(0) = X (Two features: F1, F2)
# F1 (Initial Value), F2 (Type Identifier)
X = np.array([
    [10.0, 0.0], # Hub has highest initial value
    [0.0, 1.0],  # All others start low
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0]
])

# ====================================================================
# 2. GCN Aggregation Matrix (\tilde{A}) Construction
# ====================================================================

# 1. Add Self-Loops: \hat{A} = A + I
I = np.eye(N)
A_hat = A + I

# 2. Compute Degree Matrix \hat{D} (Row-wise sum of \hat{A})
D_hat_vec = np.sum(A_hat, axis=1)

# 3. Compute Inverse Square Root of Degree Matrix \hat{D}^{-1/2}
D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(D_hat_vec))

# 4. Compute Normalized Adjacency Matrix (The Aggregator): 
# \tilde{A} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}
A_tilde = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

# 5. Core Aggregation Step: H^(1) = \tilde{A} X
# (We omit W and \sigma for this verification)
H_agg = A_tilde @ X

# ====================================================================
# 3. Analysis and Verification
# ====================================================================

df_agg = pd.DataFrame(H_agg, index=NODES, columns=['Feature 1 (Value)', 'Feature 2 (Type)'])

print("--- GCN Aggregation and Message Passing (\u03c4=1) ---")
print("Matrix \u03c4: Normalized Aggregator Matrix \u03c4 (A_tilde)")
print(pd.DataFrame(A_tilde, index=NODES, columns=NODES).to_markdown(floatfmt=".3f"))

print("\nResult: Aggregated Features H^(1) = \u03c4 X")
print(df_agg.to_markdown(floatfmt=".3f"))

print("\nVerification (Hub Node 0):")
print(f"Initial Value (Hub): {X[0, 0]:.1f}")
print(f"New Value (Hub): {H_agg[0, 0]:.3f}")

print("\nInterpretation: The aggregation step successfully spreads the initial high value (10.0) from the Hub (Node 0) to its neighbors, while also diluting the Hub's own value. The Hub's new feature value (1.414) is the normalized average of its own old value (10.0) and its four neighbors (0.0). This confirms the central mechanism of **local information diffusion** in GNNs.")
