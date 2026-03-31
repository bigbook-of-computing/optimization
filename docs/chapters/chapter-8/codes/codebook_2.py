# Source: Optimization/chapter-8/codebook.md -- Block 2

import numpy as np
import pandas as pd

# ====================================================================
# 1. Setup Graph and Problem Definition
# ====================================================================

# 4-Node Chain Graph (A-B-C-D) with Unit Weights (w_ij = 1)
NODES = ['A', 'B', 'C', 'D']
N = len(NODES)

# Adjacency matrix (A[i, j] = 1 if edge exists)
# Edges: (A, B), (B, C), (C, D)
ADJACENCY_MATRIX = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
])
# We use the adjacency matrix for summing over edges (i, j).

# ====================================================================
# 2. QUBO Matrix Construction
# ====================================================================

# Goal: Minimize the negative cut value: E = -C. 
# The Max Cut objective for x_i in {0, 1} is:
# C = sum_{<i, j>} w_ij * (x_i (1-x_j) + x_j (1-x_i))
# We minimize E = -C.

# The simplified QUBO cost function for Max Cut is generally:
# E = sum_{i} Q_{ii} x_i + sum_{i<j} Q_{ij} x_i x_j
# Where Q_{ij} = 2 * w_{ij} for the off-diagonals, and Q_{ii} = -sum_{j} w_{ij} for the diagonals.

Q = np.zeros((N, N))

for i in range(N):
    # Calculate the sum of weights connected to node i (degree of node i)
    degree_i = np.sum(ADJACENCY_MATRIX[i, :])
    
    # 1. Diagonal Terms (Q_ii * x_i): Q_ii = -degree_i
    Q[i, i] = -degree_i 
    
    for j in range(i + 1, N):
        if ADJACENCY_MATRIX[i, j] == 1:
            # 2. Off-Diagonal Terms (Q_ij * x_i * x_j): Q_ij = 2 * w_ij
            Q[i, j] = 2.0 * ADJACENCY_MATRIX[i, j]
            Q[j, i] = Q[i, j] # Ensure symmetry

# The resulting QUBO matrix Q
df_qubo = pd.DataFrame(Q, index=NODES, columns=NODES)

print("--- QUBO Matrix Q for Max Cut on A-B-C-D Chain ---")
print("Q_{ii} = Linear term (Node Bias); Q_{ij} = Quadratic term (Edge Coupling)")
print(df_qubo.to_markdown(floatfmt=".1f"))

# ====================================================================
# 3. Verification of Solution
# ====================================================================

# Ground Truth for Max Cut on a 4-node chain is 3 edges (e.g., [1, 0, 1, 0])
# We check the cost for the optimal solution x* = [1, 0, 1, 0]
x_opt = np.array([1, 0, 1, 0])
energy_opt = x_opt @ Q @ x_opt.T

print("\nVerification (Optimal Solution x* = [1, 0, 1, 0] | Max Cut = 3)")
print(f"Minimum Energy E = x* Q x* = {energy_opt:.1f}")

print("\nInterpretation: The matrix Q ensures that configurations corresponding to a large cut result in a very low (negative) energy, guiding the solver toward the optimal partitioning.")
