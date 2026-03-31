# Source: Optimization/chapter-19/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Matrices (L=8)
# ====================================================================

L = 8 # Sequence/Node length

# --- Matrix 1: GNN Adjacency (Local/Sparse) ---
# 1D Chain: Only nearest neighbors (i \pm 1) are coupled
A_GNN = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        # Local coupling: i only interacts with i-1, i, i+1
        if abs(i - j) <= 1:
            A_GNN[i, j] = 1.0
    # Normalize the rows for GNN-like aggregation
    A_GNN[i, :] /= np.sum(A_GNN[i, :])

# --- Matrix 2: Transformer Adjacency (Global/Dense) ---
# Attention matrix is conceptually dense (all-to-all interaction)
A_TRANS = np.full((L, L), 1/L)

# ====================================================================
# 2. Visualization
# ====================================================================

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot 1: GNN Adjacency Matrix (Sparse)
im1 = axs[0].imshow(A_GNN, cmap='Greens', interpolation='none', vmin=0, vmax=1)
axs[0].set_title('GNN Adjacency $A_{\\text{GNN}}$ (Local/Sparse)')
axs[0].set_xlabel('Node j')
axs[0].set_ylabel('Node i')

# Plot 2: Transformer Adjacency (Dense)
im2 = axs[1].imshow(A_TRANS, cmap='Reds', interpolation='none', vmin=0, vmax=1)
axs[1].set_title('Transformer Attention $A_{\\text{Trans}}$ (Global/Dense)')
axs[1].set_xlabel('Element j')
axs[1].set_ylabel('Element i')

fig.suptitle(r'Sparsity Comparison: Local vs. Global Coupling', fontsize=14)
plt.tight_layout()
plt.savefig('Optimization/RESEARCH/docs/chapters/chapter-19/codes/ch19_sparsity_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Analysis Summary ---
A_GNN_sparsity = np.sum(A_GNN == 0) / (L*L)
A_TRANS_sparsity = np.sum(A_TRANS == 0) / (L*L)

print("\n--- Sparsity Analysis Summary ---")
print(f"GNN Matrix Sparsity: {A_GNN_sparsity:.2%} (Fixed Local Interaction)")
print(f"Transformer Matrix Sparsity: {A_TRANS_sparsity:.2%} (All-to-All Global Interaction)")

print("\nConclusion: The GNN matrix is visually sparse and banded, reflecting the explicit constraint that distant nodes cannot directly exchange messages. The Transformer matrix is dense, confirming its capacity for **global, all-to-all coupling**. This structural difference is the foundation of the Transformer's ability to model non-local correlations.")
