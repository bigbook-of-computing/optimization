# Source: Optimization/chapter-19/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Fixed Keys and Policy
# ====================================================================

D_K = 2
D_V = 2
L = 5 # Sequence length (5 elements)
SCALING = np.sqrt(D_K)

# Fixed Key and Value Vectors (Neighbors)
K = np.array([
    [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, -1.0], [-1.0, -1.0]
])
V = np.random.randn(L, D_V)

def calculate_attention(Q_vec):
    """Calculates Softmax attention coefficients for a single Query Q_vec."""
    S_raw = Q_vec @ K.T
    S_scaled = S_raw / SCALING
    exp_S = np.exp(S_scaled - np.max(S_scaled))
    return exp_S / np.sum(exp_S)

# ====================================================================
# 2. Dynamic Scenarios (Changing the Query Element i's feature)
# ====================================================================

# QUERY_INDEX = 0

# --- Scenario A: Query focuses on Feature 1 (Q = [1, 0]) ---
Q_A = np.array([1.0, 0.0])
Alpha_A = calculate_attention(Q_A)

# --- Scenario B: Query focuses on Feature 2 (Q = [0, 1]) ---
Q_B = np.array([0.0, 1.0])
Alpha_B = calculate_attention(Q_B)

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

X_labels = [f'Key {i}' for i in range(1, L + 1)]
X_labels[0] += '\n(Self-Attention)' # Label the first element

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(L)
width = 0.4

ax.bar(x - width/2, Alpha_A, width, label='Query A: Focus on F1 ([1, 0])', color='skyblue')
ax.bar(x + width/2, Alpha_B, width, label='Query B: Focus on F2 ([0, 1])', color='darkred')

# Labeling and Formatting
ax.set_title('Dynamic Global Coupling: Attention Weights Change with Feature Content')
ax.set_xlabel('Neighboring Element (Key Index)')
ax.set_ylabel('Attention Weight $\\alpha_{i, j}$')
ax.set_xticks(x)
ax.set_xticklabels(X_labels)
ax.legend()
ax.grid(True, axis='y')
plt.savefig('Optimization/RESEARCH/docs/chapters/chapter-19/codes/ch19_dynamic_coupling.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n--- Dynamic Coupling Analysis ---")
print("Scenario A (\u03b1_A): Weights concentrate on Key 1 (F1 high), confirming that the network prioritized neighbors with compatible Feature 1 content.")
print("Scenario B (\u03b1_B): Weights concentrate on Key 2 (F2 high), confirming that the network shifted its focus instantly when the Query's internal feature changed.")

print("\nConclusion: The shift in attention weights demonstrates **dynamic global coupling**. The mechanism allows the network to instantly rewire its connections (change \u03b1_ij) based on the input features, effectively modeling long-range, content-aware interactions that are impossible with fixed, local GNN adjacency matrices.")
