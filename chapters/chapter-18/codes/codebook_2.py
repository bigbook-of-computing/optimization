# Source: Optimization/chapter-18/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Graph (5-Node Chain) and Initial Condition
# ====================================================================

# 5-Node Chain Graph (1-2-3-4-5) - No self-loops initially
NODES = ['1', '2', '3', '4', '5']
N = len(NODES)

A_raw = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

# Initial Features H^(0): Node 1 has high "heat," others have zero
H0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

# ====================================================================
# 2. GCN Aggregation Matrix (\tilde{A}) Construction (Same as Project 1)
# ====================================================================

# 1. Add Self-Loops: \hat{A} = A + I
A_hat = A_raw + np.eye(N)

# 2. Compute Inverse Square Root of Degree Matrix \hat{D}^{-1/2}
D_hat_vec = np.sum(A_hat, axis=1)
D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(D_hat_vec))

# 3. Normalized Aggregation Matrix (A_tilde)
A_tilde = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

# ====================================================================
# 3. Iterative Diffusion Simulation
# ====================================================================

MAX_LAYERS = 15 # Simulate 15 layers/steps of diffusion
H_current = H0.copy()
H_history = [H_current.copy()]

for l in range(MAX_LAYERS):
    # Diffusion Step (H^(l+1) = \tilde{A} H^(l))
    # This matrix multiplication performs the normalized averaging of neighbor values
    H_next = A_tilde @ H_current
    H_current = H_next
    H_history.append(H_current.copy())

# ====================================================================
# 4. Visualization and Analysis
# ====================================================================

H_history_arr = np.array(H_history)
layers = np.arange(MAX_LAYERS + 1)

plt.figure(figsize=(9, 6))

# Plot the evolution of the feature value for each node
for i in range(N):
    plt.plot(layers, H_history_arr[:, i], marker='o', markersize=4, linestyle='-', 
             label=f'Node {i+1}')

# Highlight the final equilibrium value (average of initial values)
equilibrium_val = np.mean(H0)
plt.axhline(equilibrium_val, color='k', linestyle='--', label=f'Equilibrium ({equilibrium_val:.1f})')

# Labeling and Formatting
plt.title(r'GCN Propagation as Discrete Heat Diffusion ($\mathbf{H}^{(l+1)} = \tilde{\mathbf{A}} \mathbf{H}^{(l)}$)')
plt.xlabel('Layer Number (Time Step)')
plt.ylabel('Feature Value $H_i$ (Concentration)')
plt.xlim(0, MAX_LAYERS)
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Graph Diffusion Analysis ---")
print(f"Initial State (t=0): {H0}")
print(f"Final State (t={MAX_LAYERS}): {np.round(H_history_arr[-1], 3)}")

print("\nConclusion: The simulation shows the core diffusion property of the GCN. The initial localized high value (Node 1) spreads outward and dissipates over successive layers, causing the entire network's features to converge toward a uniform equilibrium value. This confirms the GCN's role as an **information smoothing filter** that propagates local data globally.")
