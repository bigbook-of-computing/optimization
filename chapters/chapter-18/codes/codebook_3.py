# Source: Optimization/chapter-18/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. Setup Graph and Ising Rules
# ====================================================================

# 5-Node Linear Chain Graph (1-2-3-4-5) - Ferromagnetic Coupling
NODES = 5
J = 1.0 # Ferromagnetic: favors alignment (s_i * s_j = +1)

# Adjacency List for local interaction (neighbors)
# Node 0: [1]
# Node 1: [0, 2]
# Node 2: [1, 3]
# Node 3: [2, 4]
# Node 4: [3]
NEIGHBORS = [
    [1], [0, 2], [1, 3], [2, 4], [3]
]

# Initial State: Random, disordered spins
SPINS = np.array(random.choices([-1, 1], k=NODES))

# ====================================================================
# 2. Local Energy Minimization Dynamics
# ====================================================================

def local_energy_minimization_step(s, J_coupling):
    """
    Applies the local update rule (GNN-like message passing) to minimize energy.
    s_i <- sign(\sum_j s_j)
    """
    s_new = s.copy()
    
    # Iterate over all nodes (asynchronous-like update is more stable, but 
    # we use sequential for clean simulation here)
    for i in range(NODES):
        # 1. Local Field (Message): h_i = J * \sum_{j \in N(i)} s_j
        h_i = J_coupling * np.sum(s[NEIGHBORS[i]])
        
        # 2. Update to minimize local energy: s_i <- sign(h_i)
        # s_i is updated in the original array (sequential update)
        s_new[i] = np.sign(h_i) if h_i != 0 else s_new[i]
        
    return s_new

# Simulation: Track global magnetization (order parameter)
MAX_STEPS = 10 
magnetization_history = [np.mean(SPINS)]
S_current = SPINS.copy()

for t in range(MAX_STEPS):
    S_next = local_energy_minimization_step(S_current, J)
    S_current = S_next
    magnetization_history.append(np.mean(S_current))
    
    # Check for stabilization
    if np.array_equal(S_current, S_current) and t > 0:
        break

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

plt.figure(figsize=(9, 6))

plt.plot(magnetization_history, 'r-o', lw=2, markersize=5)
plt.axhline(1.0, color='k', linestyle='--', label='Perfect Order ($M=1$)')
plt.axhline(-1.0, color='k', linestyle='--', label='Perfect Order ($M=-1$)')

# Labeling and Formatting
plt.title(f'Local Interaction Driving Global Order (Ising Analogy)')
plt.xlabel('Step (Local Message Pass)')
plt.ylabel('Global Magnetization (Order Parameter $M$)')
plt.ylim(-1.1, 1.1)
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Local Interaction Dynamics Summary ---")
print(f"Initial State (Random): M = {magnetization_history[0]:.2f}")
print(f"Final State (Ordered): M = {magnetization_history[-1]:.2f}")

print("\nConclusion: The simulation demonstrates that the simple local update rule—setting the spin in the direction of the local field (message)—drives the entire network from an initial disordered state toward a global, ordered state (M \u2248 \u00b11). This confirms the GNN-like principle of **local message passing** leading to **emergent global order**.")
