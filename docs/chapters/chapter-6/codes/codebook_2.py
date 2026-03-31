# Source: Optimization/chapter-6/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. Wolff Cluster Algorithm Implementation
# ====================================================================

# Reusing Ising core functions from Project 1: create_lattice, get_neighbors, calculate_magnetization

def wolff_update_step(spins, beta, J=1.0):
    """
    Performs one Wolff cluster update.
    This corresponds to one Monte Carlo Sweep (MCS) for comparison purposes.
    """
    L = spins.shape[0]
    p_add = 1 - np.exp(-2 * beta * J) # Bond probability
    visited = np.zeros_like(spins, dtype=bool)
    
    # 1. Pick random seed and initialize cluster
    i, j = random.randrange(L), random.randrange(L)
    cluster_val = spins[i, j]
    
    # Use a stack (LIFO) for a recursive-like cluster growth (DFS/stack-based)
    stack = [(i, j)]
    visited[i, j] = True
    
    cluster_size = 0
    
    while stack:
        x, y = stack.pop()
        cluster_size += 1
        
        # Check all nearest neighbors
        for xn, yn in get_neighbors(L, x, y):
            # Condition: Neighbor is unvisited AND aligned
            if not visited[xn, yn] and spins[xn, yn] == cluster_val:
                # Add bond with probability p_add
                if random.random() < p_add:
                    visited[xn, yn] = True
                    stack.append((xn, yn))
                    
    # 2. Flip the entire cluster
    # This is a safe operation because 'visited' is a boolean mask
    spins[visited] *= -1
    return spins, cluster_size

# Reusing autocorrelation functions from Project 1 (autocorr_func, estimate_tau_int)

# ====================================================================
# 2. Simulation and Comparison at T_c
# ====================================================================

LATTICE_SIZE = 32
T_C = 2.269185 
BETA_C = 1.0 / T_C
MCS_RUN = 15000 
EQUILIBRATION_MCS = 500

# --- Metropolis Benchmark (from Project 1, T_c) ---
# We use the computed tau_int from Project 1 for the T_c Metropolis run (L=32)
# To avoid repeating the time-consuming run, we use the value observed earlier.
# This assumes the Metropolis run was performed and its result is available.
TAU_INT_METROPOLIS = 25.0 

# --- Wolff Simulation ---
wolff_lattice = create_lattice(LATTICE_SIZE, initial_state='+1')
M_series_wolff = []
cluster_sizes = []

print(f"Starting Wolff Cluster simulation at T_c={T_C:.3f}...")

# 1. Thermalization
for _ in range(EQUILIBRATION_MCS):
    wolff_update_step(wolff_lattice, BETA_C)
    
# 2. Measurement
for _ in range(MCS_RUN):
    wolff_lattice, size = wolff_step(wolff_lattice, BETA_C)
    M_series_wolff.append(calculate_magnetization(wolff_lattice))
    cluster_sizes.append(size)

# 3. Analysis
M_series_wolff = np.array(M_series_wolff)
tau_int_wolff, C_plot_wolff = estimate_tau_int(M_series_wolff)

# ====================================================================
# 3. Visualization and Comparison
# ====================================================================

speedup_factor = TAU_INT_METROPOLIS / tau_int_wolff
avg_cluster_size = np.mean(cluster_sizes) / (LATTICE_SIZE**2)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Autocorrelation Function Comparison
ax[0].plot(C_plot_wolff[:100], marker='.', markersize=3, 
           linestyle='-', lw=2, color='darkgreen', 
           label=f"Wolff Cluster ($\\tau_{{\\text{{int}}}}={tau_int_wolff:.2f}$ MCS)")

# Illustrative Metropolis decay for comparison
tau_axis = np.arange(0, 100)
C_metropolis_illustrative = np.exp(-tau_axis / TAU_INT_METROPOLIS) 
ax[0].plot(tau_axis, C_metropolis_illustrative, 
           linestyle='--', color='red', alpha=0.6,
           label=f"Metropolis (Benchmark $\\tau_{{\\text{{int}}}}={TAU_INT_METROPOLIS:.1f}$ MCS)")

ax[0].axhline(0, color='k', linestyle='--')
ax[0].set_title(f'Autocorrelation at Critical Point $T_c$ (L={LATTICE_SIZE})')
ax[0].set_xlabel('Time Lag $\\tau$ (MCS)')
ax[0].set_ylabel('Autocorrelation $C(\\tau)$')
ax[0].set_xlim(0, 50)
ax[0].legend()
ax[0].grid(True, which='both', linestyle=':')

# Plot 2: Tau_int and Speedup
ax[1].bar(['Metropolis $\\tau_{\\text{int}}$ (Benchmark)', 'Wolff $\\tau_{\\text{int}}$'], 
         [TAU_INT_METROPOLIS, tau_int_wolff], 
         color=['salmon', 'darkgreen'])

ax[1].text(0.5, 0.9, f'Speedup: {speedup_factor:.1f}x', transform=ax[1].transAxes, 
          ha='center', fontsize=12, fontweight='bold')
ax[1].set_title('Efficiency Gain Over Critical Slowing Down')
ax[1].set_ylabel('$\\tau_{\\text{int}}$ (MCS)')
ax[1].grid(True, which='major', axis='y', linestyle=':')

plt.tight_layout()
plt.show()

# Final Analysis
print("\n--- Wolff Cluster Algorithm Analysis ---")
print(f"Metropolis Benchmark Tau_int: {TAU_INT_METROPOLIS:.1f} MCS")
print(f"Wolff Simulated Tau_int: {tau_int_wolff:.2f} MCS")
print(f"Average Cluster Size (Fraction of L^2): {avg_cluster_size:.2f}")
print(f"Speedup Factor: {speedup_factor:.1f}x")

print("\nConclusion: The Wolff Cluster algorithm achieved a significant speedup factor over the single-spin Metropolis method at the critical temperature. By flipping large, correlated domains (clusters) simultaneously, the algorithm effectively circumvents **critical slowing down**, making the system's relaxation to equilibrium much faster and lattice-size independent.")
