# Source: Optimization/chapter-6/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. System Functions (1D Double-Well)
# ====================================================================

# Potential: E(x) = x^4 - 2x^2 + 1 (Minima at x = +/- 1)
def E(x):
    return x**4 - 2*x**2 + 1

# Metropolis Update (Local Dynamics for a single replica)
def metropolis_step(x, beta, delta_size=0.5):
    """Standard Metropolis step for one replica at fixed beta."""
    x_trial = x + random.uniform(-delta_size, delta_size)
    dE = E(x_trial) - E(x)
    
    if random.random() < np.exp(-beta * dE):
        return x_trial
    else:
        return x

# Parallel Tempering Swap Acceptance
def calculate_swap_prob(beta_i, beta_j, E_i, E_j):
    """Calculates the acceptance probability for swapping configurations."""
    # P_swap = min(1, exp( (beta_i - beta_j) * (E_j - E_i) ))
    d_beta = beta_i - beta_j
    dE = E_j - E_i
    
    # We assume beta_i > beta_j (colder replica i, hotter replica j)
    # The exponential term is positive if the cold system (i) gets a low-energy state (E_j < E_i)
    # or if the hot system (j) gets a high-energy state (E_j > E_i), but dE < 0
    return np.exp(d_beta * dE)

# ====================================================================
# 2. Parallel Tempering Simulation
# ====================================================================

# --- Simulation Parameters ---
TOTAL_STEPS = 50000
METROPOLIS_STEPS_PER_SWAP = 10 # Perform 10 local steps before attempting a swap
DELTA_SIZE = 0.5

# Temperature Ladder (Control Parameter)
# The spacing must be chosen carefully to ensure high swap acceptance rates.
BETAS = np.array([0.5, 1.0, 2.0, 5.0]) 
N_REPLICAS = len(BETAS)

# --- Initialization ---
# Start all replicas at the same cold, trapped location
X_REPLICAS = np.full(N_REPLICAS, 1.0) # Start all in the x=+1 well

# Store trajectory of the Coldest Replica (Index 3, Beta=5.0)
COLDEST_REPLICA_INDEX = N_REPLICAS - 1
X_coldest_traj = []

print(f"Starting Parallel Tempering Simulation with {N_REPLICAS} replicas...")

for t in range(TOTAL_STEPS):
    
    # 1. Local Metropolis Update (Energy Minimization)
    for i, beta in enumerate(BETAS):
        for _ in range(METROPOLIS_STEPS_PER_SWAP):
            X_REPLICAS[i] = metropolis_step(X_REPLICAS[i], beta, DELTA_SIZE)
            
    # 2. Replica Exchange (Swapping)
    # Iterate over neighboring pairs (e.g., [3,2], [2,1], [1,0])
    for i in range(N_REPLICAS - 1, 0, -1):
        # Replica i is Colder (higher beta), Replica i-1 is Hotter (lower beta)
        beta_i, beta_j = BETAS[i], BETAS[i-1]
        X_i, X_j = X_REPLICAS[i], X_REPLICAS[i-1]
        
        # Calculate energies
        E_i, E_j = E(X_i), E(X_j)
        
        # Calculate swap acceptance probability
        P_swap = calculate_swap_prob(beta_i, beta_j, E_i, E_j)
        
        if random.random() < P_swap:
            # Execute the swap (exchange the positions/configurations)
            X_REPLICAS[i], X_REPLICAS[i-1] = X_REPLICAS[i-1], X_REPLICAS[i]
            
    # Record the current state of the coldest replica
    X_coldest_traj.append(X_REPLICAS[COLDEST_REPLICA_INDEX])

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

X_coldest_traj = np.array(X_coldest_traj)
time_points = np.arange(TOTAL_STEPS)

# Check mixing efficiency
percent_in_well_neg = np.mean(X_coldest_traj < -0.5)
percent_in_well_pos = np.mean(X_coldest_traj > 0.5)

plt.figure(figsize=(10, 5))
plt.plot(time_points, X_coldest_traj, lw=0.7, color='darkred')

# Highlight the two minima for context
plt.axhline(1, color='gray', linestyle=':', alpha=0.7)
plt.axhline(-1, color='gray', linestyle=':', alpha=0.7)

plt.title(f'Parallel Tempering Trajectory of Coldest Replica ($\\beta={BETAS[-1]:.1f}$)')
plt.xlabel('Simulation Step')
plt.ylabel('Position $x$')
plt.grid(True)
plt.show()

# Final Analysis
print("\n--- Parallel Tempering Analysis (1D Double-Well) ---")
print(f"Coldest Replica Beta: \\beta = {BETAS[-1]:.1f}")
print(f"Time in Negative Well (x < -0.5): {percent_in_well_neg:.2%}")
print(f"Time in Positive Well (x > 0.5): {percent_in_well_pos:.2%}")

print("\nConclusion: The trajectory of the coldest replica exhibits frequent, large jumps between the two stable wells ($x=\\pm 1$). This global exploration, which is impossible for a single low-temperature chain, confirms that Parallel Tempering successfully uses the higher-temperature replicas to overcome the energy barrier and achieves rapid **mixing** across the multimodal distribution.")
