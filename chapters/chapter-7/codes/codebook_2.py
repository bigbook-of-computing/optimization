# Source: Optimization/chapter-7/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. Setup Parameters and PBC Functions
# ====================================================================

# --- System Parameters ---
N_PARTICLES = 4
L_BOX = 10.0
M = 1.0
DT = 0.005 
STEPS = 500

# --- Reference/Conceptual Functions ---
def minimum_image(dr, L):
    """Calculates the minimum image distance vector component."""
    # dr = ri - rj. This implements dr - L * round(dr/L)
    return dr - L * np.round(dr / L)

def wrap_position(r, L):
    """Wraps position back into the primary simulation box [0, L]."""
    return r % L

def force_conceptual(r_i, r_j, L, cutoff=1.0, epsilon=1.0):
    """
    Conceptual short-range repulsive force (Lennard-Jones-like, F ~ 1/r^7 scaling).
    """
    # 1. Calculate the minimum image distance vector
    dr = minimum_image(r_i - r_j, L)
    r_sq = np.sum(dr**2)
    
    if r_sq > cutoff**2 or r_sq == 0:
        return np.zeros_like(r_i) # No interaction or self-interaction
    
    r = np.sqrt(r_sq)
    
    # 2. Conceptual Repulsive Force (Strong inverse power law)
    r_inv = 1.0 / r
    F_mag_factor = 48 * epsilon * r_inv**13 
    
    # F_vector = F_mag_factor * (dr / r)
    F_vec = F_mag_factor * (dr / r)
    
    return F_vec

def calculate_total_force(positions, L):
    """Calculates the total force vector for all particles (O(N^2) here)."""
    N = len(positions)
    total_forces = np.zeros_like(positions)
    
    for i in range(N):
        for j in range(i + 1, N):
            # MIC is embedded in force_conceptual call
            F_ij = force_conceptual(positions[i], positions[j], L)
            total_forces[i] += F_ij
            total_forces[j] -= F_ij # Newton's third law
            
    return total_forces

# ====================================================================
# 2. Initialization and MD Loop
# ====================================================================

# Initial state
R_init = np.random.rand(N_PARTICLES, 2) * L_BOX
V_init = np.random.rand(N_PARTICLES, 2) * 2.0 - 1.0 # Initial velocity

# Storage
R_history = np.zeros((STEPS, N_PARTICLES, 2))
R_history[0] = R_init.copy()

# Setup initial state
R = R_init.copy()
V = V_init.copy()
F_current = calculate_total_force(R, L_BOX)

for step in range(1, STEPS):
    # Get current acceleration
    A_current = F_current / M
    
    # 1. Position Update (Verlet)
    R_new_unwrapped = R + V * DT + 0.5 * A_current * DT**2
    
    # Apply PBC: Wrap positions back into [0, L]
    R_new = wrap_position(R_new_unwrapped, L_BOX)
    
    # 2. New Force Evaluation (using wrapped positions for the interaction)
    F_new = calculate_total_force(R_new, L_BOX)
    A_new = F_new / M
    
    # 3. Velocity Update
    V_new = V + 0.5 * (A_current + A_new) * DT
    
    # Bookkeeping: Advance state and force
    R, V = R_new, V_new
    F_current = F_new
    R_history[step] = R_new.copy()

# ====================================================================
# 3. Visualization
# ====================================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Plot initial and final state
ax.plot(R_history[0, :, 0], R_history[0, :, 1], 'o', markersize=10, 
        color='blue', alpha=0.5, label='Initial Positions ($t=0$)')
ax.plot(R_history[-1, :, 0], R_history[-1, :, 1], 'x', markersize=10, 
        color='red', label=f'Final Positions ($t={STEPS*DT:.2f}$)')

# Draw the simulation box boundary
ax.plot([0, L_BOX, L_BOX, 0, 0], [0, 0, L_BOX, L_BOX, 0], 'k--', lw=1, label='Simulation Box')

# Labeling and Formatting
ax.set_title(f'2D Molecular Dynamics with Periodic Boundaries (N={N_PARTICLES})')
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_xlim(-0.5, L_BOX + 0.5)
ax.set_ylim(-0.5, L_BOX + 0.5)
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Verification ---
R_new_unwrapped = R + V * DT + 0.5 * F_current / M * DT**2 # Recompute the last unwrapped step
R_new = wrap_position(R_new_unwrapped, L_BOX)
total_wrapped_movement = np.sum(np.abs(R_new_unwrapped - R_new))

print("\n--- Boundary Condition Verification ---")
print(f"Box Side Length (L): {L_BOX:.1f}")
print(f"Total Conceptual Movement Wrapped by PBCs: {total_wrapped_movement:.2f} (Should be > 0)")
print(f"Final positions are all within [0, L]: {np.all((R_history[-1] >= 0) & (R_history[-1] <= L_BOX))}")
