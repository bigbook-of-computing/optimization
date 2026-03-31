# Source: Optimization/chapter-8/codebook.md -- Block 3

import numpy as np
import pandas as pd

# ====================================================================
# 1. Setup Parameters and J-Matrix Construction
# ====================================================================

NODES = ['A', 'B', 'C', 'D']
N = len(NODES)
J_VAL = -1.0 # Antiferromagnetic coupling (negative J)
H_VAL = 0.0  # No external field

# Initialize the 4x4 Ising Matrix J
J = np.zeros((N, N))

# Nearest-Neighbor Coupling (A-B, B-C, C-D)
# J[i, j] must be symmetric J[i, j] = J[j, i]

# A-B (Index 0-1)
J[0, 1] = J_VAL
J[1, 0] = J_VAL

# B-C (Index 1-2)
J[1, 2] = J_VAL
J[2, 1] = J_VAL

# C-D (Index 2-3)
J[2, 3] = J_VAL
J[3, 2] = J_VAL

# Linear term h is zero, so we only need the J matrix.

# The resulting Ising Matrix J
df_ising = pd.DataFrame(J, index=NODES, columns=NODES)

print("--- Ising Matrix J for 1D Antiferromagnetic Chain ---")
print(f"Coupling J = {J_VAL} (Negative, favors opposite spins)")
print(df_ising.to_markdown(floatfmt=".1f"))

# ====================================================================
# 2. Verification of Ground State Energy
# ====================================================================

# The optimal state (ground state) is s* = [+1, -1, +1, -1]
s_opt = np.array([1, -1, 1, -1])

# The ground state energy E = - sum J_ij * s_i * s_j
# Energy of the ground state = -(J_AB*sA*sB + J_BC*sB*sC + J_CD*sC*sD)
# E = -[(-1)*(1)*(-1) + (-1)*(-1)*(1) + (-1)*(1)*(-1)] = -[1 + 1 + 1] = -3
# We calculate the full matrix energy: E = -s^T J s
# Since J is symmetric, the matrix product is H = -0.5 * s^T J s 
# We calculate the full Hamiltonian H = -sum J_ij s_i s_j (the first term)

def calculate_hamiltonian_energy(s, J_matrix, h_vector=None):
    if h_vector is None:
        h_vector = np.zeros(len(s))
    
    # E = -s^T J s - h^T s (Note: The quadratic term is often defined as 0.5 * s^T J s in some contexts)
    # Using the standard physics definition: E = -sum J_ij s_i s_j - sum h_i s_i
    E_quad = -0.5 * s @ J_matrix @ s # Use 0.5 because J is counted twice in the matrix product
    E_lin = -np.dot(h_vector, s)
    return E_quad + E_lin

energy_ground = calculate_hamiltonian_energy(s_opt, J)

print("\nVerification (Ground State s* = [+1, -1, +1, -1])")
print(f"Calculated Ground State Energy E*: {energy_ground:.1f} (Should be -3.0)")
print("\nInterpretation: The matrix J successfully encodes the Antiferromagnetic couplings, driving the system to a ground state where the spins alternate.")
