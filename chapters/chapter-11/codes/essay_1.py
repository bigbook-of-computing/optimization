# Source: Optimization/chapter-11/essay.md -- Block 1

import numpy as np

# --- 1. Define Pairwise Potentials (Couplings/Energy) ---
# Each matrix psi_XY(x, y) defines the unnormalized probability P(x,y).
# Indices: 0, 1 for the binary states (e.g., A=0, A=1)
# psi_AB[0, 0] = P(A=0, B=0) proportional to 3
# psi_AB[1, 0] = P(A=1, B=0) proportional to 1
psi_AB = np.array([[3,1],[1,3]])  # Coupling between A and B
psi_BC = np.array([[2,1],[1,2]])  # Coupling between B and C

# --- 2. Initialize Messages ---
# Messages start uniform (uninformative) or represent marginals of endpoints.
# The messages are 2-element vectors for the two binary states {0, 1}.
m_AtoB = np.ones(2)  # Message from A to B
m_CtoB = np.ones(2)  # Message from C to B

# --- 3. Iterative Message Passing (Inference) ---
# For a chain, the exact solution is found in one pass.
# We run multiple steps to demonstrate the convergence process.
for _ in range(10):
    # a) Compute Message from B to A (m_BtoA):
    # B receives influence from its *other* neighbor (C) and combines it with the A-B coupling.
    m_BtoA = psi_AB @ m_CtoB
    
    # b) Compute Message from B to C (m_BtoC):
    # B receives influence from its *other* neighbor (A) and combines it with the B-C coupling.
    m_BtoC = psi_BC @ m_AtoB
    
    # c) Update incoming messages for next iteration (Factor to Variable)
    # The message m_AtoB is updated using the *new* message m_BtoA and psi_AB.
    # Note: We use the transpose of psi_AB here for correct matrix multiplication logic.
    m_AtoB = psi_AB.T @ m_BtoA
    m_CtoB = psi_BC.T @ m_BtoC
    
    # Normalize messages to prevent numerical overflow (optional but good practice)
    m_AtoB /= np.sum(m_AtoB)
    m_CtoB /= np.sum(m_CtoB)

# --- 4. Compute Final Belief (Marginal Probability) ---
# The final belief for node B is proportional to the product of all messages received,
# multiplied by its own local potential (which is uniform here, so m_AtoB * m_CtoB).
belief_B = m_AtoB * m_CtoB
belief_B /= np.sum(belief_B)  # Final normalization to ensure sum(p)=1

print("Belief for node B (Final Marginal Probability):", belief_B)
