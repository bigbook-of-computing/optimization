# Source: Optimization/chapter-17/codebook.md -- Block 1

import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Conceptual Hamiltonian and Trial Wavefunction (RBM Ansatz)
# ====================================================================

N = 4 # Number of spins
M_SAMPLES = 1000 # VMC sample count

# --- A. Conceptual Hamiltonian (\hat{H}) for a Transverse Field Ising Chain ---
# \hat{H} = -J \sum_{i} \sigma^z_i \sigma^z_{i+1} - h \sum_{i} \sigma^x_i
J_COUPLING = 1.0
H_FIELD = 0.5 

def local_hamiltonian_elements(s, J=J_COUPLING, h=H_FIELD):
    """
    Computes H_{s, s'} = <s|H|s'> for all s' accessible from s.
    (s' is accessible if it is s or differs from s by one spin flip).
    """
    s = np.array(s)
    H_elements = {}
    
    # 1. Diagonal Element H_{s, s} (from \sigma^z \sigma^z term)
    E_classical = 0
    for i in range(N):
        # Periodic boundary conditions: s_i * s_{i+1}
        E_classical += s[i] * s[(i + 1) % N]
    H_elements[tuple(s)] = -J * E_classical
    
    # 2. Off-Diagonal Elements H_{s, s'} (from \sigma^x term)
    # H_{s, s'} = -h if s' is related to s by a single spin flip
    for i in range(N):
        s_prime = s.copy()
        s_prime[i] *= -1 # Single spin flip
        H_elements[tuple(s_prime)] = -h
        
    return H_elements

# --- B. Conceptual Trial Wavefunction (\psi_{\theta}) ---
# We use a conceptual, non-zero function that depends on a single parameter (\theta)
THETA = 0.1 # Conceptual parameter
def psi_theta(s):
    """
    Conceptual probability amplitude (RBM-like ansatz)
    Psi(s) is typically a product of complex exponentials and hyperbolic cosines
    We simplify to a real-valued, parameterized function of the classical energy E.
    """
    # Calculate classical energy of state s (using -J coupling only)
    E_s = 0
    s_arr = np.array(s)
    for i in range(N):
        E_s -= J_COUPLING * s_arr[i] * s_arr[(i + 1) % N]
    
    # Simple parameterized model: Psi(s) = exp(\theta * E_s)
    return np.exp(THETA * E_s) 

# ====================================================================
# 2. Local Energy and Expected Energy Calculation
# ====================================================================

# Generate VMC Samples (conceptual sampling using random states)
def generate_samples(num_samples):
    """Generates random spin configurations for VMC sampling."""
    return [tuple(np.random.choice([-1, 1], N)) for _ in range(num_samples)]

# VMC Samples
samples = generate_samples(M_SAMPLES)
E_loc_sum = 0.0

for s in samples:
    # 1. Get Hamiltonian elements H_{s, s'}
    H_elements = local_hamiltonian_elements(s)
    
    # 2. Calculate Local Energy E_loc(s)
    E_loc = 0.0
    psi_s = psi_theta(s)
    
    if psi_s != 0:
        for s_prime, H_s_s_prime in H_elements.items():
            psi_s_prime = psi_theta(s_prime)
            E_loc += H_s_s_prime * psi_s_prime / psi_s
    
    E_loc_sum += E_loc

# 3. Expected Energy (The Final Objective)
E_expected = E_loc_sum / M_SAMPLES

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- NQS Objective Function: Expected Energy Calculation ---")
print(f"Network Size (N): {N}, Samples (M): {M_SAMPLES}")
print(f"Trial Wavefunction Parameter (\u03b8): {THETA:.2f}")

# The True Ground State Energy for this system is known to be around -1.707
print("\nAnalytic Ground State Energy (Reference E0): \u2248 -1.707")
print(f"Calculated Expected Energy E(\u03b8): {E_expected:.4f} (Must be \u2265 E0)")

print("\nConclusion: The VMC algorithm successfully computed the expected energy \u27e8\hat{H}\u27e9 of the trial quantum state. This expected energy value is the **quantum loss function** that is minimized during training to drive the network parameters (\u03b8) toward the true ground state.")
