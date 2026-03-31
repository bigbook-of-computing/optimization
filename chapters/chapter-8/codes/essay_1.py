# Source: Optimization/chapter-8/essay.md -- Block 1

import numpy as np
import itertools
from typing import Tuple, List, Optional

# --- 1. Define the QUBO Problem ---

# Random QUBO matrix (N=6 variables)
np.random.seed(0)
N = 6
Q = np.random.randn(N, N)
Q = (Q + Q.T) / 2  # Symmetrize Q: Ensures the energy function is well-defined.
# The diagonal of Q implicitly contains the linear terms (Section 8.3).

# --- 2. Define the Energy Function ---
def energy(x: np.ndarray) -> float:
    """Computes the QUBO energy E(x) = x^T * Q * x"""
    # Note: In QUBO, the matrix Q often includes both linear and quadratic terms
    # for full conversion from the Ising model.
    return x.T @ Q @ x

# --- 3. Brute-Force Solver ---

best_x: Optional[np.ndarray] = None
best_E: float = np.inf
configurations: List[Tuple[int, ...]] = list(itertools.product([0, 1], repeat=N))

print(f"Total variables (N): {N}")
print(f"Total configurations to check: 2^{N} = {len(configurations)}")
print("-" * 30)

# Iterate through all 2^N possible binary configurations
for bits in configurations:
    x = np.array(bits, dtype=np.float64)
    E = energy(x)
    
    if E < best_E:
        best_E, best_x = E, x.copy()
        
# --- 4. Output ---
print("Best state (Configuration Vector):", best_x)
print("Minimum energy (Ground State):", best_E)
