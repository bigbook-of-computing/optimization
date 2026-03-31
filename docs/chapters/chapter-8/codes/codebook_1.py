# Source: Optimization/chapter-8/codebook.md -- Block 1

import numpy as np
import pandas as pd

# ====================================================================
# 1. Transformation Functions
# ====================================================================

def qubo_to_ising(x):
    """
    Converts QUBO variable (0, 1) to Ising spin (-1, +1).
    s_i = 2*x_i - 1
    """
    return 2 * np.array(x) - 1

def ising_to_qubo(s):
    """
    Converts Ising spin (-1, +1) to QUBO variable (0, 1).
    x_i = (s_i + 1) / 2
    """
    return (np.array(s) + 1) / 2

# ====================================================================
# 2. Verification Test
# ====================================================================

# Test inputs
x_test = [0, 1]
s_test = [-1, 1]

# Forward and inverse transformations
s_forward = qubo_to_ising(x_test)
x_inverse = ising_to_qubo(s_test)
s_round_trip = qubo_to_ising(x_inverse)

# Create a DataFrame for verification
data = {
    'QUBO Input (x)': x_test,
    'Ising Output (s)': s_forward,
    'Ising Input (s)': s_test,
    'QUBO Output (x)': x_inverse,
    'Round Trip s': s_round_trip
}
df_verify = pd.DataFrame(data)

print("--- Ising <-> QUBO Transformation Verification ---")
print(df_verify.to_markdown(index=False))

print("\nConclusion: The transformations are successfully implemented. The QUBO domain (0, 1) maps correctly to the Ising domain (-1, +1), and the inverse transformation restores the original domain values.")
