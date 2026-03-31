# Source: Optimization/chapter-8/codebook.md -- Block 4

import numpy as np

# ====================================================================
# 1. Setup Parameters and Penalty Function
# ====================================================================

N_VARS = 5  # Number of binary variables (x1 to x5)
LAMBDA = 100.0 # Large penalty factor

def one_hot_penalty(x, lambda_val=LAMBDA):
    """
    Calculates the penalty for violating the constraint: sum(x_i) = 1.
    E_penalty = lambda * (sum(x_i) - 1)^2
    """
    x_sum = np.sum(x)
    penalty = lambda_val * (x_sum - 1)**2
    return penalty

# ====================================================================
# 2. Test Solutions
# ====================================================================

# Test Case A: Valid Solution (One element selected)
x_A = np.array([1, 0, 0, 0, 0])
E_A = one_hot_penalty(x_A)

# Test Case B: Violation (Two elements selected)
x_B = np.array([1, 1, 0, 0, 0])
E_B = one_hot_penalty(x_B)

# Test Case C: Violation (Zero elements selected)
x_C = np.array([0, 0, 0, 0, 0])
E_C = one_hot_penalty(x_C)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- One-Hot Constraint Penalty Test ---")
print(f"Constraint: Exactly one variable must be selected (\u2211x_i = 1)")
print(f"Penalty Factor (\u03bb): {LAMBDA:.0f}")

print("\nTest Case A: x = [1, 0, 0, 0, 0] (\u2211x_i = 1)")
print(f"  Penalty E = {E_A:.1f} (Correctly Zero)")

print("\nTest Case B: x = [1, 1, 0, 0, 0] (\u2211x_i = 2)")
print(f"  Penalty E = {E_B:.1f} (High Penalty)")

print("\nTest Case C: x = [0, 0, 0, 0, 0] (\u2211x_i = 0)")
print(f"  Penalty E = {E_C:.1f} (High Penalty)")

print("\nConclusion: The penalty term correctly evaluates to zero for the valid solution (Case A) and imposes a large, positive cost for violating the constraint (Cases B and C). This ensures that any solver minimizing the total energy will prioritize satisfying the constraint over optimizing the original cost function, thus encoding the problem correctly for QUBO/Ising solvers.")
