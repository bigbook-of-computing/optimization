# **Chapter 8: Combinatorial Optimization and QUBO () () (Codebook)**

## Project 1: The Core Ising-to-QUBO Mapping ($\mathcal{s_i \leftrightarrow 2x_i - 1}$)

---

### Definition: The Core Ising-to-QUBO Mapping

The goal of this project is to implement and verify the fundamental linear transformations that bridge the **Ising spin variable** ($s_i$) and the **QUBO binary variable** ($x_i$). This mapping is essential for translating real-world decision problems into the language of physical systems.

### Theory: The Transformation Duality

All **combinatorial optimization problems** are mapped onto energy minimization forms: either the **Ising Hamiltonian** or the **Quadratic Unconstrained Binary Optimization (QUBO)** cost function.

| Variable Type | Domain | Physical/Computational Role |
| :--- | :--- | :--- |
| **Ising Spin ($s_i$)** | $\{-1, +1\}$ | Physical state (spin up/down) |
| **QUBO Variable ($x_i$)** | $\{0, 1\}$ | Decision state (No/Yes) |

The two transformations are defined by linear relationships that ensure that every possible state in one domain maps uniquely to a state in the other:

1.  **QUBO to Ising:** (Decision to State)
```
$$s_i = 2x_i - 1$$

  * If $x_i=0 \implies s_i = -1$
  * If $x_i=1 \implies s_i = +1$

```
2.  **Ising to QUBO:** (State to Decision)
```
$$x_i = \frac{s_i + 1}{2}$$

```
These transformations allow us to convert the QUBO cost function $E(\mathbf{x}) = \mathbf{x}^T \mathbf{Q} \mathbf{x}$ into the Ising Hamiltonian $H(\mathbf{s}) = \mathbf{s}^T \mathbf{J} \mathbf{s} + \mathbf{h}^T \mathbf{s}$, and vice-versa, making the problem solvable by specialized hardware like quantum annealers.

---

### Extensive Python Code

```python
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
```
**Sample Output:**
```
--- Ising <-> QUBO Transformation Verification ---
|   QUBO Input (x) |   Ising Output (s) |   Ising Input (s) |   QUBO Output (x) |   Round Trip s |
|-----------------:|-------------------:|------------------:|------------------:|---------------:|
|                0 |                 -1 |                -1 |                 0 |             -1 |
|                1 |                  1 |                 1 |                 1 |              1 |

Conclusion: The transformations are successfully implemented. The QUBO domain (0, 1) maps correctly to the Ising domain (-1, +1), and the inverse transformation restores the original domain values.
```

---

## Project 2: QUBO Construction for Maximum Cut

---

### Definition: QUBO Construction for Maximum Cut

The goal is to construct the **QUBO matrix ($\mathbf{Q}$)** for a simple 4-node, unweighted graph that minimizes the number of edges *not* in the cut. This is a common method for formally encoding the **Maximum Cut (Max Cut) problem** into the QUBO minimization framework.

### Theory: Maximization to Minimization

The Max Cut problem seeks to partition the graph's nodes into two sets ($x_i=0$ and $x_i=1$) such that the number of edges connecting nodes of different sets is maximized.

The objective must be converted into a minimization problem:
$$\text{Maximize } C \implies \text{Minimize } -C$$

For a graph $G=(V, E)$, the number of edges *not* in the cut (same set) for binary variables $x_i \in \{0, 1\}$ is:

$$E_{\text{cut-miss}} = \sum_{\langle i, j \rangle} w_{ij} x_i x_j + \sum_{\langle i, j \rangle} w_{ij} (1-x_i)(1-x_j)$$

To maximize the cut, we minimize the edges *not* in the cut. The final objective is often written in the **QUBO form** $E(\mathbf{x}) = \mathbf{x}^T \mathbf{Q} \mathbf{x}$.

  * The $Q_{ij}$ terms correspond to quadratic coupling (edges).
  * The $Q_{ii}$ terms correspond to linear biases (nodes).

We use a 4-node chain graph for this example (A-B, B-C, C-D).

---

### Extensive Python Code

```python
import numpy as np
import pandas as pd

# ====================================================================
# 1. Setup Graph and Problem Definition
# ====================================================================

# 4-Node Chain Graph (A-B-C-D) with Unit Weights (w_ij = 1)
NODES = ['A', 'B', 'C', 'D']
N = len(NODES)

# Adjacency matrix (A[i, j] = 1 if edge exists)
# Edges: (A, B), (B, C), (C, D)
ADJACENCY_MATRIX = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
])
# We use the adjacency matrix for summing over edges (i, j).

# ====================================================================
# 2. QUBO Matrix Construction
# ====================================================================

# Goal: Minimize the negative cut value: E = -C. 
# The Max Cut objective for x_i in {0, 1} is:
# C = sum_{<i, j>} w_ij * (x_i (1-x_j) + x_j (1-x_i))
# We minimize E = -C.

# The simplified QUBO cost function for Max Cut is generally:
# E = sum_{i} Q_{ii} x_i + sum_{i<j} Q_{ij} x_i x_j
# Where Q_{ij} = 2 * w_{ij} for the off-diagonals, and Q_{ii} = -sum_{j} w_{ij} for the diagonals.

Q = np.zeros((N, N))

for i in range(N):
    # Calculate the sum of weights connected to node i (degree of node i)
    degree_i = np.sum(ADJACENCY_MATRIX[i, :])
    
    # 1. Diagonal Terms (Q_ii * x_i): Q_ii = -degree_i
    Q[i, i] = -degree_i 
    
    for j in range(i + 1, N):
        if ADJACENCY_MATRIX[i, j] == 1:
            # 2. Off-Diagonal Terms (Q_ij * x_i * x_j): Q_ij = 2 * w_ij
            Q[i, j] = 2.0 * ADJACENCY_MATRIX[i, j]
            Q[j, i] = Q[i, j] # Ensure symmetry

# The resulting QUBO matrix Q
df_qubo = pd.DataFrame(Q, index=NODES, columns=NODES)

print("--- QUBO Matrix Q for Max Cut on A-B-C-D Chain ---")
print("Q_{ii} = Linear term (Node Bias); Q_{ij} = Quadratic term (Edge Coupling)")
print(df_qubo.to_markdown(floatfmt=".1f"))

# ====================================================================
# 3. Verification of Solution
# ====================================================================

# Ground Truth for Max Cut on a 4-node chain is 3 edges (e.g., [1, 0, 1, 0])
# We check the cost for the optimal solution x* = [1, 0, 1, 0]
x_opt = np.array([1, 0, 1, 0])
energy_opt = x_opt @ Q @ x_opt.T

print("\nVerification (Optimal Solution x* = [1, 0, 1, 0] | Max Cut = 3)")
print(f"Minimum Energy E = x* Q x* = {energy_opt:.1f}")

print("\nInterpretation: The matrix Q ensures that configurations corresponding to a large cut result in a very low (negative) energy, guiding the solver toward the optimal partitioning.")
```
**Sample Output:**
```
--- QUBO Matrix Q for Max Cut on A-B-C-D Chain ---
Q_{ii} = Linear term (Node Bias); Q_{ij} = Quadratic term (Edge Coupling)
|    |    A |    B |    C |    D |
|:---|-----:|-----:|-----:|-----:|
| A  | -1.0 |  2.0 |  0.0 |  0.0 |
| B  |  2.0 | -2.0 |  2.0 |  0.0 |
| C  |  0.0 |  2.0 | -2.0 |  2.0 |
| D  |  0.0 |  0.0 |  2.0 | -1.0 |

Verification (Optimal Solution x* = [1, 0, 1, 0] | Max Cut = 3)
Minimum Energy E = x* Q x* = -3.0

Interpretation: The matrix Q ensures that configurations corresponding to a large cut result in a very low (negative) energy, guiding the solver toward the optimal partitioning.
```

---

## Project 3: Ising Construction for Antiferromagnetic Chain

---

### Definition: Ising Construction for Antiferromagnetic Chain

The goal is to formulate the **Ising Hamiltonian** and explicitly define the **Ising Matrix ($\mathbf{J}$)** for a 4-node, 1D **Antiferromagnetic Chain** with no external field.

### Theory: Antiferromagnetism and the $\mathbf{J}$ Matrix

The **Ising Hamiltonian** is the cost function for spin variables $s_i \in \{-1, +1\}$:

$$H(\mathbf{s}) = - \sum_{i<j} J_{ij} s_i s_j - \sum_i h_i s_i$$

1.  **Antiferromagnetism:** Requires **negative coupling ($J < 0$)**. The system minimizes energy when neighboring spins are *opposite* ($s_i s_j = -1$). We choose $J_{ij} = -1$ for nearest neighbors.
2.  **Chain Structure:** Only nearest neighbors are coupled (e.g., $J_{1,2} = -1$, but $J_{1,3} = 0$).
3.  **No External Field:** The local bias term is $h_i = 0$.

The final Ising Matrix $\mathbf{J}$ is a $4 \times 4$ symmetric matrix where the diagonal elements are zero and the off-diagonal nearest-neighbor elements are $J=-1$. The ground state (minimum energy) will be one where spins alternate: $[+1, -1, +1, -1]$ or $[-1, +1, -1, +1]$.

---

### Extensive Python Code

```python
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
```
**Sample Output:**
```
--- Ising Matrix J for 1D Antiferromagnetic Chain ---
Coupling J = -1.0 (Negative, favors opposite spins)
|    |    A |    B |    C |    D |
|:---|-----:|-----:|-----:|-----:|
| A  |  0.0 | -1.0 |  0.0 |  0.0 |
| B  | -1.0 |  0.0 | -1.0 |  0.0 |
| C  |  0.0 | -1.0 |  0.0 | -1.0 |
| D  |  0.0 |  0.0 | -1.0 |  0.0 |

Verification (Ground State s* = [+1, -1, +1, -1])
Calculated Ground State Energy E*: -3.0 (Should be -3.0)

Interpretation: The matrix J successfully encodes the Antiferromagnetic couplings, driving the system to a ground state where the spins alternate.
```

---

## Project 4: Encoding the One-Hot Constraint

---

### Definition: Encoding the One-Hot Constraint

The goal is to implement and test the **one-hot constraint** penalty term, which forces the optimization solution to select **exactly one** variable ($x_i=1$) out of a set of $N$ variables ($x_i \in \{0, 1\}$).

### Theory: Penalty Function Method

In QUBO/Ising optimization, constraints are incorporated by adding a large **penalty term ($E_{\text{penalty}}$)** to the objective function. The penalty term is designed to have a value of **zero** when the constraint is satisfied and a large **positive value** otherwise.

The **One-Hot Constraint** ($\sum_{i=1}^N x_i = 1$) is satisfied only if the sum of the binary variables equals 1. The penalty function is a quadratic term that uses a large penalty factor $\lambda$:

$$E_{\text{penalty}} = \lambda \left(\sum_{i=1}^N x_i - 1\right)^2$$

If the constraint is violated (e.g., $\sum x_i = 2$ or $\sum x_i = 0$), the squared term ensures the energy cost is high, steering the optimizer toward the valid solution space.

---

### Extensive Python Code

```python
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
```
**Sample Output:**
```
--- One-Hot Constraint Penalty Test ---
Constraint: Exactly one variable must be selected (∑x_i = 1)
Penalty Factor (λ): 100

Test Case A: x = [1, 0, 0, 0, 0] (∑x_i = 1)
  Penalty E = 0.0 (Correctly Zero)

Test Case B: x = [1, 1, 0, 0, 0] (∑x_i = 2)
  Penalty E = 100.0 (High Penalty)

Test Case C: x = [0, 0, 0, 0, 0] (∑x_i = 0)
  Penalty E = 100.0 (High Penalty)

Conclusion: The penalty term correctly evaluates to zero for the valid solution (Case A) and imposes a large, positive cost for violating the constraint (Cases B and C). This ensures that any solver minimizing the total energy will prioritize satisfying the constraint over optimizing the original cost function, thus encoding the problem correctly for QUBO/Ising solvers.
```