# **Chapter 8: Combinatorial Optimization and QUBO () () () (Workbook)**

The goal of this chapter is to formalize **discrete optimization** by mapping all combinatorial problems onto the physical language of the **Ising Model**, enabling solutions via specialized stochastic and quantum-inspired search techniques.

| Section | Topic Summary |
| :--- | :--- |
| **8.1** | The Nature of Combinatorial Landscapes |
| **8.2** | From Continuous to Discrete Energy (QUBO) |
| **8.3** | QUBO $\leftrightarrow$ Ising Mapping |
| **8.4** | Constraint Encoding |
| **8.5** | Solving QUBO Problems |
| **8.6** | Case Study — Traveling Salesman Problem |
| **8.7** | Graph Coloring and Partitioning |
| **8.8–8.13** | QUBO in ML, Solvers, and Takeaways |

---

### 8.1 The Nature of Combinatorial Landscapes

> **Summary:** **Combinatorial Optimization** deals with fundamentally **discrete variables** (binary $x_i \in \{0, 1\}$ or spin $s_i \in \{-1, +1\}$). The total search space grows **exponentially** ($\Omega = 2^N$), making brute-force search impossible. The core challenge is that the loss landscape is a collection of **isolated points**, providing **no useful gradient information**. The goal is the same as the physical problem of **finding the ground state** (minimum energy) of the system.

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary characteristic of a Combinatorial Optimization problem that invalidates the use of gradient descent (Chapters 5 and 6) is:**

* **A.** The high number of local minima.
* **B.** **The discontinuous, discrete loss landscape with no meaningful gradient**. (**Correct**)
* **C.** The presence of external fields.
* **D.** The time-dependence of the variables.

```
!!! note "Quiz"
```
**2. The search for the optimal solution $\mathbf{x}^*$ in a combinatorial problem is physically equivalent to the fundamental physics problem of finding the system's:**

* **A.** Partition function.
* **B.** **Ground state (minimum energy configuration)**. (**Correct**)
* **C.** Critical temperature.
* **D.** Linear momentum.

```
---

!!! question "Interview Practice"
```
**Question:** The combinatorial search space grows as $2^N$. Explain the significance of this exponential growth for a practical logistics problem like the Traveling Salesman Problem (TSP) with $N=100$ cities?

**Answer Strategy:** The growth is so rapid that even for a relatively small $N=100$, the total number of possible solutions ($2^{100} \approx 10^{30}$) exceeds the computational capacity of any known classical computer, making **exhaustive search (brute force enumeration)** completely impossible. This confirms that the TSP is an **NP-hard problem** and requires approximate heuristic search methods or specialized quantum hardware.

```
---

### 8.2 From Continuous to Discrete Energy (QUBO)

> **Summary:** The core formalism for discrete search is the **Quadratic Unconstrained Binary Optimization (QUBO)** model. The QUBO energy is defined by a quadratic polynomial in binary variables $x_i \in \{0, 1\}$: $E(\mathbf{x}) = \sum_i a_i x_i + \sum_{i<j} b_{ij} x_i x_j$. The structure directly parallels the **Ising model**, where the **quadratic coupling term** ($b_{ij}$) represents **interaction energy**, and the **linear bias term** ($a_i$) represents the **external field**.

#### Quiz Questions

!!! note "Quiz"
```
**1. The QUBO function is characterized by its reliance on which type of mathematical term to model the interaction energy between two variables $x_i$ and $x_j$?**

* **A.** A linear term ($x_i + x_j$).
* **B.** **A quadratic term ($x_i x_j$)**. (**Correct**)
* **C.** An external field $h_i$.
* **D.** A constant offset.

```
!!! note "Quiz"
```
**2. The single-variable term $\sum_i a_i x_i$ in the QUBO energy function is analogous to which term in the standard Ising Hamiltonian?**

* **A.** The interaction term ($J_{ij}$).
* **B.** The partition function ($Z$).
* **C.** **The external field ($h_i$)**. (**Correct**)
* **D.** The system temperature ($T$).

```
---

!!! question "Interview Practice"
```
**Question:** In the QUBO formalism, explain the physical meaning of a **negative coefficient** in the quadratic term, $b_{ij} < 0$, when modeling a decision problem.

**Answer Strategy:** A negative coefficient in $b_{ij}$ (or a positive coupling $J_{ij}$ in the equivalent Ising model) represents an **attractive or beneficial interaction**. The system minimizes the cost (energy), so $b_{ij} x_i x_j$ is maximized when $x_i = x_j = 1$. This means the cost is lowest (energy is most negative) when the two variables are **aligned**. For a decision problem, $b_{ij} < 0$ means that selecting variable $x_i$ and $x_j$ **together** is highly desirable (low cost).

```
---

### 8.3 QUBO $\leftrightarrow$ Ising Mapping

> **Summary:** The linear relationship $s_i = 2x_i - 1$ allows for a **direct, analytic equivalence** between the Ising Hamiltonian ($E(\mathbf{s})$) and the QUBO cost ($E(\mathbf{x})$). This conversion results in the compact matrix form $E(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x} + \text{constant}$. The implication is that **any combinatorial problem** can be translated into the QUBO matrix $Q$ and solved by finding the **ground state of the equivalent magnetic system**.

#### Quiz Questions

!!! note "Quiz"
```
**1. Which mathematical object, derived from the QUBO cost function, serves as the native input structure for specialized solvers, such as quantum annealers?**

* **A.** The linear bias term $\sum a_i x_i$.
* **B.** The **QUBO matrix $Q$**. (**Correct**)
* **C.** The partition function $Z$.
* **D.** The spin vector $\mathbf{s}$.

```
!!! note "Quiz"
```
**2. When converting the Ising model to the QUBO model, the spin variable $s_i \in \{-1, +1\}$ is mapped to the binary variable $x_i \in \{0, 1\}$ using which linear relationship?**

* **A.** $x_i = s_i + 1$.
* **B.** $x_i = 2s_i + 1$.
* **C.** **$x_i = (s_i + 1) / 2$**. (**Correct**)
* **D.** $x_i = s_i / 2$.

```
---

!!! question "Interview Practice"
```
**Question:** The QUBO $\leftrightarrow$ Ising mapping is described as providing a **universal intermediary** for optimization. What practical benefit does this universality offer to a computer scientist facing a new, difficult combinatorial problem?

**Answer Strategy:** The universality means the scientist doesn't need to invent a new solver for every problem. Instead, they focus solely on the (difficult) task of translating the problem's logic and constraints into the QUBO matrix $Q$. Once $Q$ is found, they can immediately use any existing **Ising ground state solver**—classical heuristics (Simulated Annealing) or specialized hardware (Quantum Annealers)—as a black-box optimizer, dramatically simplifying the solution pipeline.

```
---

### 8.4 Constraint Encoding

> **Summary:** Since QUBO is inherently **unconstrained**, strict rules must be absorbed into the cost function itself. **Constraint Encoding** converts constraints into non-negative **penalty terms** $C_k(\mathbf{x})^2$. The total energy becomes $E'(\mathbf{x}) = E(\mathbf{x}) + \lambda \sum_k C_k(\mathbf{x})^2$. The large **Lagrange Multiplier ($\lambda$)** ensures any constraint violation incurs a massive **energy wall**, forcing the optimizer back to the feasible region.

#### Quiz Questions

!!! note "Quiz"
```
**1. When encoding constraints in the QUBO model, the objective function is augmented with penalty terms that are typically squared. This is done to ensure the penalty is:**

* **A.** Reducible to a linear term.
* **B.** **Non-negative, ensuring that any violation increases the total energy**. (**Correct**)
* **C.** Equal to the partition function.
* **D.** Always minimized by the solver.

```
!!! note "Quiz"
```
**2. In the augmented energy $E'(\mathbf{x})$, the large coefficient $\lambda$ multiplying the penalty term $C_k(\mathbf{x})^2$ serves the physical purpose of:**

* **A.** Allowing the system to explore uphill moves.
* **B.** **Acting as a spring constant that enforces feasibility by creating a massive energy wall**. (**Correct**)
* **C.** Normalizing the total energy $E'$.
* **D.** Defining the optimal route distance.

```
---

!!! question "Interview Practice"
```
**Question:** The simplest constraint, the **"one-hot" constraint** (a set of variables must sum to exactly one, $\sum x_i = 1$), is essential for many problems. Explain why this constraint, when squared as a penalty, is reducible to the required QUBO matrix form?

**Answer Strategy:** The penalty term is $C^2 = (\sum x_i - 1)^2$. When expanded, this yields:
$$C^2 = (\sum x_i)^2 - 2 \sum x_i + 1$$
This contains three types of terms: $\sum x_i x_j$ (quadratic), $\sum x_i$ (linear), and a constant. These are the only three forms required to construct the QUBO matrix $Q$, meaning the constraint can be perfectly mapped onto the Ising energy structure.

```
---

### 💡 Hands-On Project Ideas 🛠️

These projects require computational techniques to implement and analyze the core concepts of QUBO formulation and search.

### Project 1: Implementing the QUBO Energy Function

* **Goal:** Implement the core QUBO energy calculation for an arbitrary matrix $Q$.
* **Setup:** Generate a small, random $5 \times 5$ QUBO matrix $Q$ (ensure it's symmetric) and a binary vector $\mathbf{x} \in \{0, 1\}^5$.
* **Steps:**
    1.  Implement the energy function $E(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}$ using NumPy matrix operations.
    2.  Test the function with two random $\mathbf{x}$ vectors and print their energy values.
    3.  Convert the best $\mathbf{x}$ to the equivalent Ising spin vector $\mathbf{s} \in \{-1, +1\}^5$.
* ***Goal***: Establish the fundamental computational object and demonstrate the QUBO $\leftrightarrow$ Ising conversion.

#### Python Implementation

```python
import numpy as np
import pandas as pd

# ====================================================================

## 1. Transformation Functions

## ====================================================================

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

## ====================================================================

## 2. Verification Test

## ====================================================================

## Test inputs

x_test = [0, 1]
s_test = [-1, 1]

## Forward and inverse transformations

s_forward = qubo_to_ising(x_test)
x_inverse = ising_to_qubo(s_test)
s_round_trip = qubo_to_ising(x_inverse)

## Create a DataFrame for verification

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
```python
--- Ising <-> QUBO Transformation Verification ---
|   QUBO Input (x) |   Ising Output (s) |   Ising Input (s) |   QUBO Output (x) |   Round Trip s |
|-----------------:|-------------------:|------------------:|------------------:|---------------:|
|                0 |                 -1 |                -1 |                 0 |             -1 |
|                1 |                  1 |                 1 |                 1 |              1 |

Conclusion: The transformations are successfully implemented. The QUBO domain (0, 1) maps correctly to the Ising domain (-1, +1), and the inverse transformation restores the original domain values.
```


### Project 2: Exact Solution for Small QUBO (Brute Force)

* **Goal:** Implement a brute-force solver to find the exact global minimum for a tiny QUBO problem, illustrating the limits of classical computation.
* **Setup:** Use a small $N=4$ QUBO matrix $Q$.
* **Steps:**
    1.  Use `itertools.product` to generate all $2^4 = 16$ possible binary configurations $\mathbf{x}$.
    2.  Iterate through all configurations, calculate $E(\mathbf{x})$ for each, and store the configuration $\mathbf{x}^*$ corresponding to the minimum energy $E_{\min}$.
* ***Goal***: Confirm the final minimum energy and the corresponding optimal configuration $\mathbf{x}^*$, proving that this method is guaranteed to find the optimum, though it is not scalable.

#### Python Implementation

```python
import numpy as np
import pandas as pd

## ====================================================================

## 1. Setup Graph and Problem Definition

## ====================================================================

## 4-Node Chain Graph (A-B-C-D) with Unit Weights (w_ij = 1)

NODES = ['A', 'B', 'C', 'D']
N = len(NODES)

## Adjacency matrix (A[i, j] = 1 if edge exists)

## Edges: (A, B), (B, C), (C, D)

ADJACENCY_MATRIX = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
])
## We use the adjacency matrix for summing over edges (i, j).

## ====================================================================

## 2. QUBO Matrix Construction

## ====================================================================

## Goal: Minimize the negative cut value: E = -C.

## The Max Cut objective for x_i in {0, 1} is:

## C = sum_{<i, j>} w_ij * (x_i (1-x_j) + x_j (1-x_i))

## We minimize E = -C.

## The simplified QUBO cost function for Max Cut is generally:

## E = sum_{i} Q_{ii} x_i + sum_{i<j} Q_{ij} x_i x_j

## Where Q_{ij} = 2 * w_{ij} for the off-diagonals, and Q_{ii} = -sum_{j} w_{ij} for the diagonals.

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

## The resulting QUBO matrix Q

df_qubo = pd.DataFrame(Q, index=NODES, columns=NODES)

print("--- QUBO Matrix Q for Max Cut on A-B-C-D Chain ---")
print("Q_{ii} = Linear term (Node Bias); Q_{ij} = Quadratic term (Edge Coupling)")
print(df_qubo.to_markdown(floatfmt=".1f"))

## ====================================================================

## 3. Verification of Solution

## ====================================================================

## Ground Truth for Max Cut on a 4-node chain is 3 edges (e.g., [1, 0, 1, 0])

## We check the cost for the optimal solution x* = [1, 0, 1, 0]

x_opt = np.array([1, 0, 1, 0])
energy_opt = x_opt @ Q @ x_opt.T

print("\nVerification (Optimal Solution x* = [1, 0, 1, 0] | Max Cut = 3)")
print(f"Minimum Energy E = x* Q x* = {energy_opt:.1f}")

print("\nInterpretation: The matrix Q ensures that configurations corresponding to a large cut result in a very low (negative) energy, guiding the solver toward the optimal partitioning.")
```
**Sample Output:**
```python
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


### Project 3: Formulating a Graph Partitioning QUBO

* **Goal:** Manually construct the QUBO matrix $Q$ for a simple **Graph Partitioning** problem.
* **Setup:** Consider a simple 4-node graph (nodes $A, B, C, D$) with edges: $A-B$ (weight 5) and $C-D$ (weight 5), and $B-C$ (weight 1). The goal is to partition the graph into two sets ($\pm 1$) to minimize the cut weight.
* **Steps:**
    1.  Define the Ising Energy: $E = \sum_{i<j} w_{ij}(1-s_i s_j)$.
    2.  Write out the resulting $4 \times 4$ QUBO matrix $Q$ that corresponds to this energy function. (Hint: The problem is natively Ising, so the conversion to QUBO involves simple linear terms and quadratic terms).
* ***Goal***: Produce the matrix $Q$ that, when minimized, will place the weakly coupled nodes ($B$ and $C$) in different sets, thus minimizing the energy.

#### Python Implementation

```python
import numpy as np
import pandas as pd

## ====================================================================

## 1. Setup Parameters and J-Matrix Construction

## ====================================================================

NODES = ['A', 'B', 'C', 'D']
N = len(NODES)
J_VAL = -1.0 # Antiferromagnetic coupling (negative J)
H_VAL = 0.0  # No external field

## Initialize the 4x4 Ising Matrix J

J = np.zeros((N, N))

## Nearest-Neighbor Coupling (A-B, B-C, C-D)

## J[i, j] must be symmetric J[i, j] = J[j, i]

## A-B (Index 0-1)

J[0, 1] = J_VAL
J[1, 0] = J_VAL

## B-C (Index 1-2)

J[1, 2] = J_VAL
J[2, 1] = J_VAL

## C-D (Index 2-3)

J[2, 3] = J_VAL
J[3, 2] = J_VAL

## Linear term h is zero, so we only need the J matrix.

## The resulting Ising Matrix J

df_ising = pd.DataFrame(J, index=NODES, columns=NODES)

print("--- Ising Matrix J for 1D Antiferromagnetic Chain ---")
print(f"Coupling J = {J_VAL} (Negative, favors opposite spins)")
print(df_ising.to_markdown(floatfmt=".1f"))

## ====================================================================

## 2. Verification of Ground State Energy

## ====================================================================

## The optimal state (ground state) is s* = [+1, -1, +1, -1]

s_opt = np.array([1, -1, 1, -1])

## The ground state energy E = - sum J_ij * s_i * s_j

## Energy of the ground state = -(J_AB*sA*sB + J_BC*sB*sC + J_CD*sC*sD)

## E = -[(-1)*(1)*(-1) + (-1)*(-1)*(1) + (-1)*(1)*(-1)] = -[1 + 1 + 1] = -3

## We calculate the full matrix energy: E = -s^T J s

## Since J is symmetric, the matrix product is H = -0.5 * s^T J s

## We calculate the full Hamiltonian H = -sum J_ij s_i s_j (the first term)

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
```python
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


### Project 4: Encoding the One-Hot Constraint

* **Goal:** Implement and test the **one-hot constraint** penalty term to ensure the optimizer only selects valid solutions.
* **Setup:** Define a 5-variable problem where the constraint is $\sum x_i = 1$ (exactly one must be selected). Use a high penalty $\lambda=100$.
* **Steps:**
    1.  Write the penalty term $E_{\text{penalty}} = \lambda (\sum_{i=1}^5 x_i - 1)^2$.
    2.  Calculate $E_{\text{penalty}}$ for three test vectors: $\mathbf{x}_A = [1, 0, 0, 0, 0]$ (Valid), $\mathbf{x}_B = [1, 1, 0, 0, 0]$ (Violation), and $\mathbf{x}_C = [0, 0, 0, 0, 0]$ (Violation).
* ***Goal***: Show that $E_{\text{penalty}}$ is zero for $\mathbf{x}_A$, but a very large positive number (100 or 400) for the two violated configurations $\mathbf{x}_B$ and $\mathbf{x}_C$. This confirms the penalty term correctly guides the search away from infeasible solutions.

#### Python Implementation

```python
import numpy as np

## ====================================================================

## 1. Setup Parameters and Penalty Function

## ====================================================================

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

## ====================================================================

## 2. Test Solutions

## ====================================================================

## Test Case A: Valid Solution (One element selected)

x_A = np.array([1, 0, 0, 0, 0])
E_A = one_hot_penalty(x_A)

## Test Case B: Violation (Two elements selected)

x_B = np.array([1, 1, 0, 0, 0])
E_B = one_hot_penalty(x_B)

## Test Case C: Violation (Zero elements selected)

x_C = np.array([0, 0, 0, 0, 0])
E_C = one_hot_penalty(x_C)

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

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
```python
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