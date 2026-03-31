# **Chapter 11: Graphical Models & Probabilistic Graphs () () () (Workbook)**

The goal of this chapter is to generalize inference to complex, coupled systems, using graph theory to model dependencies and local message passing to achieve global statistical equilibrium.

| Section | Topic Summary |
| :--- | :--- |
| **11.1** | From Single Models to Networks of Belief |
| **11.2** | Bayesian Networks (BNs) — Directed Acyclic Graphs |
| **11.3** | Markov Random Fields (MRFs) — Undirected Graphs |
| **11.4** | Conditional Independence and Markov Blankets |
| **11.5** | Factor Graphs — Unified Representation |
| **11.6** | Belief Propagation (BP) |
| **11.7** | Variational Inference and Free Energy Minimization |
| **11.8** | Loopy and Approximate Belief Propagation |
| **11.9** | Dynamic and Temporal Graphical Models |
| **11.10–11.14** | Worked Example, Code Demo, and Takeaways |


### 11.1 From Single Models to Networks of Belief

> **Summary:** Graphical models are necessary for systems with **structured dependencies** (e.g., atoms, neurons). The graph acts as a **blueprint** to factorize the complex joint probability $P(\mathbf{x})$ into a product of local terms. **Bayesian Networks (BNs)** use directed edges for causal flow, while **Markov Random Fields (MRFs)** use undirected edges for symmetric coupling. The goal is to use **local relationships** to infer **global behavior**.

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary conceptual role of a graphical model (BN or MRF) is to simplify the complex **joint probability distribution** $P(\mathbf{x})$ by:**

* **A.** Normalizing it using the partition function $Z$.
* **B.** **Factorizing it into a product of simpler, local probability terms**. (**Correct**)
* **C.** Converting the distribution to an energy function $E$.
* **D.** Calculating the model evidence $p(\mathcal{D})$.

```
!!! note "Quiz"
```
**2. Which type of graphical model encodes **symmetric interactions** or mutual dependencies without implying a direction or causal hierarchy?**

* **A.** Dynamic Bayesian Networks (DBNs).
* **B.** **Markov Random Fields (MRFs)**. (**Correct**)
* **C.** Bayesian Networks (BNs).
* **D.** Recurrent Neural Networks (RNNs).


```
!!! question "Interview Practice"
```
**Question:** The text uses the analogy of **information flow** for BNs and **exchange of forces** for MRFs. Explain how this physical distinction relates to the visual difference between the two graph types.

**Answer Strategy:**
* **BNs (Information Flow):** BNs use **directed edges** ($\to$) to model a clear causal hierarchy. This visually represents the **unidirectional flow of information** or influence from cause to effect, like an input signal propagating through a system.
* **MRFs (Exchange of Forces):** MRFs use **undirected edges** ($\leftrightarrow$) to model **symmetric coupling**. This visually represents a mutual, reciprocal relationship, analogous to the exchange of forces or interaction energy between two coupled physical entities.


```
### 11.2 Bayesian Networks — Directed Acyclic Graphs (DAGs)

> **Summary:** Bayesian Networks (BNs) use a **Directed Acyclic Graph (DAG)** to enforce a clear **causal hierarchy** or flow of information. The joint probability is factored into a product of local **Conditional Probability Distributions (CPDs)** for each variable given its **parents**. The primary task is **inference**, which involves the propagation of evidence (messages) along the directed edges of the graph.

#### Quiz Questions

!!! note "Quiz"
```
**1. The constraint that a Bayesian Network must adhere to a **Directed Acyclic Graph (DAG)** enforces which structural property?**

* **A.** That the edges must be symmetric.
* **B.** **A clear, unambiguous causal order with no directed loops**. (**Correct**)
* **C.** The equivalence to an Ising spin system.
* **D.** That the system is always in statistical equilibrium.

```
!!! note "Quiz"
```
**2. In the BN factorization, the probability of a node $x_i$ is conditioned only on the state of its:**

* **A.** Children.
* **B.** Co-parents.
* **C.** **Parents**. (**Correct**)
* **D.** All other nodes.


```
!!! question "Interview Practice"
```
**Question:** The final joint probability of a BN is a product of simple local CPDs, $p(x_i|\text{parents}(x_i))$. Explain the computational advantage of this local factorization over having to model the single, massive joint probability distribution directly.

**Answer Strategy:** The advantage is **exponential computational savings**. For a system with $N$ binary variables, the direct joint probability table requires $2^N$ entries. By factoring the distribution, we only need to specify the CPDs for each node given its small number of parents. This turns the problem from exponentially complex to one that scales polynomially with the graph's complexity, making the problem tractable for modern inference algorithms.


```
### 11.3 Markov Random Fields (MRFs)

> **Summary:** Markov Random Fields (MRFs) use an **undirected graph** to model **symmetric interactions** and mutual dependencies. The joint probability is defined by **potential functions ($\psi_C$)** over graph cliques, yielding the **Gibbs distribution**, $p(\mathbf{x}) = \frac{1}{Z} e^{-E(\mathbf{x})}$. MRFs are the statistical generalization of **Ising and Potts spin networks**, where the potentials map directly to the system's coupling energy.

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary structural characteristic of a Markov Random Field (MRF) is that its edges:**

* **A.** Must enforce a clear causal direction.
* **B.** **Are undirected, encoding symmetric dependencies**. (**Correct**)
* **C.** Must form an acyclic graph.
* **D.** Must be non-negative.

```
!!! note "Quiz"
```
**2. The exponential form of the MRF factorization directly relates the probability $p(\mathbf{x})$ to the system's total energy $E(\mathbf{x})$ via which physical distribution?**

* **A.** The Student's t-distribution.
* **B.** The Binomial distribution.
* **C.** **The Gibbs (Boltzmann) distribution**. (**Correct**)
* **D.** The Gaussian Mixture Model.


```
!!! question "Interview Practice"
```
**Question:** The MRF and the Ising Model (Chapter 8) share the same mathematical foundation. Explain how the MRF structure is used in **image denoising** as an analogy to a physical system seeking its ground state.

**Answer Strategy:**
* **Mapping:** Each pixel in the image is a node in the MRF, and its gray value (color) is the variable. Edges connect neighboring pixels.
* **Energy (Loss):** The total energy $E(\mathbf{x})$ is constructed with two penalty terms: 1) **Data fidelity** (cost for the pixel being different from its observed noisy value), and 2) **Smoothness/Coupling** (cost for neighboring pixels being different).
* **Ground State:** The algorithm (e.g., using MCMC) finds the minimum energy configuration $E_{\min}$, which is the state where the image is both close to the observed data and locally smooth (denoised), analogous to the physical system reaching its most stable, ordered configuration.


```
### 11.4 Conditional Independence and Markov Blankets

> **Summary:** The network structure reveals **conditional independencies**, a key property for decoupling the system. The **Markov Blanket ($\text{MB}_i$)** is the minimal set of surrounding variables that makes $x_i$ conditionally independent of all other nodes. In MRFs, the $\text{MB}_i$ is simply the **direct neighbors** of $x_i$. This property is the foundation for local, distributed computation, as distant influences are **screened out** by the local neighborhood.

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary utility of the **Markov Blanket ($\text{MB}_i$)** in global inference is that it: **

* **A.** Guarantees the graph has no cycles.
* **B.** **Renders variable $x_i$ conditionally independent of the rest of the network**. (**Correct**)
* **C.** Increases the total number of cliques.
* **D.** Solves the partition function.

```
!!! note "Quiz"
```
**2. In an Undirected Graphical Model (MRF), the Markov Blanket of a variable $x_i$ consists only of:**

* **A.** Its parents and children.
* **B.** All other nodes in the graph.
* **C.** **Its set of direct neighbors**. (**Correct**)
* **D.** The factor nodes connected to it.


```
!!! question "Interview Practice"
```
**Question:** Explain the computational significance of the $\text{MB}_i$ concept using the analogy of **screening** in statistical physics.

**Answer Strategy:** In statistical physics, the influence of a charge or spin in a medium is screened out by the local, surrounding particles. The Markov Blanket provides the minimal **local boundary** that performs the same function.
* **Significance:** For inference, to calculate the probability of $x_i$, we don't need to model the entire system; we only need information about $x_i$ and its $\text{MB}_i$. This property allows the complex global inference problem to be broken down into a set of highly efficient **local computations**, making distributed message passing possible.


```
### 11.5 Factor Graphs — Unified Representation

> **Summary:** **Factor Graphs** provide a clear, unified representation for both BNs and MRFs using a **bipartite graph**. Nodes are split into **variable nodes ($x_i$)** and **factor nodes ($f_a$)**. The joint distribution is the explicit product of all local factor functions: $p(\mathbf{x}) = \frac{1}{Z}\prod_a f_a(\mathbf{x}_a)$. This structure simplifies the equations for iterative inference algorithms like **Belief Propagation**.

#### Quiz Questions

!!! note "Quiz"
```
**1. A Factor Graph is categorized as a **bipartite graph** because its nodes are exclusively divided into which two types?**

* **A.** Directed and Undirected.
* **B.** Parents and Children.
* **C.** **Variable nodes ($x_i$) and Factor nodes ($f_a$)**. (**Correct**)
* **D.** Clustered and Independent.

```
!!! note "Quiz"
```
**2. In the Factor Graph formulation, the explicit function that represents the "forces" or coupling rules acting between a subset of variables is the:**

* **A.** Final belief vector.
* **B.** Partition function $Z$.
* **C.** **Factor node $f_a$**. (**Correct**)
* **D.** Message from the variable.


```
!!! question "Interview Practice"
```
**Question:** Explain the geometric advantage of the Factor Graph representation for the purpose of **Belief Propagation (BP)**?

**Answer Strategy:** The advantage is **algorithmic clarity**. By explicitly separating variables from the functions that couple them, the Factor Graph provides a simplified geometry where message passing only needs to occur between two clear entities: variable $\leftrightarrow$ factor. This eliminates the ambiguity found in complex MRFs with large cliques, providing a **straightforward and easy-to-implement framework** for the iterative message-passing equations of BP.


```
### 11.6 Belief Propagation (BP)

> **Summary:** **Belief Propagation (BP)** is the iterative inference procedure that computes the **marginal probability** (final belief) for each node. It works by exchanging **messages** between neighboring nodes, which summarize the influence of distant parts of the network. The process continues until the beliefs are **self-consistent** (convergence). BP is the computational analogue of **distributed relaxation to statistical equilibrium**.

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary statistical quantity that Belief Propagation is designed to compute for every node in the network is the:**

* **A.** Joint probability $P(\mathbf{x})$.
* **B.** **Marginal probability $P(x_i)$ (the node's final belief)**. (**Correct**)
* **C.** Model evidence $P(\mathcal{D})$.
* **D.** KL divergence $D_{\mathrm{KL}}$.

```
!!! note "Quiz"
```
**2. For what specific type of graph structure is the Belief Propagation algorithm guaranteed to converge to the exact marginal probabilities?**

* **A.** Loopy graphs.
* **B.** Undirected graphs.
* **C.** **Tree-structured graphs (graphs without cycles)**. (**Correct**)
* **D.** Fully connected graphs.


```
!!! question "Interview Practice"
```
**Question:** BP is often said to model the network's relaxation to statistical equilibrium. Explain how the **message-passing dynamics** enforce this self-consistency in the system's beliefs.

**Answer Strategy:** The message-passing equations enforce a state of **local self-consistency**. A node's updated belief must be consistent with the weighted influences (messages) received from all its neighbors. By iterating this local update across the entire graph, the system achieves a **global fixed point** where no node can reduce its uncertainty or change its marginal probability based on new information. This stable state is the global statistical equilibrium imposed by the network's structure and potentials.


```
### 11.7 Variational Inference and Free Energy Minimization

> **Summary:** When the true posterior $p(\mathcal{\theta})$ is intractable, **Variational Inference (VI)** seeks the best tractable approximation $q(\mathcal{\theta})$ by minimizing the **Kullback-Leibler (KL) divergence** $D_{\mathrm{KL}}(q||p)$. This minimization is equivalent to minimizing the **Variational Free Energy functional ($\mathcal{F}(q)$)**. This approach is the statistical analog of the **mean-field approximation** in physics, which simplifies complex interactions to make the problem solvable.

### 11.8 Loopy and Approximate Belief Propagation

> **Summary:** Applying BP to graphs with **cycles (loops)** violates its core assumption, as messages become statistically correlated. **Loopy Belief Propagation (LBP)**, however, is a common and effective **approximation algorithm**. The convergence of LBP is often linked to finding the stationary points of the **Bethe approximation** to the Variational Free Energy, reinforcing the link to statistical equilibrium.

### 11.9 Dynamic and Temporal Graphical Models

> **Summary:** **Dynamic Bayesian Networks (DBNs)** model sequential processes by linking graphs across time steps. They rely on the **Markov property** ($t+1$ depends only on $t$). Key temporal models include the **Hidden Markov Model (HMM)**, which models sequences with unobserved latent states, and the **Kalman Filter**, which models continuous Gaussian systems. The DBN transition probabilities encode the system's **probabilistic equations of motion**.



## 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core structural and dynamic concepts of Probabilistic Graphical Models.

### Project 1: Implementing and Testing the Ising Energy Factor

* **Goal:** Implement the fundamental **energy factor** that defines the coupling in an MRF/Ising model.
* **Setup:** Define a simple $2 \times 2$ grid (4 nodes) with ferromagnetic coupling ($J>0$) and no external field ($h=0$).
* **Steps:**
    1.  Define the energy contribution of a single bond (factor) between two spins $s_i, s_j$: $E_{ij} = -J s_i s_j$.
    2.  Write a function that calculates the total unnormalized probability (potential) for the entire 4-node system: $\Psi(\mathbf{s}) = \prod_{\langle i,j \rangle} e^{-E_{ij}}$.
* ***Goal***: Show that $\Psi(\mathbf{s})$ is maximized when all spins are aligned (low energy), and minimized when spins are disordered (high energy), confirming the energy-probability duality.

#### Python Implementation

```python
import numpy as np
import pandas as pd

# ====================================================================
# 1. Setup Network and Conditional Probability Tables (CPTs)
# ====================================================================

# Dependency: A -> B -> C (A is root, C is leaf)
# Variables are binary: 0 (False) or 1 (True)

# P(A) - Prior for the root node
# Index [0] is P(A=0), Index [1] is P(A=1)
P_A = np.array([0.4, 0.6]) 

# P(B | A) - Conditional Probability Table (CPT)
# Rows: P(B | Parent)
# P_B_given_A[A_state, B_state]
P_B_given_A = np.array([
    [0.8, 0.2],  # P(B=0|A=0), P(B=1|A=0)
    [0.1, 0.9]   # P(B=0|A=1), P(B=1|A=1)
])

# P(C | B) - Conditional Probability Table (CPT)
# P_C_given_B[B_state, C_state]
P_C_given_B = np.array([
    [0.9, 0.1],  # P(C=0|B=0), P(C=1|B=0)
    [0.2, 0.8]   # P(C=0|B=1), P(C=1|B=1)
])

# ====================================================================
# 2. Joint Probability Calculation (Factoring Rule)
# ====================================================================

# Goal: Compute P(A=1, B=0, C=1)
# State Indices: A_idx=1, B_idx=0, C_idx=1

A_idx = 1
B_idx = 0
C_idx = 1

# 1. Term P(A=1)
Term_A = P_A[A_idx]

# 2. Term P(B=0 | A=1)
Term_B_given_A = P_B_given_A[A_idx, B_idx]

# 3. Term P(C=1 | B=0)
Term_C_given_B = P_C_given_B[B_idx, C_idx]

# Total Joint Probability P(A, B, C) = P(A) * P(B|A) * P(C|B)
P_joint = Term_A * Term_B_given_A * Term_C_given_B

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Joint Probability Calculation using Bayesian Network ---")
print(f"Network Structure: A \u2192 B \u2192 C")
print(f"Target State: P(A={A_idx}, B={B_idx}, C={C_idx})")
print("---------------------------------------------------------------")
print(f"Term 1: P(A=1) = {Term_A:.2f}")
print(f"Term 2: P(B=0 | A=1) = {Term_B_given_A:.2f}")
print(f"Term 3: P(C=1 | B=0) = {Term_C_given_B:.2f}")

print(f"\nFinal Joint Probability P(1, 0, 1): {P_joint:.4f}")

print("\nConclusion: The Bayesian Network framework allows the complex joint probability of the state (A=1, B=0, C=1) to be computed efficiently by factoring it into a product of local conditional probabilities defined by the graph's topology.")
```
**Sample Output:**
```
--- Joint Probability Calculation using Bayesian Network ---
Network Structure: A → B → C
Target State: P(A=1, B=0, C=1)

---

Term 1: P(A=1) = 0.60
Term 2: P(B=0 | A=1) = 0.10
Term 3: P(C=1 | B=0) = 0.10

Final Joint Probability P(1, 0, 1): 0.0060

Conclusion: The Bayesian Network framework allows the complex joint probability of the state (A=1, B=0, C=1) to be computed efficiently by factoring it into a product of local conditional probabilities defined by the graph's topology.
```


### Project 2: Simulating Belief Propagation (Exact Solution)

* **Goal:** Implement the **Belief Propagation (BP)** algorithm on a simple chain to compute exact marginals.
* **Setup:** Use the binary chain $A-B-C$ defined in the worked example (Section 11.10).
* **Steps:**
    1.  Implement the iterative message update rules (variable $\leftrightarrow$ factor).
    2.  Run the iterations and verify that the calculated belief $P(B)$ converges to the final stable probability distribution.
* ***Goal***: Confirm the final numerical belief $P(B)$ (e.g., $P(B=0) \approx 0.44$, $P(B=1) \approx 0.56$), demonstrating how local messages yield the correct global inference.

#### Python Implementation

```python
import numpy as np

# ====================================================================
# 1. Setup Network and Initial Data
# ====================================================================

# Network: A - B - C (Node B is calculating the message to C)
# Variables are binary: x_A, x_B, x_C \in {0, 1}

# --- Node B's Local Evidence (Factor \psi_{B,C}) ---
# This is the CPT-like factor \psi(x_B, x_C) or the edge potential
# We simplify by using an edge potential that favors x_B == x_C
# Rows (x_B), Columns (x_C)
FACTOR_B_C = np.array([
    [0.9, 0.1],  # x_B=0 favors x_C=0 (90%)
    [0.1, 0.9]   # x_B=1 favors x_C=1 (90%)
])

# --- Incoming Message to B from A (\mu_{A \to B}) ---
# This message is B's current belief about A's state
# P(x_B=0), P(x_B=1) - Uniform prior for the next iteration
MU_A_TO_B = np.array([0.5, 0.5]) 

# ====================================================================
# 2. Belief Propagation Update
# ====================================================================

# Goal: Calculate the outgoing message from B to C: \mu_{B \to C}(x_C)
# Message formula: \mu_{B \to C}(x_C) \propto \sum_{x_B} \psi(x_B, x_C) * \mu_{A \to B}(x_B)

# The outgoing message \mu_{B \to C} will be a vector of size 2 (for x_C=0 and x_C=1)
MU_B_TO_C = np.zeros(2)

# Loop over the target variable x_C (index 0 and 1)
for x_C in range(2):
    # The sum is over x_B (index 0 and 1)
    sum_term = 0.0
    for x_B in range(2):
        # 1. Local Factor: \psi(x_B, x_C)
        factor_term = FACTOR_B_C[x_B, x_C]
        
        # 2. Product of Incoming Messages: \mu_{A \to B}(x_B)
        # Note: B only has one other neighbor (A)
        incoming_message = MU_A_TO_B[x_B]
        
        sum_term += factor_term * incoming_message
        
    MU_B_TO_C[x_C] = sum_term

# Normalize the final message (since it's only proportional)
MU_B_TO_C /= np.sum(MU_B_TO_C)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Belief Propagation Message Calculation (\u03bc_{B \u2192 C}) ---")
print(f"Incoming Message from A (\u03bc_{A \u2192 B}): {np.round(MU_A_TO_B, 3)}")
print("---------------------------------------------------------------")
print("Factor \u03c8(x_B, x_C): Favors x_B = x_C")

print(f"\nOutgoing Message \u03bc_{B \u2192 C}: {np.round(MU_B_TO_C, 3)}")
print(f"P(x_C=0): {MU_B_TO_C[0]:.3f}")
print(f"P(x_C=1): {MU_B_TO_C[1]:.3f}")

print("\nConclusion: Node B successfully processed its local evidence (\u03c8_{B,C}) and the incoming message (\u03bc_{A \u2192 B}) to compute a new outgoing message (\u03bc_{B \u2192 C}). The message is nearly uniform (0.5, 0.5) because the incoming message from A was uniform, demonstrating that global inference is achieved by iteratively passing and combining local beliefs.")
```

### Project 3: Modeling Structure with a Directed Bayesian Network (BN)

* **Goal:** Construct and analyze a simple BN to illustrate the flow of conditional probability (causality).
* **Setup:** Define three categorical variables: $A$ (Rain $\to$ 1), $B$ (Sprinkler $\to$ 1), $C$ (Wet Grass $\to$ 1). Define the dependencies: $A \to C$, $B \to C$.
* **Steps:**
    1.  Define the three CPTs ($P(A)$, $P(B)$, $P(C|A,B)$).
    2.  Calculate the **joint probability** of a specific event, e.g., $P(\text{Rain=1, Sprinkler=0, Wet Grass=1})$ using the BN's factorization rule.
* ***Goal***: Illustrate how the factored product of CPTs computes the probability of a complex system state, governed by a directed causal structure.

#### Python Implementation

```python
import numpy as np

# ====================================================================
# 1. Setup HMM Parameters (Two Hidden States: Cold=0, Hot=1)
# ====================================================================

# Hidden States: Cold (0), Hot (1)
# Observations: Low Energy (0), High Energy (1)

# 1. Transition Matrix (P(z_t | z_{t-1}))
# Rows: z_{t-1} (Start), Columns: z_t (End)
# Favors staying in the same state (P_Cold_to_Cold = 0.9)
A = np.array([
    [0.9, 0.1],  # Cold -> Cold (0.9), Cold -> Hot (0.1)
    [0.2, 0.8]   # Hot -> Cold (0.2), Hot -> Hot (0.8)
])

# 2. Observation Matrix (P(x_t | z_t))
# Rows: z_t (Hidden State), Columns: x_t (Observation)
# Cold state strongly predicts Low Energy, Hot state strongly predicts High Energy
B = np.array([
    [0.9, 0.1],  # Cold predicts Low E (0.9), High E (0.1)
    [0.3, 0.7]   # Hot predicts Low E (0.3), High E (0.7)
])

# Initial Probability (Prior belief at t=0)
PI = np.array([0.7, 0.3]) # Start with a strong belief in the Cold state

# ====================================================================
# 2. Forward Algorithm Implementation (State Estimation)
# ====================================================================

# Sequence of observations: Low E (0) -> High E (1)
# We track the belief \alpha_t at each step
Observations = [0, 1] 
Belief_History = [PI.copy()]

# The forward algorithm loop
belief = PI.copy() # Current belief P(z_t | x_1:t)

for t, x_t in enumerate(Observations):
    # --- 1. Prediction Step (Predict next state based on transition dynamics) ---
    # Prediction: P(z_t | x_1:t-1) = sum_{z_{t-1}} P(z_t | z_{t-1}) * P(z_{t-1} | x_1:t-1)
    # Predicted_belief = belief_t-1 @ A (matrix multiplication)
    predicted_belief = belief @ A
    
    # --- 2. Observation Update (Correct prediction with noisy data) ---
    # Update: P(z_t | x_1:t) \propto P(x_t | z_t) * Predicted_belief
    
    # Likelihood of observing x_t for each state z_t
    likelihood_x_t = B[:, x_t] 
    
    # Updated belief (unnormalized)
    unnorm_belief = predicted_belief * likelihood_x_t
    
    # Normalization (Crucial step for proper probability)
    belief = unnorm_belief / np.sum(unnorm_belief)
    
    Belief_History.append(belief)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

df_belief = pd.DataFrame(Belief_History, columns=['P(Cold)', 'P(Hot)'])
df_belief.index.name = 'Time Step'

print("--- HMM Forward Algorithm: State Estimation ---")
print(f"Observation Sequence: {Observations}")
print(df_belief.to_markdown(floatfmt=".3f"))

# Plot the evolution of belief
df_belief.plot(kind='line', style=['-o', '--s'], figsize=(8, 5))
plt.title('HMM Belief Evolution: P($z_t$ | $x_{1:t}$)')
plt.xlabel('Time Step (t)')
plt.ylabel('Belief Probability')
plt.xticks(np.arange(len(Belief_History)), labels=['t=0 (Prior)', 't=1 (Obs=0)', 't=2 (Obs=1)'])
plt.ylim(0, 1.0)
plt.grid(True)
plt.show()

print("\nConclusion: The belief system starts strongly Cold (0.7). After the second observation (x=1, High Energy), the belief in the Hot state increases sharply (from 0.3 to \u22480.64), demonstrating the core HMM task of updating the hidden state probability based on a sequence of noisy, external observations.")
```

### Project 4: Dynamic Model (HMM) for State Estimation

* **Goal:** Implement the logic of a **Hidden Markov Model (HMM)** to model a system where the true state is hidden.
* **Setup:** Model a simple physical system with two hidden states ($\text{Cold, Hot}$) and three noisy observations ($\text{Low Energy, Med Energy, High Energy}$). Define the transition matrix ($P(\text{Cold} \to \text{Hot})$) and the observation matrix ($P(\text{Low E} | \text{Cold})$).
* **Steps:**
    1.  Simulate a short sequence of observations (e.g., $\text{Low E} \to \text{Low E} \to \text{High E}$).
    2.  Implement the forward algorithm (or the Viterbi algorithm conceptually) to compute the **probability of the hidden state** at each step, $P(\mathbf{z}_t | \mathbf{x}_{1:t})$.
* ***Goal***: Show that after the noisy observations (e.g., $\text{High E}$), the marginal belief for the hidden state (the true energy) shifts sharply towards the $\text{Hot}$ state, demonstrating the core inference task of state estimation in dynamic systems.

#### Python Implementation

```python
import numpy as np

# ====================================================================
# 1. Setup Conceptual Functions
# ====================================================================

# We model the ELBO components conceptually to show the maximization logic.
# Assume the true model P is a known function of a single parameter \theta.

# True Model Parameters
TRUE_THETA = 5.0
DATA = 100.0 # Hypothetical summary statistic of the data

# 1. Energy Term (ln P(D, \theta))
# Conceptual Joint Likelihood: Penalizes deviation from the data (DATA)
def log_joint_likelihood(theta, data_summary):
    # Penalizes distance from data center (e.g., L2 loss)
    return -0.5 * (theta - data_summary)**2

# 2. Entropy Term (-ln Q(\theta))
# Conceptual Entropy for a simple Gaussian Q ~ N(\mu_Q, \sigma_Q)
# The Gaussian entropy is H(Q) = 0.5 * log(2\pi e \sigma_Q^2)
def entropy(sigma_q):
    # We use -H(Q) for -E_Q[ln Q] in the ELBO formula
    return 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_q**2)

# ====================================================================
# 2. ELBO Calculation and Optimization Logic
# ====================================================================

def calculate_elbo(mu_q, sigma_q, data_summary=DATA):
    """
    Conceptual ELBO for a Gaussian Q: ELBO = E_Q[ln P(D,\theta)] - E_Q[ln Q(\theta)]
    """
    # 1. Energy Term: E_Q [ln P(D,\theta)] - We approximate this with the likelihood at mu_Q
    # In a full VI, this is calculated with Monte Carlo sampling over Q.
    energy_term = log_joint_likelihood(mu_q, data_summary) 
    
    # 2. Entropy Term: E_Q [ln Q(\theta)] = -H(Q)
    # The term -E_Q[ln Q] is the negative entropy
    neg_entropy_term = -entropy(sigma_q)
    
    return energy_term - neg_entropy_term

# --- Optimization Scenario ---
# We track ELBO evolution as Q is optimized toward the true Posterior.

MU_Q_INIT = 0.0 # Initial guess for Q's mean
SIGMA_Q_INIT = 4.0 # Initial guess for Q's standard deviation (wide)

# We conceptualize the optimization:
# Step 1: Initial (Poor) Q
ELBO_INIT = calculate_elbo(MU_Q_INIT, SIGMA_Q_INIT)

# Step 2: Optimized (Better) Q
# The mean moves toward the data center (100) and the variance shrinks.
MU_Q_OPT = 90.0 
SIGMA_Q_OPT = 1.0 
ELBO_OPT = calculate_elbo(MU_Q_OPT, SIGMA_Q_OPT)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

elbo_values = [ELBO_INIT, ELBO_OPT]
names = ['Initial Q (Low ELBO)', 'Optimized Q (High ELBO)']

print("--- Variational Inference (VI) and ELBO Maximization ---")

# Plot ELBO evolution
plt.figure(figsize=(8, 5))
plt.bar(names, elbo_values, color=['skyblue', 'darkgreen'])
plt.title(r'ELBO Maximization: Inference as Optimization')
plt.ylabel('Evidence Lower Bound (ELBO)')
plt.grid(True, axis='y')
plt.show()

print("\nConclusion: The ELBO increases from the initial, uninformed distribution (Q_INIT) to the optimized distribution (Q_OPT). This demonstrates that **Variational Inference** solves the inference problem by framing it as a deterministic **maximization of the ELBO**, which is computationally equivalent to minimizing the statistical distance (KL divergence) between the approximation Q and the true Posterior P.")
``````
--- Variational Inference (VI) and ELBO Maximization ---
```