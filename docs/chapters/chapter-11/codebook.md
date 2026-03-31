# **Chapter 11: Graphical Models & Probabilistic Graphs (Codebook)**

## Project 1: Bayesian Network (BN) Factoring (Chain Dependency)

---

### Definition: Bayesian Network Factoring

The goal is to model a dependency structure using a simple **Bayesian Network (BN)** and calculate the **joint probability** of a specific system state using the **factoring rule**. This demonstrates how graph topology simplifies complex probability calculations.

### Theory: Conditional Independence and Factoring

A Bayesian Network represents the conditional dependencies between variables using a **Directed Acyclic Graph (DAG)**. The structure allows the complex joint probability to be factored into a product of simpler **conditional probability tables (CPTs)**:

$$P(X_1, X_2, \dots, X_N) = \prod_{i=1}^N P(X_i \mid \text{Parents}(X_i))$$

For the simple chain dependency $A \to B \to C$:

$$P(A, B, C) = P(A) P(B \mid A) P(C \mid B)$$

We compute the probability of the state $P(A=1, B=0, C=1)$ using predefined CPTs.

---

### Extensive Python Code

```python
import numpy as np
import pandas as pd

## ====================================================================

## 1. Setup Network and Conditional Probability Tables (CPTs)

## ====================================================================

## Dependency: A -> B -> C (A is root, C is leaf)

## Variables are binary: 0 (False) or 1 (True)

## P(A) - Prior for the root node

## Index [0] is P(A=0), Index [1] is P(A=1)

P_A = np.array([0.4, 0.6])

## P(B | A) - Conditional Probability Table (CPT)

## Rows: P(B | Parent)

## P_B_given_A[A_state, B_state]

P_B_given_A = np.array([
    [0.8, 0.2],  # P(B=0|A=0), P(B=1|A=0)
    [0.1, 0.9]   # P(B=0|A=1), P(B=1|A=1)
])

## P(C | B) - Conditional Probability Table (CPT)

## P_C_given_B[B_state, C_state]

P_C_given_B = np.array([
    [0.9, 0.1],  # P(C=0|B=0), P(C=1|B=0)
    [0.2, 0.8]   # P(C=0|B=1), P(C=1|B=1)
])

## ====================================================================

## 2. Joint Probability Calculation (Factoring Rule)

## ====================================================================

## Goal: Compute P(A=1, B=0, C=1)

## State Indices: A_idx=1, B_idx=0, C_idx=1

A_idx = 1
B_idx = 0
C_idx = 1

## 1. Term P(A=1)

Term_A = P_A[A_idx]

## 2. Term P(B=0 | A=1)

Term_B_given_A = P_B_given_A[A_idx, B_idx]

## 3. Term P(C=1 | B=0)

Term_C_given_B = P_C_given_B[B_idx, C_idx]

## Total Joint Probability P(A, B, C) = P(A) * P(B|A) * P(C|B)

P_joint = Term_A * Term_B_given_A * Term_C_given_B

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

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
```python
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

---

## Project 2: Belief Propagation (BP) Message Update

---

### Definition: Belief Propagation Message Update

The goal is to implement the core formula for the message-passing update in a **Factor Graph** (or equivalent **Markov Random Field**), demonstrating the mechanism of **Belief Propagation (BP)**.

### Theory: Local Message Passing

Belief Propagation is an iterative inference algorithm that computes the marginal probability of each variable by passing **messages** (local estimates of belief) between neighboring nodes.

The message $\mathcal{\mu}_{i \to j}$ sent from node $i$ to neighbor $j$ is proportional to the product of local evidence ($\psi_i$) and all incoming messages from $i$'s other neighbors ($\mathcal{N}(i) \setminus \{j\}$):

$$\mu_{i \to j}(x_j) \propto \sum_{x_i} \psi_{i}(x_i, x_j) \prod_{k \in \mathcal{N}(i) \setminus \{j\}} \mu_{k \to i}(x_i)$$

This formula is the computational backbone of collective inference, where global consensus is achieved through local communication.

---

### Extensive Python Code

```python
import numpy as np

## ====================================================================

## 1. Setup Network and Initial Data

## ====================================================================

## Network: A - B - C (Node B is calculating the message to C)

## Variables are binary: x_A, x_B, x_C \in {0, 1}

## --- Node B's Local Evidence (Factor \psi_{B,C}) ---

## This is the CPT-like factor \psi(x_B, x_C) or the edge potential

## We simplify by using an edge potential that favors x_B == x_C

## Rows (x_B), Columns (x_C)

FACTOR_B_C = np.array([
    [0.9, 0.1],  # x_B=0 favors x_C=0 (90%)
    [0.1, 0.9]   # x_B=1 favors x_C=1 (90%)
])

## --- Incoming Message to B from A (\mu_{A \to B}) ---

## This message is B's current belief about A's state

## P(x_B=0), P(x_B=1) - Uniform prior for the next iteration

MU_A_TO_B = np.array([0.5, 0.5])

## ====================================================================

## 2. Belief Propagation Update

## ====================================================================

## Goal: Calculate the outgoing message from B to C: \mu_{B \to C}(x_C)

## Message formula: \mu_{B \to C}(x_C) \propto \sum_{x_B} \psi(x_B, x_C) * \mu_{A \to B}(x_B)

## The outgoing message \mu_{B \to C} will be a vector of size 2 (for x_C=0 and x_C=1)

MU_B_TO_C = np.zeros(2)

## Loop over the target variable x_C (index 0 and 1)

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

## Normalize the final message (since it's only proportional)

MU_B_TO_C /= np.sum(MU_B_TO_C)

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

print("--- Belief Propagation Message Calculation (\u03bc_{B \u2192 C}) ---")
print(f"Incoming Message from A (\u03bc_{A \u2192 B}): {np.round(MU_A_TO_B, 3)}")
print("---------------------------------------------------------------")
print("Factor \u03c8(x_B, x_C): Favors x_B = x_C")

print(f"\nOutgoing Message \u03bc_{B \u2192 C}: {np.round(MU_B_TO_C, 3)}")
print(f"P(x_C=0): {MU_B_TO_C[0]:.3f}")
print(f"P(x_C=1): {MU_B_TO_C[1]:.3f}")

print("\nConclusion: Node B successfully processed its local evidence (\u03c8_{B,C}) and the incoming message (\u03bc_{A \u2192 B}) to compute a new outgoing message (\u03bc_{B \u2192 C}). The message is nearly uniform (0.5, 0.5) because the incoming message from A was uniform, demonstrating that global inference is achieved by iteratively passing and combining local beliefs.")
```

---

## Project 3: Dynamic Model (HMM) for State Estimation

---

### Definition: Dynamic Model (HMM) for State Estimation

The goal is to implement the core logic of the **Forward Algorithm** in a **Hidden Markov Model (HMM)** to solve the inference task of **state estimation**. This is crucial for modeling dynamic systems where the underlying state ($\mathbf{z}$) is hidden from noisy observations ($\mathbf{x}$).

### Theory: Forward Algorithm and State Belief

An HMM models a sequence of hidden states ($\mathbf{z}_t$) connected by a **transition matrix** $P(\mathbf{z}_t \mid \mathbf{z}_{t-1})$ and generating observable outputs ($\mathbf{x}_t$) via an **observation matrix** $P(\mathbf{x}_t \mid \mathbf{z}_t)$.

The **Forward Algorithm** iteratively computes the forward probability vector ($\mathcal{\alpha}_t$), which is the belief over the hidden state at time $t$ given all observations up to that point:

$$\mathcal{\alpha}_t(\mathbf{z}_t) = P(\mathbf{z}_t, \mathbf{x}_{1:t}) \propto \text{Observation Likelihood} \times \text{Transition Prediction}$$

The prediction step updates the prior belief based on the system's known transition dynamics, and the observation step corrects this prediction based on the noisy measurement.

---

### Extensive Python Code

```python
import numpy as np

## ====================================================================

## 1. Setup HMM Parameters (Two Hidden States: Cold=0, Hot=1)

## ====================================================================

## Hidden States: Cold (0), Hot (1)

## Observations: Low Energy (0), High Energy (1)

## 1. Transition Matrix (P(z_t | z_{t-1}))

## Rows: z_{t-1} (Start), Columns: z_t (End)

## Favors staying in the same state (P_Cold_to_Cold = 0.9)

A = np.array([
    [0.9, 0.1],  # Cold -> Cold (0.9), Cold -> Hot (0.1)
    [0.2, 0.8]   # Hot -> Cold (0.2), Hot -> Hot (0.8)
])

## 2. Observation Matrix (P(x_t | z_t))

## Rows: z_t (Hidden State), Columns: x_t (Observation)

## Cold state strongly predicts Low Energy, Hot state strongly predicts High Energy

B = np.array([
    [0.9, 0.1],  # Cold predicts Low E (0.9), High E (0.1)
    [0.3, 0.7]   # Hot predicts Low E (0.3), High E (0.7)
])

## Initial Probability (Prior belief at t=0)

PI = np.array([0.7, 0.3]) # Start with a strong belief in the Cold state

## ====================================================================

## 2. Forward Algorithm Implementation (State Estimation)

## ====================================================================

## Sequence of observations: Low E (0) -> High E (1)

## We track the belief \alpha_t at each step

Observations = [0, 1]
Belief_History = [PI.copy()]

## The forward algorithm loop

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

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

df_belief = pd.DataFrame(Belief_History, columns=['P(Cold)', 'P(Hot)'])
df_belief.index.name = 'Time Step'

print("--- HMM Forward Algorithm: State Estimation ---")
print(f"Observation Sequence: {Observations}")
print(df_belief.to_markdown(floatfmt=".3f"))

## Plot the evolution of belief

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

---

## Project 4: Variational Inference (VI) via ELBO Maximization

---

### Definition: Variational Inference (VI) via ELBO Maximization

The goal is to model the **Variational Inference (VI)** approach as an optimization problem. This is the advanced alternative to MCMC, where a simpler distribution $Q$ is optimized by maximizing the **Evidence Lower Bound (ELBO)**.

### Theory: ELBO and Optimization Duality

VI reframes the intractable calculation of the true Posterior $P(\mathcal{\theta} \mid \mathcal{D})$ into a tractable **optimization problem**. We approximate $P$ with a simpler distribution $Q(\mathcal{\theta})$ by minimizing the **Kullback-Leibler (KL) Divergence** $D_{\mathrm{KL}}(Q||P)$.

Minimizing the KL divergence is analytically equivalent to maximizing the **Evidence Lower Bound (ELBO)**:

$$\text{ELBO} = \mathbb{E}_Q [\ln P(\mathcal{D}, \mathcal{\theta})] - \mathbb{E}_Q [\ln Q(\mathcal{\theta})]$$

  * **ELBO Components:** The ELBO has two terms that act as a dual: the **Energy Term** ($\mathbb{E}_Q [\ln P(\mathcal{D}, \mathcal{\theta})]$) and the **Entropy Term** ($\mathbb{E}_Q [-\ln Q(\mathcal{\theta})]$).
  * **Optimization:** The VI algorithm maximizes the ELBO iteratively, confirming that **inference is a form of energy minimization**.

---

### Extensive Python Code

```python
import numpy as np

## ====================================================================

## 1. Setup Conceptual Functions

## ====================================================================

## We model the ELBO components conceptually to show the maximization logic.

## Assume the true model P is a known function of a single parameter \theta.

## True Model Parameters

TRUE_THETA = 5.0
DATA = 100.0 # Hypothetical summary statistic of the data

## 1. Energy Term (ln P(D, \theta))

## Conceptual Joint Likelihood: Penalizes deviation from the data (DATA)

def log_joint_likelihood(theta, data_summary):
    # Penalizes distance from data center (e.g., L2 loss)
    return -0.5 * (theta - data_summary)**2

## 2. Entropy Term (-ln Q(\theta))

## Conceptual Entropy for a simple Gaussian Q ~ N(\mu_Q, \sigma_Q)

## The Gaussian entropy is H(Q) = 0.5 * log(2\pi e \sigma_Q^2)

def entropy(sigma_q):
    # We use -H(Q) for -E_Q[ln Q] in the ELBO formula
    return 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_q**2)

## ====================================================================

## 2. ELBO Calculation and Optimization Logic

## ====================================================================

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

## --- Optimization Scenario ---

## We track ELBO evolution as Q is optimized toward the true Posterior.

MU_Q_INIT = 0.0 # Initial guess for Q's mean
SIGMA_Q_INIT = 4.0 # Initial guess for Q's standard deviation (wide)

## We conceptualize the optimization:

## Step 1: Initial (Poor) Q

ELBO_INIT = calculate_elbo(MU_Q_INIT, SIGMA_Q_INIT)

## Step 2: Optimized (Better) Q

## The mean moves toward the data center (100) and the variance shrinks.

MU_Q_OPT = 90.0
SIGMA_Q_OPT = 1.0
ELBO_OPT = calculate_elbo(MU_Q_OPT, SIGMA_Q_OPT)

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

elbo_values = [ELBO_INIT, ELBO_OPT]
names = ['Initial Q (Low ELBO)', 'Optimized Q (High ELBO)']

print("--- Variational Inference (VI) and ELBO Maximization ---")

## Plot ELBO evolution

plt.figure(figsize=(8, 5))
plt.bar(names, elbo_values, color=['skyblue', 'darkgreen'])
plt.title(r'ELBO Maximization: Inference as Optimization')
plt.ylabel('Evidence Lower Bound (ELBO)')
plt.grid(True, axis='y')
plt.show()

print("\nConclusion: The ELBO increases from the initial, uninformed distribution (Q_INIT) to the optimized distribution (Q_OPT). This demonstrates that **Variational Inference** solves the inference problem by framing it as a deterministic **maximization of the ELBO**, which is computationally equivalent to minimizing the statistical distance (KL divergence) between the approximation Q and the true Posterior P.")
``````python
--- Variational Inference (VI) and ELBO Maximization ---
```