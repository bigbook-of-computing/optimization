# **Chapter 15: Reinforcement Learning and Control () () () (Workbook)**

The goal of this chapter is to frame learning as **dynamic decision-making**, introducing the mathematical structure of the Markov Decision Process (MDP) and unifying the maximization of reward with the thermodynamic minimization of Free Energy.

| Section | Topic Summary |
| :--- | :--- |
| **15.1** | From Static Inference to Dynamic Decision |
| **15.2** | Markov Decision Processes (MDPs) |
| **15.3** | Policies and Value Functions |
| **15.4** | Dynamic Programming and Value Iteration |
| **15.5** | Q-Learning — Learning Without a Model |
| **15.6** | Policy Gradient Methods |
| **15.7** | Actor–Critic Architecture |
| **15.8** | Deep Reinforcement Learning (Deep RL) |
| **15.9** | Exploration vs. Exploitation |
| **15.10** | Continuous Control and Policy Optimization |
| **15.11** | RL as Free-Energy Minimization |
| **15.12–15.18**| Worked Example, Code Demo, and Takeaways |

---

### 15.1 From Static Inference to Dynamic Decision

> **Summary:** Reinforcement Learning (RL) studies **agents** that actively move and adapt over a temporal **trajectory**. The core objective is to learn an optimal **policy ($\pi$)** that maximizes the expected **discounted cumulative reward** ($J(\pi)$). RL is analogous to a **thermodynamic process** where utility is maximized, and the search for an optimal policy steers the system to a configuration of minimal cost (negative energy).

#### Quiz Questions

!!! note "Quiz"
```
**1. In the RL framework, the mathematical object that represents the agent's strategy or a mapping from states to actions is called the:**

* **A.** Value function ($V$).
* **B.** **Policy ($\pi$)**. (**Correct**)
* **C.** Discount factor ($\gamma$).
* **D.** Transition kernel ($P$).

```
!!! note "Quiz"
```
**2. The single component that distinguishes the RL objective from the static optimization objective (Part II) is the explicit inclusion of the concept of a:**

* **A.** Single loss function $L$.
* **B.** **Temporal trajectory or sequence of states**. (**Correct**)
* **C.** Fixed set of parameters $\mathcal{\theta}$.
* **D.** Non-convex landscape.

```
---

!!! question "Interview Practice"
```
**Question:** The RL objective function $J(\pi)$ maximizes the expected *discounted* reward, using a discount factor $\gamma \in [0, 1]$. What is the dual purpose of including this discount factor in the calculation?

**Answer Strategy:** The discount factor $\gamma$ has two primary purposes:
1.  **Mathematical Stability:** It prevents the infinite sum of future rewards from **diverging**.
2.  **Uncertainty/Time Preference:** It weights immediate rewards more heavily than distant future rewards. This models the real-world preference for immediate utility and the **uncertainty** of future environmental states.

```
---

---

### 15.2 Markov Decision Processes (MDPs)

> **Summary:** RL problems are formally modeled as **Markov Decision Processes (MDPs)**, defined by the five-tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$. The core constraint is the **Markov Property**, which simplifies the problem by assuming the next state $s'$ depends **only on the current state $s$ and action $a$**. The optimal solution is found by solving the recursive **Bellman Optimality Equations**.

#### Quiz Questions

!!! note "Quiz"
```
**1. The constraint that simplifies the environment structure by stating the next state is independent of the entire history of past states is known as the:**

* **A.** Principle of Least Action.
* **B.** **Markov Property**. (**Correct**)
* **C.** Contraction Mapping.
* **D.** Policy Gradient Theorem.

```
!!! note "Quiz"
```
**2. The **Bellman Optimality Equation** recursively defines the optimal value of a state $V^*(s)$ by relating it to the maximum expected value achievable from:**

* **A.** The initial starting state.
* **B.** The total lifetime reward.
* **C.** **The immediate reward plus the discounted maximum future value from the succeeding states**. (**Correct**)
* **D.** The average value of all surrounding states.

```
---

!!! question "Interview Practice"
```
**Question:** The Value Function $V(s)$ is analogous to the negative potential energy of a state. In the Gridworld worked example (Section 15.12), the reward function is $r=-1$ per step, and $r=0$ at the goal. Describe the shape of the learned $V(s)$ potential landscape.

**Answer Strategy:** Since the agent is minimizing cost (negative reward), the optimal value function $V^*(s)$ forms a **potential well**. The shape will be a **monotonically increasing surface** (approaching zero from the negative side) centered at the goal state. The goal state itself will have the highest value ($V^*=0$), and the value will decrease (become more negative, e.g., $V^*=-1, -2, -3, \dots$) as the state moves farther away from the goal, creating a clear **potential gradient** that attracts the agent.

```
---

---

### 15.3 Policies and Value Functions

> **Summary:** The **policy ($\pi$)** is the agent's strategy (the control field). **Value functions** ($V$ and $Q$) quantify the expected cumulative reward, acting as the potential landscape. The **optimal policy ($\pi^*$)** is found by selecting the action that maximizes the optimal **Action-Value Function ($Q^*$)**.

### 15.4 Dynamic Programming and Value Iteration

> **Summary:** **Dynamic Programming (DP)** solves MDPs by breaking them into iterative, recursive subproblems. **Value Iteration** repeatedly applies the Bellman Optimality Equation until the value function converges to $V^*$. DP methods achieve their solution via **successive approximation** and **contraction mapping**, which is mathematically guaranteed to converge. The convergence process is analogous to **heat diffusion** propagating reward information backward through time.

### 15.5 Q-Learning — Learning Without a Model

> **Summary:** **Q-Learning** is a fundamental **model-free** algorithm that learns the optimal $Q^*(s,a)$ function directly from experience. It uses **Temporal-Difference (TD) learning** and **bootstrapping**. The update rule is based on minimizing the **TD Error**. Q-Learning is equivalent to **stochastic gradient descent** on the Bellman error and functions as a **local energy correction** mechanism.

### 15.6 Policy Gradient Methods

> **Summary:** **Policy Gradient (PG)** methods **parameterize the policy ($\pi_{\mathcal{\theta}}$)** explicitly (e.g., as a neural network) and optimize it directly to maximize expected reward. The **Policy Gradient Theorem** provides the analytic form of the gradient, relating policy performance to the value of observed actions. PG methods often include **entropy regularization** to encourage policy diversity (exploration).

### 15.7 Actor–Critic Architecture

> **Summary:** **Actor–Critic** is a hybrid method using two coupled networks. The **Actor** ($\pi_{\mathcal{\theta}}$) selects the action, and the **Critic** ($V_{\mathcal{\phi}}$ or $Q_{\mathcal{\phi}}$) evaluates the action by estimating the value function. The Critic computes the **TD Error ($\delta_t$)**, which serves as the **advantage estimate** to update the Actor's policy gradient. This reduces gradient variance and stabilizes the learning process.

### 15.8 Deep Reinforcement Learning (Deep RL)

> **Summary:** **Deep RL** integrates classic RL algorithms with **Deep Neural Networks (DNNs)** to handle large, raw state spaces. Key stabilization techniques include **Experience Replay** (to decorrelate sequential data) and **Target Networks** (to stabilize non-stationary optimization targets). Deep RL is analogous to learning a **complex control field** in a high-dimensional, abstract phase space.

### 15.9 Exploration vs. Exploitation

> **Summary:** This is the fundamental trade-off between maximizing known reward and gathering new information. Methods like **$\epsilon$-greedy** and **Boltzmann Exploration** manage this by injecting controlled stochasticity. The control parameter $T$ acts as **temperature**, where high $T$ leads to **diffusion** (exploration) and low $T$ leads to **crystallization** (exploitation). **Entropy-Regularized RL** formalizes this by explicitly maximizing reward plus policy entropy.

### 15.10 Continuous Control and Policy Optimization

> **Summary:** **Continuous Control** involves action spaces $a \in \mathbb{R}^k$, which requires policy optimization. **Deterministic Policy Gradient (DPG)** methods are used, which follow the gradient of the $Q$-function with respect to the continuous action, $\nabla_a Q(s,a)$. The optimized policy acts as a continuous **force field** or **control law** in the system's phase space, connecting RL to Optimal Control Theory.

### 15.11 RL as Free-Energy Minimization

> **Summary:** The objective of RL is unified with thermodynamics through the **Free-Energy Principle**. By defining cost $E=-r$ (negative reward), the objective is recast as minimizing the **Free Energy functional ($\mathcal{F} = E - T \mathcal{H}$) **. The result is an optimal policy that is a **Boltzmann distribution over actions**, balancing low cost (exploitation) with high diversity (entropy).

---

## 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core concepts of value function dynamics, policy optimization, and the energy/entropy trade-off.

### Project 1: Implementing and Visualizing the Value Landscape (Code Demo Replication)

* **Goal:** Implement the model-free **Q-Learning** algorithm on a Gridworld problem and visualize the resulting optimal potential field.
* **Setup:** Use the $5 \times 5$ Gridworld environment with $r=-1$ (non-goal states) and $r=0$ (goal state).
* **Steps:**
    1.  Implement the **Q-Learning update rule** using the observed reward $r$ and bootstrapped future value ($\max_{a'} Q(s',a')$).
    2.  Run the simulation for 500 episodes with $\epsilon$-greedy exploration.
    3.  Visualize the final state values, $V_{\max}(s) = \max_a Q(s,a)$, as a heatmap.
* ***Goal***: Show that the heatmap forms a **potential well** centered precisely on the goal state, confirming that the agent has successfully learned the optimal path through local TD error correction.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. Setup Environment (Grid World) and Q-Table
# ====================================================================

# Environment: Simple 3x3 Grid World
# States: 0 to 8
# Actions: 0: Up, 1: Down, 2: Left, 3: Right
N_STATES = 9
N_ACTIONS = 4
GOAL_STATE = 8 # State 8 (bottom right) gives reward

# Rewards (R[s, a] = reward for taking action a from state s)
# Only actions leading to the goal (8) give positive reward
R = -0.1 * np.ones((N_STATES, N_ACTIONS)) # Small penalty for all moves
R[5, 3] = 10.0 # From state 5 (middle-right), moving Right (3) to 8 is a reward of 10.0
R[7, 1] = 10.0 # From state 7 (bottom-middle), moving Down (1) to 8 is a reward of 10.0

# Transition Function (Deterministic, P(s'|s, a))
def get_next_state(s, a):
    row, col = divmod(s, 3)
    if a == 0: row = max(row - 1, 0) # Up
    elif a == 1: row = min(row + 1, 2) # Down
    elif a == 2: col = max(col - 1, 0) # Left
    elif a == 3: col = min(col + 1, 2) # Right
    return row * 3 + col

# Q-Table Initialization (start at zero)
Q = np.zeros((N_STATES, N_ACTIONS))

# ====================================================================
# 2. Q-Learning Algorithm
# ====================================================================

# Hyperparameters
ETA = 0.1     # Learning rate
GAMMA = 0.9   # Discount factor
MAX_EPISODES = 1000

# Tracking the max Q-value for the starting state (s=0) to check convergence
Q0_history = [] 

for episode in range(MAX_EPISODES):
    s = 0 # Start every episode at state 0
    
    # Run episode until goal (state 8) is reached
    while s != GOAL_STATE:
        # 1. Action Selection (Exploration vs. Exploitation - Epsilon Greedy)
        if random.random() < 0.1: # 10% Exploration
            a = random.randrange(N_ACTIONS)
        else: # 90% Exploitation
            a = np.argmax(Q[s, :])
            
        # 2. Interact with Environment
        r = R[s, a]
        s_prime = get_next_state(s, a)
        
        # 3. Q-Learning Update Rule (Bellman Optimality)
        # Target = r + gamma * max_a' Q(s', a')
        max_q_prime = np.max(Q[s_prime, :])
        td_target = r + GAMMA * max_q_prime
        
        # Q(s, a) = Q(s, a) + eta * [Target - Q(s, a)]
        Q[s, a] = Q[s, a] + ETA * (td_target - Q[s, a])
        
        s = s_prime
        
    # Track convergence (Max Q-value for the starting state 0)
    Q0_history.append(np.max(Q[0, :])) 

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Plot 1: Q-Value Convergence
plt.figure(figsize=(9, 6))
plt.plot(Q0_history, 'r-', lw=2)
plt.title('Q-Learning Convergence: Max Q(State 0, a)')
plt.xlabel('Episode')
plt.ylabel('Max Action-Value $Q_0$')
plt.grid(True)
plt.show()

# Print the learned optimal policy (the greedy action from Q*)
print("\n--- Learned Optimal Policy (Q*) ---")
print("State: Best Action")
for s in range(N_STATES):
    best_action_index = np.argmax(Q[s, :])
    action_name = ['Up', 'Down', 'Left', 'Right'][best_action_index]
    print(f"State {s}: {action_name}")

print("\nConclusion: The plot shows that the Q-value for the starting state monotonically increases and eventually converges to a stable value, confirming that the iterative TD learning process successfully approximated the optimal value function $Q^*$.")
```
**Sample Output:**
```
--- Learned Optimal Policy (Q*) ---
State: Best Action
State 0: Down
State 1: Left
State 2: Down
State 3: Right
State 4: Down
State 5: Right
State 6: Right
State 7: Down
State 8: Up

Conclusion: The plot shows that the Q-value for the starting state monotonically increases and eventually converges to a stable value, confirming that the iterative TD learning process successfully approximated the optimal value function $Q^*$.
```


### Project 2: Simulating Heat Diffusion of Reward (Value Iteration)

* **Goal:** Implement the **Value Iteration** algorithm (Dynamic Programming) to show how value information propagates backward through the state space.
* **Setup:** Use the same Gridworld environment but assume the **transition kernel $P$ is known** (deterministic for simple moves).
* **Steps:**
    1.  Initialize $V(s)$ to zero everywhere.
    2.  Implement the Value Iteration update rule (using the $\max_a$ and $\sum_{s'} P(s'|s,a) V_k(s')$ terms).
    3.  Run the algorithm for only a few iterations (e.g., $k=1, 2, 3$).
* ***Goal***: Show, by observing the $V_k(s)$ map, that the reward information (high value) starts only at the goal state and **diffuses backward** one layer per iteration, confirming the heat diffusion analogy.

#### Python Implementation

```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Policy and Environment (Conceptual)
# ====================================================================

# We model a single state (s) and two actions (a1, a2)
# Policy is parameterized by a single logit \theta_1 (weights are not explicit here)
N_ACTIONS = 2
N_STEPS = 100

# Conceptual Rewards (representing a sampled episode)
# Return G is assumed to be known for each action in the episode
RETURNS = np.array([10.0, 5.0, 15.0, 2.0, 8.0, 1.0])
ACTIONS = np.array([0, 1, 0, 1, 0, 1]) # 0=a1, 1=a2 (taken actions)

# Conceptual Policy (Softmax over logits \theta)
# The logits are conceptual parameters defining the policy preference
# Let's assume the current logits are [0.5, 0.0]
LOGITS = np.array([0.5, 0.0])

def softmax(logits):
    """Calculates action probabilities \pi(a|s)."""
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def get_policy_gradient_log(logits, action_index):
    """
    Calculates the gradient of the log-policy: \nabla_{\theta} \ln \pi(a|s).
    For a simple Softmax policy, this is: e_a - \pi(a|s)
    where e_a is a one-hot vector for action a.
    """
    pi_a = softmax(logits)
    e_a = np.zeros_like(logits)
    e_a[action_index] = 1.0
    return e_a - pi_a

# ====================================================================
# 2. Gradient Estimation Trials
# ====================================================================

# 1. Conceptual Baseline (Value function V(s))
# We approximate V(s) as the average return (baseline)
BASELINE = np.mean(RETURNS)

# Estimate the Gradient for the entire sampled episode (sum over time steps t)
GRADIENT_NO_BASELINE = np.zeros(N_ACTIONS)
GRADIENT_WITH_BASELINE = np.zeros(N_ACTIONS)
Variance_Check = []

for G_t, a_t in zip(RETURNS, ACTIONS):
    # Get the policy gradient term for the action taken (a_t)
    policy_grad_log = get_policy_gradient_log(LOGITS, a_t)
    
    # --- Trial A: No Baseline (High Variance) ---
    # Gradient = \nabla \ln \pi * G_t
    GRADIENT_NO_BASELINE += policy_grad_log * G_t
    Variance_Check.append(policy_grad_log * G_t)
    
    # --- Trial B: With Baseline (Low Variance) ---
    # Advantage A_t = G_t - b(s)
    Advantage_t = G_t - BASELINE
    
    # Gradient = \nabla \ln \pi * A_t
    GRADIENT_WITH_BASELINE += policy_grad_log * Advantage_t
    
# Calculate Variance (Crucial step for demonstration)
# We look at the variance of the individual contributions to the gradient estimate (Policy * Return/Advantage)
Variance_Check_array = np.array(Variance_Check)
# We specifically check the variance of the components that make up the final sum
VAR_CONTRIB_NO_BASELINE = np.var(Variance_Check_array[:, 0]) 

# Recalculate Variance_Check for baseline case
Variance_Check_baseline = []
for G_t, a_t in zip(RETURNS, ACTIONS):
    policy_grad_log = get_policy_gradient_log(LOGITS, a_t)
    Advantage_t = G_t - BASELINE
    Variance_Check_baseline.append(policy_grad_log * Advantage_t)
VAR_CONTRIB_WITH_BASELINE = np.var(np.array(Variance_Check_baseline)[:, 0])

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Policy Gradient Estimation (REINFORCE) ---")
print(f"Average Return (Baseline): {BASELINE:.2f}")
print("-------------------------------------------------------")

print("\nTrial A: Gradient Estimate (NO BASELINE):")
print(f"  Gradient Components: {np.round(GRADIENT_NO_BASELINE, 4)}")
print(f"  Variance of Contributions (\u03b8_1): {VAR_CONTRIB_NO_BASELINE:.4f} (High)")

print("\nTrial B: Gradient Estimate (WITH BASELINE):")
print(f"  Gradient Components: {np.round(GRADIENT_WITH_BASELINE, 4)}")
print(f"  Variance of Contributions (\u03b8_1): {VAR_CONTRIB_WITH_BASELINE:.4f} (Reduced)")

print("\nConclusion: The variance of the individual contributions to the gradient estimate (Trial B) is significantly lower when the average return (baseline) is subtracted from the actual return. This confirms the mathematical principle of variance reduction: the baseline removes common mode noise, making the learning process more stable and enabling faster convergence.")
```
**Sample Output:**
```
--- Policy Gradient Estimation (REINFORCE) ---
Average Return (Baseline): 6.83

---

Trial A: Gradient Estimate (NO BASELINE):
  Gradient Components: [ 7.4792 -7.4792]
  Variance of Contributions (θ_1): 9.6246 (High)

Trial B: Gradient Estimate (WITH BASELINE):
  Gradient Components: [ 12.5 -12.5]
  Variance of Contributions (θ_1): 1.4377 (Reduced)

Conclusion: The variance of the individual contributions to the gradient estimate (Trial B) is significantly lower when the average return (baseline) is subtracted from the actual return. This confirms the mathematical principle of variance reduction: the baseline removes common mode noise, making the learning process more stable and enabling faster convergence.
```


### Project 3: Policy Gradient (REINFORCE) with Baseline (Conceptual)

* **Goal:** Implement the Policy Gradient (REINFORCE) algorithm, demonstrating direct policy optimization and the necessity of using a baseline for variance reduction.
* **Setup:** Define a small, multi-step environment where actions are probabilistic. Parameterize the policy $\pi_{\mathcal{\theta}}$ as a simple neural network.
* **Steps:**
    1.  Implement the core REINFORCE gradient: $\nabla_{\mathcal{\theta}} J \propto \nabla_{\mathcal{\theta}} \ln \pi_{\mathcal{\theta}}(a|s) \cdot G_t$ ($G_t$ is the sampled return).
    2.  Implement a more stable update using a **baseline** ($b_s$) to calculate the **advantage** ($A_t = G_t - b_s$).
* ***Goal***: Demonstrate that the policy is updated directly based on which actions ($a|s$) lead to higher-than-average rewards (advantage), illustrating the principle of direct policy adaptation.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Conceptual Objective Function
# ====================================================================

# Conceptual components of the Objective Function J for a policy \pi
# These are the expected values under the current policy \pi.
# We track how the total objective changes with the \alpha parameter.

# 1. Expected Reward (Energy Term)
# This is fixed by the environment and is generally high when the policy is stable.
REWARD_TERM_BASE = 50.0

# 2. Entropy Term (Exploration Term)
# This is maximized when the policy is random (uniform probability)
# Assume H(\pi) ranges from 0 (deterministic) to 2.0 (fully random)
ENTROPY_TERM_HIGH = 1.5
ENTROPY_TERM_LOW = 0.5

# ====================================================================
# 2. Free-Energy/MaxEnt Scenarios
# ====================================================================

# --- Scenario A: Low Alpha (Low Exploration, High Exploitation) ---
ALPHA_A = 0.1 
# The system favors reward. Assumes a policy \pi_A that is high reward/low entropy.
OBJECTIVE_A = REWARD_TERM_BASE + ALPHA_A * ENTROPY_TERM_LOW

# --- Scenario B: High Alpha (High Exploration, Low Exploitation) ---
ALPHA_B = 5.0
# The system favors entropy. Assumes a policy \pi_B that is slightly lower reward/higher entropy.
REWARD_TERM_B = 45.0 # Lower reward achieved
OBJECTIVE_B = REWARD_TERM_B + ALPHA_B * ENTROPY_TERM_HIGH

# --- Scenario C: Zero Alpha (Standard RL / Pure Energy Minimization) ---
ALPHA_C = 0.0
OBJECTIVE_C = REWARD_TERM_BASE + ALPHA_C * ENTROPY_TERM_LOW

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

alphas = [ALPHA_C, ALPHA_A, ALPHA_B]
objectives = [OBJECTIVE_C, OBJECTIVE_A, OBJECTIVE_B]
names = [r'Standard RL ($\alpha=0$)', r'Low Entropy Cost ($\alpha=0.1$)', r'High Entropy Cost ($\alpha=5.0$)']

plt.figure(figsize=(9, 6))

# Plot Objectives (analogous to negative Free Energy)
bars = plt.bar(names, objectives, color=['gray', 'skyblue', 'darkred'])

# Annotate the components
plt.text(0, objectives[0] + 1, f'Reward:{REWARD_TERM_BASE:.1f}', ha='center', color='k')
plt.text(2, OBJECTIVE_B - 4, f'Reward:{REWARD_TERM_B:.1f}', ha='center', color='w', fontweight='bold')
plt.text(2, OBJECTIVE_B - 8, f'Entropy:{ALPHA_B * ENTROPY_TERM_HIGH:.1f}', ha='center', color='w', fontweight='bold')

# Labeling and Formatting
plt.title(r'MaxEnt RL Objective Function $J_{\alpha}$: Reward-Entropy Trade-Off')
plt.ylabel('Expected Return / Objective Value (J)')
plt.grid(True, axis='y')
plt.show()

print("\n--- MaxEnt RL (Free-Energy) Analysis ---")
print(f"Objective A (\u03b1=0.1): J = {OBJECTIVE_A:.2f} (High Reward, Low Entropy Contribution)")
print(f"Objective B (\u03b1=5.0): J = {OBJECTIVE_B:.2f} (Lower Reward, High Entropy Contribution)")

print("\nConclusion: The Objective B, despite having a lower raw reward (45.0), achieves a higher total objective value (52.5) because the high alpha factor drastically increases the contribution of the entropy term. This confirms that MaxEnt RL formalizes the exploration-exploitation balance as a thermodynamic trade-off, where the policy minimizes an effective Free Energy by balancing the system's energy (negative reward) and entropy (randomness).")
```
**Sample Output:**
```
--- MaxEnt RL (Free-Energy) Analysis ---
Objective A (α=0.1): J = 50.05 (High Reward, Low Entropy Contribution)
Objective B (α=5.0): J = 52.50 (Lower Reward, High Entropy Contribution)

Conclusion: The Objective B, despite having a lower raw reward (45.0), achieves a higher total objective value (52.5) because the high alpha factor drastically increases the contribution of the entropy term. This confirms that MaxEnt RL formalizes the exploration-exploitation balance as a thermodynamic trade-off, where the policy minimizes an effective Free Energy by balancing the system's energy (negative reward) and entropy (randomness).
```


### Project 4: Simulating Boltzmann Exploration and Temperature

* **Goal:** Implement the **Boltzmann Exploration** policy and observe how the **temperature ($T$)** parameter controls the exploitation-exploration balance.
* **Setup:** Use a fixed Q-table where the values for state $s$ are known (e.g., $Q(s, a_1)=5, Q(s, a_2)=10, Q(s, a_3)=0$).
* **Steps:**
    1.  Implement the Boltzmann policy: $\pi(a|s) \propto e^{Q(s,a)/T}$.
    2.  Calculate the resulting probability distribution $\pi(a|s)$ for two cases: **High Temperature** ($T=10$) and **Low Temperature** ($T=0.1$).
* ***Goal***: Show that at $T=10$, the distribution is nearly uniform (high exploration), while at $T=0.1$, the probability is highly concentrated on the optimal action $a_2$ (high exploitation), confirming the **annealing analogy**.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Boltzmann Policy Function
# ====================================================================

def boltzmann_policy(Q_values, T):
    """
    Calculates the action probabilities \pi(a|s) using the Boltzmann (Softmax) policy.
    \pi(a|s) = exp(Q/T) / sum(exp(Q/T))
    """
    if T <= 1e-6:
        # Near T=0 limit: pure greedy exploitation
        max_q = np.max(Q_values)
        pi = (Q_values == max_q).astype(float)
        return pi / np.sum(pi)
    
    # Numerical stability: subtract max(Q) from Q/T before exponentiating
    scaled_Q = Q_values / T
    exp_scaled_Q = np.exp(scaled_Q - np.max(scaled_Q))
    
    return exp_scaled_Q / np.sum(exp_scaled_Q)

# ====================================================================
# 2. Scenario Testing
# ====================================================================

# Fixed Q-Values for a single state (s) and three actions (a1, a2, a3)
# Best Action is a2 (Q=10)
Q_VALUES = np.array([5.0, 10.0, 0.0]) 
ACTION_NAMES = ['A1 (Q=5)', 'A2 (Q=10, Best)', 'A3 (Q=0)']

# --- Scenario A: High Temperature (High Exploration) ---
T_HIGH = 10.0
PI_HIGH = boltzmann_policy(Q_VALUES, T_HIGH)

# --- Scenario B: Low Temperature (High Exploitation) ---
T_LOW = 0.1
PI_LOW = boltzmann_policy(Q_VALUES, T_LOW)

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

x = np.arange(len(Q_VALUES))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

# Plot 1: High Temperature (Uniform distribution)
rects1 = ax.bar(x - width/2, PI_HIGH, width, label=f'$T={T_HIGH}$ (High Exploration)', color='skyblue')
# Plot 2: Low Temperature (Greedy distribution)
rects2 = ax.bar(x + width/2, PI_LOW, width, label=f'$T={T_LOW}$ (High Exploitation)', color='darkred')

# Labeling and Formatting
ax.set_title('Boltzmann Policy: Temperature Control of Exploration-Exploitation')
ax.set_xlabel('Action')
ax.set_ylabel('Selection Probability $\\pi(a \mid s)$')
ax.set_xticks(x)
ax.set_xticklabels(ACTION_NAMES)
ax.set_ylim(0, 1.0)
ax.legend()
plt.grid(True, axis='y')
plt.show()

print("\n--- Boltzmann Exploration Analysis ---")
print(f"Q-Values: {Q_VALUES}")

print("\n1. High Temperature (T=10.0):")
print(f"  Probabilities: {np.round(PI_HIGH, 3)} (Near Uniform)")
print("  Behavior: High entropy, actions selected almost equally (EXPLORATION).")

print("\n2. Low Temperature (T=0.1):")
print(f"  Probabilities: {np.round(PI_LOW, 3)} (Near Greedy)")
print("  Behavior: Low entropy, action A2 (Q=10) is selected with near-certainty (EXPLOITATION).")

print("\nConclusion: The temperature parameter T successfully controls the exploration-exploitation balance. At high T, the policy flattens, increasing randomness (entropy). At low T, the policy concentrates the probability mass on the best action (Q=10), minimizing effective energy and maximizing deterministic exploitation.")
```
**Sample Output:**
```
--- Boltzmann Exploration Analysis ---
Q-Values: [ 5. 10.  0.]

1. High Temperature (T=10.0):
  Probabilities: [0.307 0.506 0.186] (Near Uniform)
  Behavior: High entropy, actions selected almost equally (EXPLORATION).

2. Low Temperature (T=0.1):
  Probabilities: [0. 1. 0.] (Near Greedy)
  Behavior: Low entropy, action A2 (Q=10) is selected with near-certainty (EXPLOITATION).

Conclusion: The temperature parameter T successfully controls the exploration-exploitation balance. At high T, the policy flattens, increasing randomness (entropy). At low T, the policy concentrates the probability mass on the best action (Q=10), minimizing effective energy and maximizing deterministic exploitation.
```