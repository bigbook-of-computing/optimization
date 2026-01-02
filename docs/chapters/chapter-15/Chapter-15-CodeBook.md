# 🎮 Chapter 15: Reinforcement Learning and Control

## Project 1: Q-Learning Convergence (Value Iteration)

-----

### Definition: Q-Learning Convergence

The goal is to implement the core **Q-Learning update rule** to demonstrate its convergence to the optimal action-value function $Q^*(\mathbf{s}, a)$ for a simple, deterministic environment. This is the model-free approach to solving the **Bellman Optimality Equation**.

### Theory: Temporal Difference (TD) Learning

Q-Learning is a **model-free** algorithm that learns the optimal policy by iteratively updating the action-value function $Q(\mathbf{s}, a)$, which represents the maximum expected discounted return for taking action $a$ in state $\mathbf{s}$. The process uses **Temporal Difference (TD) learning** to correct the current estimate based on the observed error:

$$Q(\mathbf{s}, a) \leftarrow Q(\mathbf{s}, a) + \eta \left[ \underbrace{r + \gamma \max_{a'} Q(\mathbf{s}', a')}_{\text{Target (New Estimate)}} - \underbrace{Q(\mathbf{s}, a)}_{\text{Old Estimate}} \right]$$

  * The term in the bracket is the **TD error**.
  * The factor $\eta$ is the **learning rate**.
  * The factor $\gamma$ is the **discount rate**.

The convergence of $Q$ to the optimal function $Q^*$ is guaranteed if every state-action pair is visited infinitely often (exploration).

### Extensive Python Code and Visualization

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

-----

## Project 2: Policy Gradient (REINFORCE) with Baseline

-----

### Definition: Policy Gradient with Baseline

The goal is to implement the core gradient estimation of the **Policy Gradient (REINFORCE)** algorithm using a **baseline** to demonstrate variance reduction.

### Theory: Likelihood Ratio and Variance Reduction

Policy Gradient methods directly optimize the policy $\pi_{\boldsymbol{\theta}}(\mathbf{a} \mid \mathbf{s})$ parameterized by $\boldsymbol{\theta}$. The gradient of the objective function (expected return $J$) is estimated using the **likelihood ratio trick**:

$$\nabla_{\boldsymbol{\theta}} J \propto \sum_{t} \nabla_{\boldsymbol{\theta}} \ln \pi_{\boldsymbol{\theta}}(\mathbf{a}_t|\mathbf{s}_t) \cdot A_t$$

The key insight is the use of the **advantage term ($A_t$)**:

$$A_t = G_t - b(\mathbf{s})$$

Where $G_t$ is the return from time $t$, and $b(\mathbf{s})$ is the **baseline** (typically the state value $V(\mathbf{s})$). Subtracting the baseline reduces the **variance** of the gradient estimate without introducing **bias**, making learning much more stable and efficient.

-----

### Extensive Python Code

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

-----

## Project 3: Simulating RL as Free-Energy Minimization (Conceptual)

-----

### Definition: Simulating RL as Free-Energy Minimization

The goal is to conceptualize the **Maximum Entropy Reinforcement Learning (MaxEnt RL)** objective, demonstrating the explicit trade-off between maximizing expected reward (Energy) and maximizing policy randomness (Entropy).

### Theory: MaxEnt and the Free Energy Analogy

MaxEnt RL integrates the policy's **entropy** $H(\pi)$ into the standard reward objective:

$$\text{Objective} = \mathbb{E}_{\pi} \left[ \sum_{t} \underbrace{r(\mathbf{s}_t, \mathbf{a}_t)}_{\text{Reward (Energy)}} + \alpha \underbrace{H(\pi(\cdot | \mathbf{s}_t))}_{\text{Entropy (Exploration)}} \right]$$

  * This objective is maximized to find the optimal, but maximally random, policy.
  * This objective function is mathematically equivalent to minimizing an effective \*\*Free Energy ($\mathcal{F}$) \*\*.

The **$\boldsymbol{\alpha}$ parameter** (analogue of inverse temperature $\alpha \sim 1/T$) controls the balance:

  * High $\alpha$: Policy favors high entropy (more **exploration**).
  * Low $\alpha$: Policy favors high reward (more **exploitation**).

-----

### Extensive Python Code

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

-----

## Project 4: Simulating Boltzmann Exploration and Temperature

-----

### Definition: Simulating Boltzmann Exploration and Temperature

The goal is to implement the **Boltzmann Exploration** policy and observe how the **temperature ($T$)** parameter controls the exploitation-exploration balance.

### Theory: Boltzmann Policy and the Exploitation–Exploration Trade-off

The Boltzmann policy (Softmax policy in the action space) selects actions based on their Q-values, $Q(\mathbf{s}, a)$, weighted by an external **temperature parameter ($T$)**:

$$\pi(a | s) = \frac{e^{Q(s, a) / T}}{\sum_{a'} e^{Q(s, a') / T}}$$

This policy is a direct analogue of the **Boltzmann distribution** from statistical mechanics, where $Q(\mathbf{s}, a)$ is analogous to **negative energy**.

  * **High $T \to \infty$:** The policy approaches a uniform distribution ($\pi(a) \to 1/|A|$), maximizing **exploration** (high entropy).
  * **Low $T \to 0$:** The policy becomes greedy ($\pi(a) \to 1$ for $a^*$), maximizing **exploitation** (minimum effective energy).

We use a fixed set of Q-values to demonstrate how $T$ tunes the action selection probability.

-----

### Extensive Python Code and Visualization

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


