## 🎮 Chapter 15: Reinforcement Learning and Control (Workbook)

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

**1. In the RL framework, the mathematical object that represents the agent's strategy or a mapping from states to actions is called the:**

* **A.** Value function ($V$).
* **B.** **Policy ($\pi$)**. (**Correct**)
* **C.** Discount factor ($\gamma$).
* **D.** Transition kernel ($P$).

**2. The single component that distinguishes the RL objective from the static optimization objective (Part II) is the explicit inclusion of the concept of a:**

* **A.** Single loss function $L$.
* **B.** **Temporal trajectory or sequence of states**. (**Correct**)
* **C.** Fixed set of parameters $\boldsymbol{\theta}$.
* **D.** Non-convex landscape.

---

#### Interview-Style Question

**Question:** The RL objective function $J(\pi)$ maximizes the expected *discounted* reward, using a discount factor $\gamma \in [0, 1]$. What is the dual purpose of including this discount factor in the calculation?

**Answer Strategy:** The discount factor $\gamma$ has two primary purposes:
1.  **Mathematical Stability:** It prevents the infinite sum of future rewards from **diverging**.
2.  **Uncertainty/Time Preference:** It weights immediate rewards more heavily than distant future rewards. This models the real-world preference for immediate utility and the **uncertainty** of future environmental states.

---
***

### 15.2 Markov Decision Processes (MDPs)

> **Summary:** RL problems are formally modeled as **Markov Decision Processes (MDPs)**, defined by the five-tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$. The core constraint is the **Markov Property**, which simplifies the problem by assuming the next state $s'$ depends **only on the current state $s$ and action $a$**. The optimal solution is found by solving the recursive **Bellman Optimality Equations**.

#### Quiz Questions

**1. The constraint that simplifies the environment structure by stating the next state is independent of the entire history of past states is known as the:**

* **A.** Principle of Least Action.
* **B.** **Markov Property**. (**Correct**)
* **C.** Contraction Mapping.
* **D.** Policy Gradient Theorem.

**2. The **Bellman Optimality Equation** recursively defines the optimal value of a state $V^*(s)$ by relating it to the maximum expected value achievable from:**

* **A.** The initial starting state.
* **B.** The total lifetime reward.
* **C.** **The immediate reward plus the discounted maximum future value from the succeeding states**. (**Correct**)
* **D.** The average value of all surrounding states.

---

#### Interview-Style Question

**Question:** The Value Function $V(s)$ is analogous to the negative potential energy of a state. In the Gridworld worked example (Section 15.12), the reward function is $r=-1$ per step, and $r=0$ at the goal. Describe the shape of the learned $V(s)$ potential landscape.

**Answer Strategy:** Since the agent is minimizing cost (negative reward), the optimal value function $V^*(s)$ forms a **potential well**. The shape will be a **monotonically increasing surface** (approaching zero from the negative side) centered at the goal state. The goal state itself will have the highest value ($V^*=0$), and the value will decrease (become more negative, e.g., $V^*=-1, -2, -3, \dots$) as the state moves farther away from the goal, creating a clear **potential gradient** that attracts the agent.

---
***

### 15.3 Policies and Value Functions

> **Summary:** The **policy ($\pi$)** is the agent's strategy (the control field). **Value functions** ($V$ and $Q$) quantify the expected cumulative reward, acting as the potential landscape. The **optimal policy ($\pi^*$)** is found by selecting the action that maximizes the optimal **Action-Value Function ($Q^*$)**.

### 15.4 Dynamic Programming and Value Iteration

> **Summary:** **Dynamic Programming (DP)** solves MDPs by breaking them into iterative, recursive subproblems. **Value Iteration** repeatedly applies the Bellman Optimality Equation until the value function converges to $V^*$. DP methods achieve their solution via **successive approximation** and **contraction mapping**, which is mathematically guaranteed to converge. The convergence process is analogous to **heat diffusion** propagating reward information backward through time.

### 15.5 Q-Learning — Learning Without a Model

> **Summary:** **Q-Learning** is a fundamental **model-free** algorithm that learns the optimal $Q^*(s,a)$ function directly from experience. It uses **Temporal-Difference (TD) learning** and **bootstrapping**. The update rule is based on minimizing the **TD Error**. Q-Learning is equivalent to **stochastic gradient descent** on the Bellman error and functions as a **local energy correction** mechanism.

### 15.6 Policy Gradient Methods

> **Summary:** **Policy Gradient (PG)** methods **parameterize the policy ($\pi_{\boldsymbol{\theta}}$)** explicitly (e.g., as a neural network) and optimize it directly to maximize expected reward. The **Policy Gradient Theorem** provides the analytic form of the gradient, relating policy performance to the value of observed actions. PG methods often include **entropy regularization** to encourage policy diversity (exploration).

### 15.7 Actor–Critic Architecture

> **Summary:** **Actor–Critic** is a hybrid method using two coupled networks. The **Actor** ($\pi_{\boldsymbol{\theta}}$) selects the action, and the **Critic** ($V_{\boldsymbol{\phi}}$ or $Q_{\boldsymbol{\phi}}$) evaluates the action by estimating the value function. The Critic computes the **TD Error ($\delta_t$)**, which serves as the **advantage estimate** to update the Actor's policy gradient. This reduces gradient variance and stabilizes the learning process.

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

### Project 2: Simulating Heat Diffusion of Reward (Value Iteration)

* **Goal:** Implement the **Value Iteration** algorithm (Dynamic Programming) to show how value information propagates backward through the state space.
* **Setup:** Use the same Gridworld environment but assume the **transition kernel $P$ is known** (deterministic for simple moves).
* **Steps:**
    1.  Initialize $V(s)$ to zero everywhere.
    2.  Implement the Value Iteration update rule (using the $\max_a$ and $\sum_{s'} P(s'|s,a) V_k(s')$ terms).
    3.  Run the algorithm for only a few iterations (e.g., $k=1, 2, 3$).
* ***Goal***: Show, by observing the $V_k(s)$ map, that the reward information (high value) starts only at the goal state and **diffuses backward** one layer per iteration, confirming the heat diffusion analogy.

### Project 3: Policy Gradient (REINFORCE) with Baseline (Conceptual)

* **Goal:** Implement the Policy Gradient (REINFORCE) algorithm, demonstrating direct policy optimization and the necessity of using a baseline for variance reduction.
* **Setup:** Define a small, multi-step environment where actions are probabilistic. Parameterize the policy $\pi_{\boldsymbol{\theta}}$ as a simple neural network.
* **Steps:**
    1.  Implement the core REINFORCE gradient: $\nabla_{\boldsymbol{\theta}} J \propto \nabla_{\boldsymbol{\theta}} \ln \pi_{\boldsymbol{\theta}}(a|s) \cdot G_t$ ($G_t$ is the sampled return).
    2.  Implement a more stable update using a **baseline** ($b_s$) to calculate the **advantage** ($A_t = G_t - b_s$).
* ***Goal***: Demonstrate that the policy is updated directly based on which actions ($a|s$) lead to higher-than-average rewards (advantage), illustrating the principle of direct policy adaptation.

### Project 4: Simulating Boltzmann Exploration and Temperature

* **Goal:** Implement the **Boltzmann Exploration** policy and observe how the **temperature ($T$)** parameter controls the exploitation-exploration balance.
* **Setup:** Use a fixed Q-table where the values for state $s$ are known (e.g., $Q(s, a_1)=5, Q(s, a_2)=10, Q(s, a_3)=0$).
* **Steps:**
    1.  Implement the Boltzmann policy: $\pi(a|s) \propto e^{Q(s,a)/T}$.
    2.  Calculate the resulting probability distribution $\pi(a|s)$ for two cases: **High Temperature** ($T=10$) and **Low Temperature** ($T=0.1$).
* ***Goal***: Show that at $T=10$, the distribution is nearly uniform (high exploration), while at $T=0.1$, the probability is highly concentrated on the optimal action $a_2$ (high exploitation), confirming the **annealing analogy**.
