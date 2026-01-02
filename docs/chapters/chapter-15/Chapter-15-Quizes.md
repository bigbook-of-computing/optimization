# **Chapter 15 : Quizes**

---

!!! note "Quiz"
    **1. What is the primary objective of a Reinforcement Learning agent?**

    - A. To learn the joint probability distribution of the states, $P(\mathbf{s})$.
    - B. To minimize the reconstruction error of the input data.
    - C. To maximize the expected cumulative discounted reward over a trajectory.
    - D. To classify states into discrete categories.

    ??? info "See Answer"
        **Correct: C**

        *(RL is fundamentally about learning a policy for sequential decision-making that maximizes a long-term reward signal, defined as $J(\pi) = \mathbb{E}_{\pi}[\sum_{t=0}^\infty \gamma^t r_t]$.)*

---

!!! note "Quiz"
    **2. A Markov Decision Process (MDP) is formally defined by a 5-tuple. Which of the following is NOT part of the standard MDP tuple?**

    - A. A set of states, $\mathcal{S}$.
    - B. A transition probability kernel, $P(s'|s,a)$.
    - C. A policy, $\pi(a|s)$.
    - D. A reward function, $R(s,a)$.

    ??? info "See Answer"
        **Correct: C**

        *(The policy $\pi$ is what the agent *learns*; it is the solution to the MDP, not part of its definition. The MDP is defined by $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$.)*

---

!!! note "Quiz"
    **3. The Bellman Optimality Equation provides a recursive definition for the optimal value function. What is the physical analogy for the process of solving these equations?**

    - A. A system undergoing nuclear fission.
    - B. A process of dynamic relaxation where the system settles into a steady-state equilibrium of expected return.
    - C. A particle accelerating in a uniform magnetic field.
    - D. The process of quantum tunneling through a potential barrier.

    ??? info "See Answer"
        **Correct: B**

        *(Iteratively applying the Bellman operator is like a physical system relaxing to its lowest energy state. The value function converges to a stable equilibrium, analogous to heat diffusion or potential field relaxation.)*

---

!!! note "Quiz"
    **4. In the context of RL, what is the difference between a state-value function ($V^\pi(s)$) and an action-value function ($Q^\pi(s,a)$)?**

    - A. $V^\pi(s)$ is for deterministic policies, while $Q^\pi(s,a)$ is for stochastic policies.
    - B. $V^\pi(s)$ measures the expected return from a state $s$, while $Q^\pi(s,a)$ measures the expected return after taking a specific action $a$ from state $s$ and following the policy thereafter.
    - C. $V^\pi(s)$ is used in model-based RL, while $Q^\pi(s,a)$ is used in model-free RL.
    - D. There is no difference; the terms are interchangeable.

    ??? info "See Answer"
        **Correct: B**

        *($Q^\pi(s,a)$ is more specific as it evaluates the quality of a particular action within a state, which is why model-free methods like Q-Learning focus on learning it directly.)*

---

!!! note "Quiz"
    **5. Q-Learning is a "model-free" algorithm. What does this mean?**

    - A. It does not use a neural network.
    - B. It does not require explicit knowledge of the environment's transition probabilities $P(s'|s,a)$ or reward function $R(s,a)$.
    - C. It can only solve problems with a finite number of states.
    - D. It does not have a policy.

    ??? info "See Answer"
        **Correct: B**

        *(Model-free algorithms learn directly from interaction and experience (samples of $\langle s, a, r, s' \rangle$) rather than from a pre-defined model of the world's dynamics.)*

---

!!! note "Quiz"
    **6. The Q-Learning update rule is based on minimizing the "Temporal-Difference (TD) Error." What is the TD Error?**

    - A. The difference between the predicted Q-value and the true, final return of the episode.
    - B. The difference between the current Q-value estimate and a "bootstrapped" target value calculated from the immediate reward and the next state's estimated value.
    - C. The error rate of the neural network used to approximate Q.
    - D. The time it takes for one episode to complete.

    ??? info "See Answer"
        **Correct: B**

        *(The TD Error is $[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$. It's the error in the current prediction, which the algorithm works to minimize.)*

---

!!! note "Quiz"
    **7. How do Policy Gradient (PG) methods differ from value-based methods like Q-Learning?**

    - A. PG methods can only be used in discrete action spaces.
    - B. PG methods directly parameterize and optimize the policy $\pi_{\mathbf{\theta}}(a|s)$, whereas Q-Learning learns a value function and derives the policy implicitly.
    - C. PG methods are always model-based.
    - D. PG methods do not use a discount factor.

    ??? info "See Answer"
        **Correct: B**

        *(Policy Gradient methods search in the space of policies, while Q-Learning searches in the space of value functions.)*

---

!!! note "Quiz"
    **8. What is the primary role of the "Critic" in an Actor-Critic architecture?**

    - A. To select the actions for the agent to take.
    - B. To store the experience replay buffer.
    - C. To evaluate the actions taken by the Actor by learning a value function ($V$ or $Q$) and providing a low-variance TD error signal.
    - D. To determine the exploration rate $\epsilon$.

    ??? info "See Answer"
        **Correct: C**

        *(The Critic "criticizes" the Actor's choices by calculating the advantage of the taken action, which the Actor then uses to update its policy. This reduces the high variance of simpler PG methods like REINFORCE.)*

---

!!! note "Quiz"
    **9. Deep Reinforcement Learning (Deep RL) uses "Experience Replay" to stabilize training. What is the purpose of this technique?**

    - A. To increase the learning rate.
    - B. To store past experiences and randomly sample from them, which breaks the strong temporal correlation between consecutive samples and stabilizes learning.
    - C. To replay successful episodes more frequently.
    - D. To visualize the agent's past behavior.

    ??? info "See Answer"
        **Correct: B**

        *(Training on i.i.d. samples is a core assumption of SGD. Experience replay makes the data more i.i.d.-like, preventing the unstable updates that can result from highly correlated sequential data.)*

---

!!! note "Quiz"
    **10. What is the "exploration vs. exploitation" dilemma in RL?**

    - A. The choice between using a deep or a shallow neural network.
    - B. The trade-off between exploring the environment to find potentially better strategies and exploiting known strategies to maximize immediate reward.
    - C. The decision to use a model-based or model-free algorithm.
    - D. The conflict between the Actor and the Critic networks.

    ??? info "See Answer"
        **Correct: B**

        *(This is the fundamental challenge in RL. Too much exploitation leads to suboptimal policies, while too much exploration leads to poor performance.)*

---

!!! note "Quiz"
    **11. In Boltzmann exploration, the policy is $\pi(a|s) \propto e^{Q(s,a)/T}$. What is the role of the temperature parameter $T$?**

    - A. It controls the learning rate of the Q-function.
    - B. It determines the size of the neural network.
    - C. It controls the balance between exploration and exploitation. High $T$ leads to more exploration (randomness), while low $T$ leads to more exploitation (greediness).
    - D. It is a measure of the actual temperature of the computer's CPU.

    ??? info "See Answer"
        **Correct: C**

        *($T$ acts as a "thermal energy" parameter. As $T \to \infty$, the policy becomes uniform. As $T \to 0$, the policy becomes deterministic and greedy. This is analogous to simulated annealing.)*

---

!!! note "Quiz"
    **12. Why can't standard Q-Learning be directly applied to continuous action spaces?**

    - A. Continuous actions require a model of the environment.
    - B. The Q-Learning update requires a $\max_{a'} Q(s',a')$ operation, which is an intractable optimization problem over a continuous space.
    - C. Neural networks cannot output continuous values.
    - D. The TD error is always zero in continuous spaces.

    ??? info "See Answer"
        **Correct: B**

        *(Finding the maximum value over an infinite set of actions at every step is computationally infeasible. This necessitates policy-gradient methods for continuous control.)*

---

!!! note "Quiz"
    **13. According to the framework of Entropy-Regularized RL, the optimal policy can be seen as a Boltzmann distribution over actions, $\pi^*(a|s) \propto e^{Q(s,a)/T}$. What does this imply?**

    - A. The agent's decisions are purely random.
    - B. The agent is minimizing a free-energy functional, balancing the maximization of reward (minimizing energy) with the maximization of policy diversity (entropy).
    - C. The agent can only take two actions at any given time.
    - D. The Q-function must be a linear function.

    ??? info "See Answer"
        **Correct: B**

        *(This formulation unifies RL with statistical mechanics, showing that the optimal policy is one that is as random as possible while still being consistent with maximizing reward.)*

---

!!! note "Quiz"
    **14. In the Gridworld example, the value landscape $V_{\max}(s)$ learned by Q-Learning forms a "potential well." What does the gradient of this landscape represent?**

    - A. The direction of maximum randomness.
    - B. The optimal policy, guiding the agent along the shortest path to the goal.
    - C. The amount of time spent in each state.
    - D. The probability of transitioning between states.

    ??? info "See Answer"
        **Correct: B**

        *(The optimal policy is to move "uphill" on the value landscape (or "downhill" on the negative energy landscape), which corresponds to the path of steepest ascent in value.)*

---

!!! note "Quiz"
    **15. The Bellman equation in discrete time is the analogue of which equation in continuous-time optimal control theory?**

    - A. The Schrödinger Equation.
    - B. The Navier-Stokes Equations.
    - C. The Hamilton-Jacobi-Bellman (HJB) Equation.
    - D. Maxwell's Equations.

    ??? info "See Answer"
        **Correct: C**

        *(The HJB equation is the continuous-time formulation of the principle of optimality, making it the direct counterpart to the discrete-time Bellman equation.)*

---

!!! note "Quiz"
    **16. What is the physical principle that is analogous to an RL agent finding an optimal policy to minimize cost over a trajectory?**

    - A. The Heisenberg Uncertainty Principle.
    - B. The Law of Universal Gravitation.
    - C. The Principle of Least Action.
    - D. The Zeroth Law of Thermodynamics.

    ??? info "See Answer"
        **Correct: C**

        *(Just as a physical system follows a path that minimizes the action integral, an RL agent learns a policy that minimizes the cumulative cost (or maximizes reward) over its trajectory.)*

---

!!! note "Quiz"
    **17. In multi-agent RL, complex behaviors like cooperation and competition can arise without being explicitly programmed. This phenomenon is known as:**

    - A. Overfitting.
    - B. Emergent behavior and self-organization.
    - C. Catastrophic forgetting.
    - D. The credit assignment problem.

    ??? info "See Answer"
        **Correct: B**

        *(This is analogous to spontaneous order in many-body physical systems, where complex global patterns arise from simple local interactions.)*

---

!!! note "Quiz"
    **18. What is the purpose of a "baseline" in the REINFORCE Policy Gradient algorithm?**

    - A. To increase the learning rate.
    - B. To reduce the variance of the gradient estimate without introducing bias, leading to more stable learning.
    - C. To ensure the rewards are always positive.
    - D. To define the initial values of the policy parameters.

    ??? info "See Answer"
        **Correct: B**

        *(By subtracting a baseline (like the state-value $V(s)$) from the sampled return $G_t$, the resulting advantage $A_t$ has lower variance, making the policy updates more reliable.)*

---

!!! note "Quiz"
    **19. What is the "bootstrapping" property of Temporal-Difference (TD) learning?**

    - A. The algorithm learns from complete episodes only.
    - B. The algorithm updates its value estimates based on other, future value estimates, rather than waiting for the final outcome.
    - C. The algorithm requires a pre-trained model to start.
    - D. The algorithm randomly resets its weights during training.

    ??? info "See Answer"
        **Correct: B**

        *(The TD target $r + \gamma \max_{a'} Q(s',a')$ uses the current estimate $Q(s',a')$ to update $Q(s,a)$. This "pulling oneself up by one's bootstraps" allows learning to occur at every step.)*

---

!!! note "Quiz"
    **20. In Deep Q-Networks (DQN), what is the role of the "target network"?**

    - A. To generate the input states for the main network.
    - B. To provide a stable, temporarily fixed target for the TD error calculation, mitigating the instability of a "moving target" problem.
    - C. It is a copy of the network used for exploration only.
    - D. It is the final, converged network that is saved after training.

    ??? info "See Answer"
        **Correct: B**

        *(By using a delayed copy of the Q-network to compute the TD target, the optimization landscape is temporarily stabilized, which was a crucial innovation for making Deep RL work.)*

---

!!! note "Quiz"
    **21. The policy gradient theorem, $\nabla J = \mathbb{E}[\nabla_{\mathbf{\theta}} \ln \pi(a\|s) Q(s,a)]$, tells us that the policy parameters should be updated to:**

    - A. Increase the probability of all actions equally.
    - B. Decrease the probability of actions that were taken.
    - C. Increase the probability of actions that led to a higher-than-average return (a positive Q-value or advantage).
    - D. Make the policy as deterministic as possible.

    ??? info "See Answer"
        **Correct: C**

        *(The term $\nabla_{\mathbf{\theta}} \ln \pi(a\|s)$ points in the direction that increases the probability of action $a$. This direction is weighted by $Q(s,a)$, so "good" actions are reinforced and "bad" actions are suppressed.)*

---

!!! note "Quiz"
    **22. What is the physical analogy for the exploration-exploitation trade-off in RL?**

    - A. The process of simulated annealing, where a system starts at a high temperature (exploration) and gradually cools to a low-temperature, crystalline state (exploitation).
    - B. The conservation of momentum in a closed system.
    - C. The photoelectric effect.
    - D. The expansion of the universe.

    ??? info "See Answer"
        **Correct: A**

        *(The temperature parameter in Boltzmann exploration directly mirrors the annealing process, controlling the system's transition from a random, high-entropy state to an ordered, low-energy state.)*

---

!!! note "Quiz"
    **23. In the context of RL as free-energy minimization, what does the policy's entropy, $\mathcal{H}[\pi]$, represent?**

    - A. The computational cost of running the policy.
    - B. The policy's diversity, randomness, or capacity for exploration.
    - C. The error rate of the policy.
    - D. The memory required to store the policy's parameters.

    ??? info "See Answer"
        **Correct: B**

        *(A high-entropy policy is more stochastic and explores more widely. Entropy-regularized RL explicitly encourages this diversity to find more robust solutions.)*

---

!!! note "Quiz"
    **24. Dynamic Programming methods like Value Iteration are guaranteed to converge to the optimal value function because the Bellman operator is a:**

    - A. Non-linear projection.
    - B. Contraction mapping.
    - C. Random permutation.
    - D. Lossy compression.

    ??? info "See Answer"
        **Correct: B**

        *(A contraction mapping is an operator that, when applied repeatedly, brings any two points in a metric space closer together. This property guarantees that the iterative application of the Bellman operator will converge to a unique fixed point, which is the optimal value function $V^*$.)*

---

!!! note "Quiz"
    **25. How does the RL framework bridge to the study of complex systems and physics?**

    - A. By proving that all physical systems can be controlled by a single RL agent.
    - B. By demonstrating that emergent behaviors, such as self-organization and cooperation, can arise from simple, local reward-maximization rules, similar to how order emerges in many-body physical systems.
    - C. By showing that RL algorithms can run faster on quantum computers.
    - D. By using the same programming languages as physics simulations.

    ??? info "See Answer"
        **Correct: B**

        *(RL provides a computational platform for studying how complex, adaptive, and intelligent behavior can emerge from the fundamental principle of utility maximization, linking it to the study of complexity in physics and biology.)*
