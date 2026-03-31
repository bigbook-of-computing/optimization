# **Chapter 6: 6. Advanced Gradient Dynamics**

# **Introduction**

Chapter 5 established gradient descent as the fundamental dynamical law of optimization: a deterministic, **overdamped relaxation** process where the parameter velocity is directly proportional to the negative gradient force ($d\mathbf{\theta}/dt = -\gamma \nabla L$), analogous to a particle moving through thick viscous fluid. While this framework is theoretically sound—the loss function acts as a Lyapunov function, guaranteeing monotonic energy dissipation and convergence to critical points—it suffers catastrophic failures on the high-dimensional, **anisotropic ravine landscapes** characteristic of real machine learning problems. The root cause is the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ of the Hessian: when curvature varies by orders of magnitude across different parameter directions, a single global learning rate $\eta$ must be conservatively set to prevent divergence in the stiffest direction, inevitably causing agonizingly slow convergence along the flat "sloppy" valley floors. This manifests as the infamous **zigzagging** behavior—the gradient points perpendicular to the valley floor (across the steep walls), causing the optimizer to bounce back and forth while making negligible forward progress toward the minimum.

This chapter transcends the overdamped approximation by introducing two transformative concepts from classical mechanics and adaptive control theory: **momentum** (inertia) and **per-parameter adaptive scaling** (dynamic friction). We begin by upgrading the equation of motion from zero-mass overdamped dynamics to the full **damped harmonic oscillator** equation $m\ddot{\mathbf{\theta}} = -\gamma m\dot{\mathbf{\theta}} - \nabla L$, incorporating inertia as a velocity accumulation mechanism that enables the optimizer to **coast across plateaus** (maintaining speed when gradients are small) and **smooth out zigzags** (averaging oscillatory forces). We develop **Momentum** and its mathematically refined variant **Nesterov Accelerated Gradient (NAG)**, which employs anticipatory gradient evaluation at predicted future positions to enable faster convergence. Moving beyond global inertia, we introduce **adaptive methods**—**RMSProp** and **Adam (Adaptive Moment Estimation)**—which maintain running estimates of per-parameter gradient statistics (first and second moments $m_t, v_t$) to automatically compute coordinate-specific effective learning rates $\eta_i \propto 1/\sqrt{v_i}$. We reveal that adaptive methods fundamentally **redefine the geometry** of parameter space, acting as online learned diagonal preconditioners that transform anisotropic elliptical ravines into isotropic spherical bowls, approximating the ideal (but intractable) Newton's method scaling by $H^{-1}$.

By the end of this chapter, you will understand that modern optimizers are sophisticated **damped dynamical systems** whose total energy (Hamiltonian $\mathcal{H} = L + \frac{1}{2m}|\mathbf{p}|^2$) is continuously dissipated via friction until reaching equilibrium, that momentum enables barrier crossing and prevents stalling via accumulated kinetic energy, and that adaptive scaling is a form of **learned geometry** that makes the optimization isotropic in a transformed metric space—connecting to Natural Gradient Descent and the Fisher Information Matrix. You will recognize the empirical finding that stochastic noise (from mini-batch sampling) combined with adaptive dynamics preferentially drives the system toward **flat, wide minima** (thermodynamically stable states) that generalize better than sharp minima. These foundations prepare you for **Chapter 7**, where we confront landscapes so rugged (spin glasses) or discrete (combinatorial optimization) that gradient information becomes useless, requiring a shift to **heuristic methods** like **Simulated Annealing** that replace deterministic gradient forces with temperature-controlled stochastic jumps to tunnel through energy barriers.

---

# **Chapter 6: Outline**

| **Sec.** | **Title**                                                 | **Core Ideas & Examples**                                                                                                                                                                                      |
| -------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **6.1**  | **Beyond Simple Descent**                                 | Failure modes of gradient descent: zigzagging in anisotropic ravines (high condition number $\kappa$), stalling on plateaus (small $\|\nabla L\|$); need for memory (inertia) and per-parameter adaptivity; upgrade from overdamped $\mathbf{v} \propto -\nabla L$ to damped oscillator $m\ddot{\mathbf{\theta}} = -\gamma m\dot{\mathbf{\theta}} - \nabla L$; physical foundation for momentum and adaptive methods. |
| **6.2**  | **Momentum: Learning with Inertia**                       | Velocity accumulation $\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta \nabla L$, update $\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \mathbf{v}_{t+1}$; momentum coefficient $\beta \approx 0.9$ controls memory/damping; physical analogy—particle with mass coasting across plateaus, averaging out zigzags; enables barrier crossing via kinetic energy; trade-offs: overshooting with high $\beta$, still uses global $\eta$. |
| **6.3**  | **Nesterov Accelerated Gradient (NAG)**                   | Anticipatory dynamics: compute gradient at predicted future position $\nabla L(\mathbf{\theta}_t + \beta \mathbf{v}_t)$ instead of current $\nabla L(\mathbf{\theta}_t)$; "look-ahead" correction prevents overshooting; provably accelerated convergence on convex functions; analogy—inertial navigation (skier anticipating turn, car steering before corner); smoother geodesic path on curved loss manifold. |
| **6.4**  | **RMSProp: Adaptive Step Sizes**                          | Per-parameter learning rates via running average of squared gradients $s_t = \rho s_{t-1} + (1-\rho)(\nabla L_t)^2$; update $\mathbf{\theta}_{t+1} = \mathbf{\theta}_t - \frac{\eta}{\sqrt{s_t+\epsilon}}\nabla L_t$; inverse curvature scaling: large $s_i$ (stiff) → small effective $\eta_i$ (strong brake), small $s_j$ (sloppy) → large effective $\eta_j$ (light brake); transforms elliptical contours to spherical, lowers condition number. |
| **6.5**  | **Adam: Adaptive Moment Estimation**                      | Combines momentum ($m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L$) and adaptive scaling ($v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$); bias correction $m_{\text{hat}} = m_t/(1-\beta_1^t)$, $v_{\text{hat}} = v_t/(1-\beta_2^t)$; update $\mathbf{\theta}_{t+1} = \mathbf{\theta}_t - \eta \frac{m_{\text{hat}}}{\sqrt{v_{\text{hat}}}+\epsilon}$; defaults $\beta_1=0.9$, $\beta_2=0.999$; "smart particle" navigator and damper; robust, widely used, limitation—may prefer sharp minima. |
| **6.6**  | **Geometry of Adaptive Dynamics**                         | Adaptive methods redefine metric: $\Delta \theta_i \propto \frac{1}{\sqrt{v_i}} m_i$ transforms distance $d\mathbf{\theta}^2_{\text{adaptive}} \sim v^{-1} d\mathbf{\theta} \cdot d\mathbf{\theta}$; diagonal approximation of inverse Hessian $H^{-1}$ (Newton's method $\mathbf{\theta} - H^{-1}\nabla L$ intractable); isotropic motion in transformed data-dependent space; connection to Natural Gradient Descent via Fisher Information Matrix $I(\mathbf{\theta})^{-1}\nabla L$. |
| **6.7**  | **Stability, Convergence, and Energy View**               | Hamiltonian (total energy) $\mathcal{H} = L(\mathbf{\theta}) + \frac{1}{2m}\|\mathbf{p}\|^2$ (potential + kinetic); dissipation via damping $\frac{d\mathcal{H}}{dt} = -\gamma \|\mathbf{v}\|^2 \le 0$ ensures stability and convergence; damped oscillator picture—critically damped vs. underdamped regimes; fixed points at minima; phase space $(\mathbf{\theta}, \mathbf{v})$ dynamics; attractors as basins. |
| **6.8**  | **Comparing Methods on a Ravine Function**                | Test function $L(\theta_1, \theta_2) = 0.5\theta_1^2 + 5\theta_2^2$ (anisotropic, $\kappa=10$); GD zigzags (oscillates on steep $\theta_2$, slow on flat $\theta_1$); Momentum smooths oscillations via averaging, accelerates along valley; Adam most direct—adaptive scaling transforms ravine to isotropic bowl; trajectory comparison shows geometry vs. dynamics trade-offs. |
| **6.9**  | **Code Demo: Comparing GD, Momentum, and Adam**           | Python implementation on ravine $L=0.5\theta_1^2 + 5\theta_2^2$; GD update $\mathbf{\theta} - \eta\nabla L$, Momentum with $\beta=0.9$, Adam with $\beta_1=0.9, \beta_2=0.999$; visualize trajectories on contour plot from start $(3,3)$ to minimum $(0,0)$; GD blue zigzag, Momentum orange smooth, Adam green direct; confirms inertia + adaptivity overcome anisotropy. |
| **6.10** | **Practical Notes and Heuristics**                        | Adam defaults: $\eta \approx 10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$; learning rate annealing (cooling) to dampen stochastic oscillations for precise convergence; cosine annealing with warm restarts (SGDR) for cyclic exploration; flat minima vs. sharp minima—high noise/momentum find flat basins (thermodynamically stable), correlate with better generalization; overfitting at sharp minima. |
| **6.11** | **Takeaways & Bridge to Chapter 7**                       | Recap: momentum adds inertia (coasting, oscillation dampening), adaptive methods (RMSProp, Adam) learn geometry via per-parameter scaling (diagonal $H^{-1}$ approximation), damped dynamical system with Hamiltonian dissipation, flat minima from stochastic+adaptive dynamics; limitations—gradient-based methods fail on discontinuous/combinatorial landscapes, deep barriers in spin glasses; bridge to Chapter 7: heuristic methods (Simulated Annealing) use temperature-controlled stochastic jumps to tunnel barriers when gradients useless. |

---

In Chapter 5, we established **Gradient Descent (GD)** as a **deterministic, overdamped relaxation** process, where the velocity of the model parameters is directly proportional to the force ($\mathbf{v} \propto -\nabla L$). While robust in theory, this simple dynamic fails spectacularly in the high-dimensional, anisotropic ravines of real loss landscapes. It forces the optimizer to use an extremely small step size ($\eta$) constrained by the stiffest direction, leading to painfully slow convergence along the flat valley floor.

This chapter moves beyond the **overdamped** approximation, introducing concepts from classical mechanics to create richer, more efficient optimization dynamics that form the foundation of modern machine learning.

---

## **6.1 Beyond Simple Descent**

---

### **Motivation: The Failure in Ravines**

The primary weakness of standard Gradient Descent (GD) and even Stochastic Gradient Descent (SGD) is their inability to navigate ill-conditioned, **anisotropic** loss landscapes—a geometry characterized by a large **condition number** of the Hessian.

$$
\kappa = \lambda_{\max} / \lambda_{\min}
$$

* **Zigzagging:** The gradient vector, pointing in the direction of steepest descent, is perpendicular to the elliptical contour lines. In a narrow ravine, the gradient points *across* the steep walls, leading to oscillation and a "zigzag" path that makes almost no progress toward the minimum along the flat valley floor.
* **Stalling:** In vast, flat regions like plateaus or the bottoms of "sloppy" valleys, the gradient magnitude $\Vert \nabla L \Vert$ approaches zero. The update step $\mathbf{\theta}_{t+1} = \mathbf{\theta}_t - \eta \nabla L$ becomes vanishingly small, causing the optimization to stall prematurely.

To address these failures, we must introduce **memory** and **intelligence** into the dynamics.

---

### **Need for Richer Dynamics: Inertia and Adaptivity**

The solution is to upgrade the optimizer's dynamical system by breaking the simple, friction-dominated $\mathbf{v} \propto \mathbf{F}$ relationship. We require two new properties to traverse the landscape efficiently:

* **Inertia (Momentum):** Introducing a "mass" to the system allows the velocity to accumulate contributions from past gradients. This memory helps the optimizer **smooth out zigzags** (momentum cancels out oscillations across the ravine) and **coast across plateaus** (maintaining speed even when the current gradient is small).
* **Per-Parameter Adaptivity:** Recognizing that the landscape's stiffness varies by parameter ($\theta_i$ versus $\theta_j$), we need a mechanism to set an effective, coordinate-specific learning rate $\eta_i$. This allows the optimizer to take large steps in flat (sloppy) directions and conservative steps in steep (stiff) directions, effectively **sphericizing the loss landscape on the fly**.

---

### **Physical Analogy: The Damped Harmonic Oscillator**

The new, richer dynamics that incorporate inertia can be directly modeled using a generalized physical equation of motion. While Chapter 5 used the simplified zero-mass (overdamped) limit, modern optimizers are better described by the full equation for a particle with mass $m$ moving in a viscous medium under a force derived from a potential $L(\mathbf{\theta})$:

$$
\frac{d^2\mathbf{\theta}}{dt^2} + \gamma \frac{d\mathbf{\theta}}{dt} + \frac{1}{m}\nabla L(\mathbf{\theta}) = 0
$$

We can multiply by $m$ and define $\mathbf{F} = -\nabla L(\mathbf{\theta})$ to obtain a familiar form:

$$
m\frac{d^2\mathbf{\theta}}{dt^2} = -\gamma m \frac{d\mathbf{\theta}}{dt} - \nabla L(\mathbf{\theta})
$$

This equation provides the foundation for our next concepts:
* The term $m\frac{d^2\mathbf{\theta}}{dt^2}$ is **Inertia** (or mass $\times$ acceleration).
* The term $-\gamma m \frac{d\mathbf{\theta}}{dt}$ is the **Damping Force** (or friction, $\propto$ velocity).
* The term $-\nabla L(\mathbf{\theta})$ is the **Potential Force**.

By numerically integrating this equation (rather than the zero-mass version in Chapter 5), we introduce **Momentum** (Chapter 6.2) and transition from static relaxation to a sophisticated, **damped dynamical system** where the parameters evolve with memory and inertia.

---

## **6.2 Momentum — Learning with Inertia**

The **Momentum** method, often simply called **Momentum**, is the first and most foundational technique to improve upon basic gradient descent by introducing the concept of **inertia** or "memory" into the optimization dynamics. It is a direct response to the "zigzagging" and "stalling" problems inherent in the overdamped model of Chapter 5.

---

### **The Algorithm: Introducing Velocity**

Momentum accelerates learning by accumulating a velocity vector, $\mathbf{v}_t$, which incorporates both the current gradient and the history of previous gradients. The parameters are no longer updated solely by the instantaneous gradient; they are updated by this accumulated velocity.

The update rule introduces a new hyperparameter, $\beta$, which is the **momentum coefficient** (or dampening factor).

$$
\begin{aligned}
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t - \eta \nabla L(\mathbf{\theta}_t) \\
\mathbf{\theta}_{t+1} &= \mathbf{\theta}_t + \mathbf{v}_{t+1}
\end{aligned}
$$

* $\mathbf{v}_t$ is the **velocity** vector at time $t$.
* $\beta$ (a value typically set around 0.9) determines how much of the previous velocity is retained. It controls the amount of memory or persistence.
* $-\eta \nabla L(\mathbf{\theta}_t)$ is the current, instantaneous "force" pulling the parameters downhill.

This process transforms the optimization from a first-order map into a second-order recurrence relation.

---

### **Physical View: Inertia and Damping**

The Momentum update is the numerical approximation of the continuous equation of motion for a particle with mass $m$ undergoing **underdamped motion** in a viscous medium.

* **Inertia:** The term $\beta \mathbf{v}_t$ allows the optimizer to build up speed (momentum) and coast across flat regions where $\nabla L \approx 0$. This kinetic energy prevents the system from stalling.
* **Damping:** The coefficient $\beta$ acts as a friction or **damping factor**. If $\beta=0.9$, the motion is lightly damped, meaning $90\%$ of the current velocity persists, allowing for efficient movement. If $\beta=0$, we recover standard Gradient Descent (no memory).
* **Oscillation Cancellation:** In narrow ravines (Section 5.3), the current gradient zigzags, but the accumulated velocity $\mathbf{v}_t$ averages these oscillations out, smoothing the trajectory and allowing the optimizer to travel quickly along the flat valley floor toward the minimum.

This is analogous to a **ball rolling in a bumpy valley**. The kinetic energy gained allows the ball to roll straight over small bumps (local noise or shallow minima) instead of getting stuck in them.

---

### **Trade-offs and Limitations**

Momentum significantly accelerates convergence but introduces a critical new trade-off.

| Aspect | Description | Consequence |
| :--- | :--- | :--- |
| **Too High $\beta$** | Overly persistent velocity (low damping). | The particle may **overshoot** the minimum entirely and climb far up the other side, wasting steps. |
| **Too Low $\beta$** | Little memory (high damping). | The motion devolves back toward slow, zigzagging GD. |
| **Single $\eta$** | Still uses one global learning rate, $\eta$. | Still sensitive to the *anisotropy* of the landscape. Must use a small $\eta$ to prevent divergence in stiff directions. |

Momentum solves the inertia problem but does not address the fundamental issue of anisotropy and the compromised global learning rate. This requires the **adaptive methods** explored in the following sections.

---

## **6.3 Nesterov Accelerated Gradient (NAG)**

The **Nesterov Accelerated Gradient (NAG)**, originally developed by Yurii Nesterov for convex optimization, is a mathematically refined variation of the standard Momentum method. While standard Momentum (Section 6.2) is based on the idea of a particle that **reacts** to the gradient at its current position, NAG employs a form of **anticipatory dynamics** to correct its trajectory before overshooting the target.

---

### **The Anticipatory Idea: Looking Ahead**

The core weakness of standard Momentum is that it calculates the gradient $\nabla L(\mathbf{\theta}_t)$ at the current position $\mathbf{\theta}_t$ and then applies the accumulated velocity $\mathbf{v}_t$. By the time the next step is calculated, the system has effectively "overshot" the minimum due to its inertia, causing wasted motion.

NAG addresses this by recognizing that the velocity $\mathbf{v}_t$ already points in the direction of the *next* location. It exploits this predictive power by calculating the gradient *not* at the current location $\mathbf{\theta}_t$, but at the potential future location, $\mathbf{\theta}_t + \beta \mathbf{v}_t$.

The new step is therefore calculated based on an **"anticipatory" gradient**.

---

### **The Algorithm: Predictive Motion**

The NAG update rule modifies the velocity calculation to use the lookahead position:

$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta \nabla L(\mathbf{\theta}_t + \beta \mathbf{v}_t)
$$

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \mathbf{v}_{t+1}
$$

* $\beta \mathbf{v}_t$ represents the **predicted step** based on the existing velocity (momentum).
* $\nabla L(\mathbf{\theta}_t + \beta \mathbf{v}_t)$ is the gradient calculated at that predicted future position.

This predictive gradient helps the momentum method "turn the corner" earlier in curved valleys. By correcting the direction *before* the overshoot occurs, NAG achieves mathematically provable **accelerated convergence** rates compared to classic Gradient Descent or standard Momentum, especially in smooth, convex settings.

---

### **Physical Analogy: Inertial Navigation**

The NAG update is analogous to a vehicle using **inertial navigation** or a skier correcting their line.

* **Standard Momentum (Car Steering):** Imagine driving a car toward a turn. Standard Momentum calculates the necessary turn when the car *reaches* the turn. Due to the car's inertia, it drifts wide before correcting.
* **NAG (Predictive Steering):** NAG, by contrast, calculates the gradient at the predicted future point. It senses the curve *ahead* and starts turning the wheel *before* reaching the corner. This allows the system to follow a more geodesic (minimum-distance) path on the curved loss manifold.

The introduction of this **"anticipatory" force** makes the optimizer more efficient and stable, making NAG a widely used technique in high-performance deep learning.

!!! tip "Momentum as Oscillation Damping"
```
Momentum's power comes from averaging out oscillatory forces. In a narrow ravine, the gradient alternates direction (pointing left, then right across the steep walls), but the accumulated velocity $\beta \mathbf{v}_t$ acts as a low-pass filter, canceling these perpendicular oscillations while reinforcing the consistent downward component along the valley floor. This transforms erratic zigzags into smooth, accelerated descent.

```
---

## **6.4 RMSProp — Adaptive Step Sizes**

RMSProp, or **Root Mean Square Propagation**, is a crucial adaptive learning rate algorithm that addresses the long-standing problem of **anisotropy** (the ravine effect) that plagues both standard Gradient Descent and Momentum. By dynamically scaling the learning rate for *each parameter* based on the history of its past gradients, RMSProp allows the optimizer to move quickly along shallow (sloppy) directions and carefully along steep (stiff) ones.

---

### **The Problem of Anisotropy**

In real loss landscapes, different parameters $\theta_i$ and $\theta_j$ can correspond to vastly different curvatures.

* If parameter $\theta_i$ lies in a steep, stiff direction ($\lambda_{\max}$ is large), its gradients will be large.
* If parameter $\theta_j$ lies along a flat, sloppy direction ($\lambda_{\min}$ is small), its gradients will be small.

A single, global learning rate $\eta$ must be set small enough to prevent the large gradients of $\theta_i$ from causing divergence, which inevitably means $\eta$ is far too small for $\theta_j$, leading to slow convergence.

---

### **The Solution: Running Average of Squared Gradients**

RMSProp introduces a **per-parameter scaling factor** that normalizes the gradient by a running, exponentially decaying average of its squared magnitude.

The algorithm maintains a vector of accumulated squared gradients, $s_t$:

$$
\begin{aligned}
s_t &= \rho s_{t-1} + (1-\rho) (\nabla L_t)^2 \\
\mathbf{\theta}_{t+1} &= \mathbf{\theta}_t - \frac{\eta}{\sqrt{s_t+\epsilon}} \nabla L_t
\end{aligned}
$$

* $\nabla L_t$ is the stochastic gradient vector at time $t$.
* The division $(\nabla L_t)^2$ is calculated element-wise.
* $\rho$ (the decay rate, typically around 0.9) determines the length of the window over which squared gradients are averaged.
* $\epsilon$ (a small constant, e.g., $10^{-8}$) is added for numerical stability to prevent division by zero.

The scaling term $\frac{1}{\sqrt{s_t}}$ effectively acts as an **inverse curvature metric**.

---

### **Physical Analogy: Adaptive Friction**

RMSProp can be viewed as an optimization with an **adaptive, coordinate-dependent friction coefficient**.

* **In Steep Directions (Large Gradients):** If a parameter $\theta_i$ has historically large gradients, $s_{t,i}$ will be large. The effective learning rate $\frac{\eta}{\sqrt{s_{t,i}}}$ becomes small, acting as a **strong brake** (high friction) to prevent oscillations and divergence.
* **In Flat Directions (Small Gradients):** If a parameter $\theta_j$ has historically small gradients, $s_{t,j}$ will be small. The effective learning rate $\frac{\eta}{\sqrt{s_{t,j}}}$ remains relatively large, acting as a **light brake** (low friction) to allow rapid movement along the valley floor.

RMSProp dynamically re-scales the learning space, making the elliptical contours of the loss landscape appear more spherical to the optimizer, effectively lowering the condition number $\kappa$ and leading to faster, more stable convergence.

---

## **6.5 Adam — Adaptive Moment Estimation**

The **Adam** (Adaptive Moment Estimation) optimizer is arguably the most widely used algorithm in modern deep learning. It is a high-performance synthesis that combines the core strengths of the preceding methods: the **inertia** (momentum) of Chapter 6.2 and the **adaptive step sizing** (RMSProp) of Chapter 6.4.

Adam is not merely a combination; it is a refinement that leverages the estimated first and second *moments* of the stochastic gradient—hence its name.

---

### **The Algorithm: Fusing Dynamics**

Adam maintains and updates three primary vectors for each parameter $\theta_i$ at every time step $t$: the gradient $\nabla L_t$, the first moment estimate $m_t$, and the second raw moment estimate $v_t$.

1.  **Estimate First Moment (Momentum):** Accumulates a decaying average of past gradients.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L_t
$$

2.  **Estimate Second Moment (Adaptive Scaling):** Accumulates a decaying average of *squared* gradients.

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L_t)^2
$$

3.  **Bias Correction:** Both $m_t$ and $v_t$ are initialized near zero and are biased toward zero during the early steps. This step corrects the estimates to account for this initial bias.

$$
m_{\text{hat}} = m_t / (1-\beta_1^t)
$$

$$
v_{\text{hat}} = v_t / (1-\beta_2^t)
$$

4.  **Final Update Rule:** Combines the bias-corrected moments to compute the update.

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t - \eta \frac{m_{\text{hat}}}{\sqrt{v_{\text{hat}}} + \epsilon}
$$

Typical default hyperparameters are $\beta_1 = 0.9$ (retains $90\%$ of velocity), $\beta_2 = 0.999$ (retains $99.9\%$ of squared gradient history), and $\epsilon = 10^{-8}$.

---

### **Physical Analogy: The Smart Particle**

Adam is analogous to a **"smart particle"** or a sophisticated self-adjusting robot navigating the high-dimensional loss landscape.

* **$m_{\text{hat}}$ (The Navigator):** This term provides the **direction** and **inertia** (momentum). It averages the instantaneous forces to determine the overall direction of the valley floor, smoothing out the noise from SGD.
* **$\sqrt{v_{\text{hat}}}$ (The Damper):** This term dynamically adjusts the **friction** and **mass** for each coordinate. Like RMSProp, it applies heavy damping (a small effective $\eta$) in steep directions and light damping (a large effective $\eta$) in flat directions.

By dividing the momentum-based direction ($m_{\text{hat}}$) by the adaptive scale ($\sqrt{v_{\text{hat}}}$), Adam automatically scales its steps to be large when progress is consistently made (high momentum, low stiffness) and small when the parameter is thrashing against a steep wall.

---

### **Practical Strengths and Limitations**

Adam's widespread adoption stems from its reliability across many domains.

* **Robustness:** It requires very little hyperparameter tuning and generally achieves excellent results in high-noise environments (like deep learning with small mini-batches), sparse gradient regimes, and non-stationary problems.
* **Efficiency:** The computational cost per step is very low, requiring only a few extra operations per parameter compared to basic SGD.

A noted limitation is that while Adam is superb at *finding* a low-loss region quickly, its default $\sqrt{v_{\text{hat}}}$ scaling can sometimes lead to convergence toward **sharp minima** (Section 4.4) that generalize slightly worse than the **flat minima** found by well-tuned, non-adaptive optimizers like SGD with Momentum. This issue is typically addressed with minor variants (e.g., AdamW) or specialized learning rate schedules (Chapter 6.10).

!!! example "Adam as Unified Navigator"
```
Adam represents the synthesis of two fundamental ideas: momentum provides directional persistence (like a ball rolling with inertia), while adaptive scaling provides automatic brake adjustment (like ABS in a car). Consider navigating a mountain road with sharp turns (anisotropic ravine): momentum keeps you moving forward smoothly without jerking the wheel at every bump, while adaptive braking prevents skidding on steep sections. Together, they create a robust optimizer that handles diverse terrain without manual tuning.

```
---

## **6.6 Geometry of Adaptive Dynamics**

Adaptive methods like RMSProp and Adam (Sections 6.4 and 6.5) are not merely heuristics for picking better learning rates; they fundamentally **redefine the geometry** of the optimization problem. By dynamically scaling the gradient $\nabla L$ by the inverse square root of the accumulated squared gradients ($\mathbf{v}_{\text{hat}}^{-1/2}$), these algorithms effectively change the metric tensor of the parameter space.

---

### **Coordinate Re-scaling and the Modified Metric**

The core idea of adaptivity is to transform the anisotropic (elliptical) loss landscape into a more isotropic (spherical) one.

In standard Gradient Descent, the movement is measured using the standard Euclidean distance (or $L^2$ norm) in parameter space, $d\mathbf{\theta}^2 = \sum_i d\theta_i^2$. However, adaptive algorithms implicitly define a **new, non-Euclidean distance** $d\mathbf{\theta}^2_{\text{adaptive}}$ where steps along steep directions are inherently smaller in the context of the overall move.

The step size for each parameter $\theta_i$ is governed by:

$$
\Delta \theta_i \propto \frac{1}{\sqrt{v_{\text{hat},i}}} m_{\text{hat},i}
$$

This is equivalent to operating in a metric where the distance $d\mathbf{\theta}$ is weighted by the inverse of the parameter's average historical steepness:

$$
d\mathbf{\theta}^2_{\text{adaptive}} \sim (\text{diag}(\mathbf{v}_{\text{hat}}))^{-1} d\mathbf{\theta} \cdot d\mathbf{\theta}
$$

* **Interpretation:** The optimization trajectory is now performing an *isotropic* move within this **transformed, data-dependent space**. The parameters are no longer scaled equally; instead, they are rescaled such that $1$ unit of motion along a stiff dimension ($\theta_i$, where $v_{\text{hat},i}$ is large) is made numerically smaller than $1$ unit of motion along a sloppy dimension ($\theta_j$, where $v_{\text{hat},j}$ is small).

---

### **The Optimizer as a Geometric Learner**

This interpretation reveals that the optimizer is performing a form of **learning geometry**. The optimizer uses the history of gradients ($\mathbf{v}_{\text{hat}}$) to construct a localized, diagonal approximation of the curvature, acting as a **preconditioner** (Section 5.3).

Ideally, the optimal preconditioning matrix is the inverse Hessian $H^{-1}$ (Newton's method, Section 5.10). Since calculating $H^{-1}$ is intractable, adaptive methods settle for a sparse, diagonal approximation $\mathbf{v}_{\text{hat}}^{-1/2}$. This approximation does an excellent job of correcting for the differing scales (the ravine effect) by:
* **Compressing** the steps in stiff directions.
* **Stretching** the steps in sloppy directions.

---

### **Analogy to Natural Gradient Descent**

This dynamic re-scaling is closely related to the concept of **Natural Gradient Descent**.

In information geometry (Section 2.3), the distance between two probability distributions is measured by the **Fisher Information Matrix** $I(\mathbf{\theta})$. The Natural Gradient is defined as $I(\mathbf{\theta})^{-1} \nabla L(\mathbf{\theta})$, which ensures that the optimizer always takes the steepest possible step when distance is measured in terms of **statistical distinguishability** rather than arbitrary Euclidean units.

Adaptive methods approximate this principle. They aim to take steps that are **isotropic** in the transformed geometry defined by the data's historical variations. The optimizer is thereby taking the most efficient path along the underlying manifold of learned solutions.

---

## **6.7 Stability, Convergence, and Energy View**

The introduction of **Momentum** (inertia) and **Adaptive Scaling** (dynamic friction) transforms our optimization process from the simple, first-order, overdamped dynamics of basic Gradient Descent (Chapter 5) into a complex, second-order, underdamped system. This physical shift provides powerful guarantees about **stability** and **convergence**, which can be best understood by defining a total energy function, or **Hamiltonian**, for the optimization.

---

### **Defining the Effective Hamiltonian**

In classical mechanics, the state of a system is defined by its position $\mathbf{\theta}$ and its momentum $\mathbf{p} = m\mathbf{v}$, where $\mathbf{v}$ is velocity. The total energy, or **Hamiltonian ($\mathcal{H}$)**, is the sum of the potential energy (our loss, $L(\mathbf{\theta})$) and the kinetic energy (related to the particle's motion):

$$
\mathcal{H}(\mathbf{\theta}, \mathbf{p}) = L(\mathbf{\theta}) + \frac{1}{2m}|\mathbf{p}|^2
$$

Where $m$ is the effective mass of the optimizer's parameter vector.

* $L(\mathbf{\theta})$ is the **Potential Energy**. It pulls the system toward minima.
* $\frac{1}{2m}|\mathbf{p}|^2$ is the **Kinetic Energy**. It is introduced via momentum and allows the system to escape local traps and coast across flat regions.

The parameters $\mathbf{\theta}$ and velocity $\mathbf{v}$ (or momentum $\mathbf{p}$) now form the **phase space** of our optimization dynamical system.

---

### **Dissipation and Monotonic Convergence**

If this were an idealized physical system, $\mathcal{H}$ would be conserved. However, optimization, like any physical relaxation, requires energy to be *dissipated*. The Momentum term introduces a **damping force** (friction, $\gamma$) proportional to velocity.

The time derivative of the total energy $\mathcal{H}$ for a damped system shows that the energy is constantly being dissipated:

$$
\frac{d\mathcal{H}}{dt} = -\gamma |\mathbf{v}|^2 \le 0
$$

This fundamental result has two key implications for convergence:
* **Stability:** Since $d\mathcal{H}/dt$ is always non-positive, the total energy of the system is guaranteed to decrease (or stay constant) over time. This ensures the optimization is stable and does not diverge due to the added inertia, provided the damping (friction, $\gamma$) is appropriately balanced.
* **Convergence:** The dissipation ensures the system will eventually stop moving ($\mathbf{v} \to 0$) and settle at a **fixed point** where the potential energy $L(\mathbf{\theta})$ is at a minimum.

---

### **The Damped Oscillator Picture**

Momentum methods act as **damped oscillators** in the rugged potential $L(\mathbf{\theta})$.

* The system is given a \"kick\" by the gradient (force).
* It then oscillates toward the minimum due to its inertia (momentum).
* The oscillation is controlled and gradually reduced by the damping (friction coefficient $\gamma$).

A well-tuned Momentum optimizer (high $\beta$ for inertia, moderate $\eta$ for force) operates in the **critically damped** or **underdamped** regime. This is the ideal state where the system is guaranteed to converge to the low-energy fixed point (the minimum) faster than the fully overdamped approach of pure Gradient Descent, which lacks the energy to overcome plateaus.

By embracing this physical analogy of energy, momentum, and dissipation, we treat modern optimization as a robust, dynamical system converging to **attractors** (basins of minimal loss) in its high-dimensional phase space.

---

## **6.8 Comparing Methods on a Ravine Function**

The concepts of inertia (Momentum) and adaptive scaling (Adam/RMSProp) are best understood by visualizing their behavior on a challenging, anisotropic landscape. The "ravine" or "valley" function serves as the canonical stress test for any optimization algorithm.

---

### **The Ravine Test Function**

We use a simple, two-dimensional quadratic loss function that illustrates **ill-conditioned curvature**.

$$
L(\theta_1, \theta_2) = 0.5\theta_1^2 + 5\theta_2^2
$$

The loss surface defined by $L$ exhibits extreme **anisotropy**:
* **$\theta_1$ Direction (Sloppy/Flat):** The quadratic coefficient is $0.5$. Movement along this axis is slow, corresponding to a small eigenvalue of the Hessian.
* **$\theta_2$ Direction (Stiff/Steep):** The quadratic coefficient is $5$. Movement along this axis is rapid and highly sensitive, corresponding to a large eigenvalue.

The contour lines of this function are narrow, elongated ellipses, with the minimum at $(0, 0)$. The optimization challenge is to travel quickly along the flat $\theta_1$ axis without overshooting and diverging on the steep $\theta_2$ axis.

---

### **Trajectory Comparison**

By running Gradient Descent (GD), Momentum, and Adam from the same starting point, we observe three distinct dynamical behaviors:

| Algorithm | Mechanism | Observed Trajectory | Physical Interpretation |
| :--- | :--- | :--- | :--- |
| **Gradient Descent (GD)** | Overdamped relaxation. | **Zigzagging.** The trajectory oscillates violently across the steep $\theta_2$ axis while making agonizingly slow progress down the flat $\theta_1$ axis. | The gradient force is dominated by the steep direction, causing continuous overshooting and correction. |
| **Momentum** | Adds inertia (memory) $\beta$. | **Smoother Path.** The initial zigzagging is dampened because the momentum term $\beta \mathbf{v}_t$ averages out the orthogonal oscillations. The optimizer accelerates along the valley floor, reaching the minimum faster than GD. | Inertia allows the particle to ignore the short-term bumps (steep walls) and maintain direction along the main axis of descent. |
| **Adam** | Adaptive scaling and inertia. | **Direct Convergence.** The trajectory is the most efficient, quickly aligning with the valley floor and converging with minimal oscillation. | The optimizer automatically applies a small effective $\eta$ to the stiff $\theta_2$ direction and a large effective $\eta$ to the sloppy $\theta_1$ direction, transforming the ravine into a nearly isotropic bowl. |

---

### **The Role of Geometry and Dynamics**

This comparison demonstrates that the difficulty of optimization lies in the **ill-conditioned geometry** of the loss surface, not the local minimum value.
* **Momentum** improves performance by adjusting the *dynamics* (adding $\mathbf{v}$ to dampen motion).
* **Adam** improves performance by implicitly transforming the *geometry* (scaling the metric of the space).

Both approaches vastly outperform simple GD, confirming that modern optimization requires a model of motion richer than simple proportional friction.

??? question "Why Does Anisotropy Cause Zigzagging?"
```
In a ravine landscape, the gradient $\nabla L$ points perpendicular to the contour lines. For an elliptical ravine aligned with coordinate axes, the gradient has a large component in the steep $\theta_2$ direction and a small component in the flat $\theta_1$ direction. Since GD uses $\mathbf{\theta}_{t+1} = \mathbf{\theta}_t - \eta \nabla L$, it takes large steps perpendicular to the valley (overshooting across the steep walls) and tiny steps along the valley floor (slow progress toward minimum). The result is a zigzag path that wastes most updates correcting perpendicular overshoot rather than advancing toward the optimum.

```
---

## **6.9 Code Demo — Comparing GD, Momentum, and Adam**

This demonstration provides a practical implementation of the optimization methods discussed, visually contrasting their trajectories on the **ravine test function** $L(\theta_1, \theta_2) = 0.5\theta_1^2 + 5\theta_2^2$ (Section 6.8). This comparison vividly shows how inertia (Momentum) and adaptive scaling (Adam) overcome the slow, zigzagging behavior of pure Gradient Descent (GD).

The setup assumes a constant learning rate $\eta$ is used by all three methods for a fair, albeit challenging, comparison on this ill-conditioned surface.

---

```python
import numpy as np
import matplotlib.pyplot as plt

## --- 1. Define Loss Gradient (Ravine Function) ---

## L(x, y) = 0.5*x^2 + 5*y^2

## Gradient is: [dL/dx, dL/dy] = [x, 10*y]

def grad(theta):
    # theta is a 2D vector [x, y]
    x, y = theta
    return np.array([x, 10*y])

## --- 2. Helper Function to Run Optimization ---

def run_optimizer(update_func, theta0=np.array([3.0, 3.0]), steps=100):
    """Runs a given optimization update function for a set number of steps."""
    traj = [theta0.copy()]
    theta = theta0.copy()
    for _ in range(steps):
        # The update function is responsible for calculating the next theta
        theta = update_func(theta)
        traj.append(theta.copy())
    return np.array(traj)

## --- 3. Optimization Algorithms ---

eta = 0.05 # Learning rate, compromised by the stiff direction (y/theta_2)

## 3.1. Gradient Descent (GD)

def gd_update(theta):
    """Standard Gradient Descent: theta_t+1 = theta_t - eta * grad(theta_t)"""
    return theta - eta * grad(theta)

## 3.2. Momentum

v = np.zeros(2) # Global state: velocity vector
beta = 0.9      # Momentum coefficient (inertia)
def momentum_update(theta):
    """Momentum: v_t+1 = beta*v_t - eta*grad, theta_t+1 = theta_t + v_t+1"""
    # NOTE: The global 'v' must be updated within the function's closure context
    global v
    v = beta * v - eta * grad(theta)
    return theta + v

## 3.3. Adam (Simplified)

m = np.zeros(2)  # First moment estimate (momentum)
v2 = np.zeros(2) # Second moment estimate (squared gradient history)
b1, b2 = 0.9, 0.999
eps = 1e-8
## Use a mutable list to track time step t for bias correction (t is passed as [1] initially)

def adam_update(theta, t_counter=[1]):
    """Adam: Combines momentum (m) and adaptive scaling (v2) with bias correction."""
    global m, v2
    g = grad(theta)
    t = t_counter[0] # Current time step

    # 1. Update biased moment estimates
    m = b1 * m + (1-b1)*g
    v2 = b2 * v2 + (1-b2)*(g*g)

    # 2. Bias correction (required especially early on)
    m_hat = m / (1-b1**t)
    v_hat = v2 / (1-b2**t)

    # 3. Final adaptive update
    theta -= eta * m_hat / (np.sqrt(v_hat)+eps)
    t_counter[0] += 1 # Increment time step
    return theta

## --- 4. Run Trajectories ---

## Reset global state variables before each run

traj_gd = run_optimizer(gd_update)
v[:] = 0 # Reset momentum velocity state
traj_m = run_optimizer(momentum_update)
m[:] = 0; v2[:] = 0 # Reset Adam moment states
traj_a = run_optimizer(adam_update)

## --- 5. Visualization ---

## Plotting the loss contours

t1 = np.linspace(-4, 4, 100)
t2 = np.linspace(-4, 4, 100)
T1, T2 = np.meshgrid(t1, t2)
L_loss = 0.5 * T1**2 + 5 * T2**2

plt.figure(figsize=(9, 7))
CS = plt.contour(T1, T2, L_loss, levels=np.logspace(-1, 3, 20), cmap='magma')
plt.plot(traj_gd[:,0], traj_gd[:,1], '-o', label='GD (Zigzagging)', alpha=0.8, markersize=3, lw=1.5, color='royalblue')
plt.plot(traj_m[:,0], traj_m[:,1], '-o', label='Momentum (Coasting)', alpha=0.8, markersize=3, lw=1.5, color='darkorange')
plt.plot(traj_a[:,0], traj_a[:,1], '-o', label='Adam (Adaptive)', alpha=0.8, markersize=3, lw=1.5, color='mediumseagreen')

plt.scatter(0, 0, marker='*', s=300, color='gold', label='Minimum')
plt.scatter(traj_gd[0,0], traj_gd[0,1], marker='s', s=80, color='black', label='Start')

plt.title('Optimization Trajectories on a Ravine Function')
plt.xlabel(r'Parameter $\theta_1$ (Sloppy/Flat Direction)')
plt.ylabel(r'Parameter $\theta_2$ (Stiff/Steep Direction)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
```

---

**Interpretation**

The resulting visualization dramatically illustrates the effect of the added dynamics:

  * **GD (Blue):** The path is marked by high-amplitude **oscillations (zigzagging)** perpendicular to the optimal path. The step size $\eta=0.05$ is too large for the stiff $\theta_2$ direction, causing it to overshoot the optimal line repeatedly. Convergence is agonizingly slow.
  * **Momentum (Orange):** The trajectory is significantly **smoother and faster**. The inertia ($\beta=0.9$) dampens the perpendicular oscillations by averaging them out, allowing the optimizer to accelerate effectively along the flat valley floor (the $\theta_1$ direction).
  * **Adam (Green):** Adam is the **most direct** and fastest to converge. By estimating the high variance in $\theta_2$ and the low variance in $\theta_1$, it automatically applies a large effective $\eta$ to $\theta_1$ and a small one to $\theta_2$. This self-scaling transforms the effective geometry of the problem from an elliptical ravine into a nearly circular bowl, leading to rapid, direct descent.

This demo confirms that the principles of **inertia** (Momentum) and **adaptive scaling** (Adam) are crucial for navigating the anisotropic and ill-conditioned geometries common to all large-scale loss functions.

---

---

## **6.10 Practical Notes and Heuristics**

The shift from simple gradient methods to advanced dynamics like Adam introduces powerful capabilities, but also requires new practical considerations for achieving robust and efficient training. These heuristics bridge the gap between theoretical dynamics and empirical results.

---

### **Tuning Parameters: The Adam Defaults**

Adam's popularity stems partly from its robust default hyperparameter settings, which work well across a broad range of problems.

* **Learning Rate ($\eta$):** Typically starts around $10^{-3}$ or $10^{-4}$. While adaptive methods reduce $\eta$'s sensitivity, it remains the most critical parameter for overall convergence speed and final solution quality.
* **Momentum Decay ($\beta_1$):** Controls the exponential decay rate for the first moment (momentum term, $m_t$). The default of **$0.9$** is widely used, meaning 90% of the past momentum is retained at each step.
* **Squared Gradient Decay ($\beta_2$):** Controls the decay rate for the second moment (adaptive scaling term, $v_t$). The default of **$0.999$** is nearly universal, providing a very long memory for the local curvature estimate.
* **Epsilon ($\epsilon$):** A small constant added for numerical stability to prevent division by zero, typically **$10^{-8}$**.

---

### **Learning Rate Decay Schedules (Annealing)**

Despite using adaptive scaling, the learning rate $\eta$ should still be reduced over time—a process often called **annealing** (Section 5.7).

* **The Problem:** The stochastic nature of SGD and Adam causes parameters to perpetually oscillate around the minimum (Section 5.8). This oscillation, or "jiggling," is beneficial early on for escaping local traps, but it prevents precise convergence.
* **The Solution:** Gradually reducing $\eta$ **"cools" the system** (lowers the effective temperature, $T_{\text{eff}}$). As the temperature drops, the stochastic fluctuations dampen, allowing the system to settle into the precise bottom of the basin.
* **Advanced Schedules:** Modern schedules, like **Cosine Annealing with Warm Restarts (SGDR)**, cyclically vary the learning rate, allowing the optimizer to periodically "heat up" (large $\eta$) to escape sharp minima before "cooling down" (small $\eta$) to converge precisely in a new, potentially better basin.

---

### **Empirical Insight: Flat Minima and Generalization**

A key empirical discovery connects the dynamics of optimization to a model's ability to **generalize** to new data (Section 4.4).

* **Flat Minima:** A wide, flat basin in the loss landscape corresponds to a solution that is **robust** to small changes in parameters. Such solutions are less sensitive to noise in the training data and often generalize better.
* **Sharp Minima:** A narrow, deep, **spiky** minimum often corresponds to **overfitting**. The model has perfectly memorized the training set but is highly sensitive to new data.

Empirical evidence suggests that optimization paths that employ **high noise (small batch size)** or **momentum** tend to preferentially settle into **flatter basins**. This confirms the power of the stochastic/thermal analogy: the noise, acting as temperature, helps the system find the *thermodynamically stable* (widest, lowest-free-energy) minimum, which correlates strongly with the best generalization performance.

---

The practical implementation of these dynamic methods turns the training process into a sophisticated engineering task that constantly balances inertia, adaptive friction, and thermal noise to achieve the fastest and most robust convergence.

---

## **6.11 Takeaways & Bridge to Chapter 7**

This chapter completed the shift from the static map of the loss landscape (Chapter 4) to the rich, dynamic laws of motion that govern modern optimization, effectively transforming our learning process into a physics problem.

---

### **What We Accomplished in Chapter 6**

We moved beyond the limitations of simple, overdamped gradient descent by integrating concepts from classical mechanics and adaptive control.

* **Momentum (Inertia):** We introduced the concept of **inertia** into the optimizer's dynamics (Chapter 6.2). The resulting velocity vector $\mathbf{v}_t$ acts as memory, allowing the system to **coast** across flat plateaus and **dampen oscillations** in anisotropic ravines, leading to accelerated convergence.
* **Adaptive Scaling (Friction):** Algorithms like RMSProp and Adam (Chapter 6.4, 6.5) introduced **adaptive, per-parameter learning rates**. This mechanism learns the local curvature from the history of squared gradients, acting as a **dynamic friction** that automatically rescales the loss landscape, making it appear isotropic and overcoming the high condition number ($\kappa$) problem (Chapter 6.6).
* **Unified Dynamics:** Optimizers like Adam synthesize inertia and adaptivity, behaving as a **damped dynamical system** whose total energy is continuously dissipated until convergence is reached (Chapter 6.7).

$$
\mathcal{H} = L + \frac{1}{2m}|\mathbf{p}|^2
$$
* **Empirical Insight:** We established that the resulting stochastic dynamics preferentially settle into **flat minima**, which are physically analogous to **thermodynamically stable** states and empirically correlate with better model **generalization** (Chapter 6.10).

---

### **Bridge to Chapter 7: When Gradients Fail**

We have built sophisticated **deterministic-stochastic** optimizers, but their success still fundamentally relies on the **gradient $\nabla L$** being a meaningful guide. However, some landscapes break this reliance entirely:

1.  **Combinatorial Landscapes:** In problems where parameters are discrete ($\pm 1$, 0/1, integers, Chapter 8), the loss function is discontinuous. The gradient is zero or undefined everywhere, rendering gradient-based methods useless.
2.  **Rugged Glassy Landscapes:** In ultra-complex systems (like a spin glass, Chapter 4.3), the landscape is so rugged that a deterministic path is quickly trapped in a local minimum, separated from the global optimum by high energy barriers. The noise of SGD is often insufficient to jump these large barriers.

This difficulty motivates a strategic shift: instead of refining the **deterministic** components (momentum, adaptivity), we must weaponize the **stochastic** components.

In **Chapter 7: "Stochastic & Heuristic Optimization,"** we return to the most powerful exploration tool in physics: **controlled randomness**. We will explore heuristic methods, such as **Simulated Annealing**, that replace the continuous force of the gradient with temperature-dependent stochastic jumps. This physically-inspired approach allows us to escape deep local traps and explore the most rugged and complex landscapes, transforming the search problem into a thermodynamic annealing process.

We have learned how to roll down a hill; now, we learn how to **tunnel through mountains**.

## **References**

[1] Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *ICML*, 1139–1147.

[2] Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2). *Doklady AN USSR*, 269, 543–547.

[3] Tieleman, T., & Hinton, G. (2012). Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural Networks for Machine Learning*.

[4] Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.

[5] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*.

[6] Reddi, S. J., Kale, S., & Kumar, S. (2018). On the Convergence of Adam and Beyond. *ICLR*.

[7] Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*, 10(2), 251–276.