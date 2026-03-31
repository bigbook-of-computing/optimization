# **Chapter 5: Gradient Methods: The Workhorses (Workbook)**

The goal of this chapter is to establish **gradient descent** as the fundamental law of motion for optimization, interpreting the learning rate, convergence, and noise as essential components of a physical dynamical system.

| Section | Topic Summary |
| :--- | :--- |
| **5.1** | The Principle of Steepest Descent |
| **5.2** | Learning Rate and Stability |
| **5.3** | Gradient Descent in Vector Spaces |
| **5.4** | Stochastic Gradient Descent (SGD) |
| **5.5** | Mini-Batch and Variance Trade-Off |
| **5.6** | Gradient Descent as Relaxation Dynamics |
| **5.7** | Practical Aspects and Diagnostics |
| **5.8–5.11**| Worked Example, Code Demo, and Takeaways |



### 5.1 The Principle of Steepest Descent

> **Summary:** The core idea of optimization is to move in the direction of **steepest descent**. This direction is given by the **negative gradient** vector, $-\nabla L(\mathcal{\theta})$. **Gradient Descent** is an iterative algorithm that moves parameters $\mathcal{\theta}$ in this direction with a step size $\eta$ (the **learning rate**). This process is analogous to the **overdamped relaxation** of a physical particle in a highly viscous medium, where its motion is dominated by friction and the restoring force ($\mathbf{F} = -\nabla L$).

#### Quiz Questions

!!! note "Quiz"
    **1. The gradient descent update rule $\mathcal{\theta}_{t+1} = \mathcal{\theta}_t - \eta \nabla L(\mathcal{\theta}_t)$ is the numerical integration of which continuous physical process?**
    
    * **A.** The $F=ma$ law.
    * **B.** **Gradient flow, $\frac{d\mathcal{\theta}}{dt} = -\nabla L(\mathcal{\theta})$**. (**Correct**)
    * **C.** Conservation of momentum.
    * **D.** The Hamiltonian dynamics.
    
!!! note "Quiz"
    **2. In the physical analogy of optimization, the term **overdamped relaxation** is used because the particle's motion is assumed to be dominated by which physical force?**
    
    * **A.** Inertia.
    * **B.** Gravity.
    * **C.** **Friction (viscosity)**. (**Correct**)
    * **D.** Magnetic force.
    
---

!!! question "Interview Practice"
    **Question:** Gradient Descent transforms the algebraic problem of solving $\nabla L = 0$ into a **dynamical system**. What are the key components of this dynamical system in terms of optimization terminology?
    
    **Answer Strategy:**
    * **State Space:** The parameter space $\mathbb{R}^{D_\theta}$.
    * **State:** The parameter vector $\mathcal{\theta}(t)$.
    * **Potential/Energy:** The loss function $L(\mathcal{\theta})$.
    * **Equation of Motion:** The gradient descent algorithm itself.
    
---

### 5.2 Learning Rate and Stability

> **Summary:** The **learning rate ($\eta$)** is a hyperparameter that controls the step size and acts as the **damping coefficient** of the system. The stability of gradient descent is governed by the curvature $a$ of the landscape; convergence requires **$0 < \eta < 1/a$**. If $\eta$ is too large ($\eta > 1/a$), the system becomes unstable and **diverges**. A single global $\eta$ is a poor compromise in high-dimensional anisotropic spaces.

#### Quiz Questions

!!! note "Quiz"
    **1. Based on the 1D stability analysis of $L(\theta) = a\theta^2$, which condition causes the optimization trajectory to oscillate with **explosively growing** amplitude?**
    
    * **A.** $0 < \eta < 1/(2a)$.
    * **B.** $\eta = 1/(2a)$.
    * **C.** **$\eta > 1/a$**. (**Correct**)
    * **D.** $\eta$ is set to zero.
    
!!! note "Quiz"
    **2. A physical system that is **overdamped** in the context of gradient descent is analogous to a simulation where the learning rate ($\eta$) is:**
    
    * **A.** Too large, causing instability.
    * **B.** **Too small, causing the optimization to "creep" slowly towards the minimum**. (**Correct**)
    * **C.** Exactly equal to 1.
    * **D.** Oscillating around the minimum.
    
---

!!! question "Interview Practice"
    **Question:** The stability analysis dictates that the learning rate $\eta$ must be smaller than $1/a$, where $a$ is the curvature. How does this requirement create a speed bottleneck when the optimization landscape is **anisotropic** (containing both stiff and sloppy directions)?.
    
    **Answer Strategy:** The global learning rate $\eta$ must be set small enough to be stable in the **stiffest direction** (the direction with the largest curvature, $\lambda_{\max}$). If $\eta$ is too large, the optimizer would diverge along this steep wall. However, this same small $\eta$ is then **far too small** for the flat, **sloppy directions** ($\lambda_{\min}$). Consequently, the optimizer makes agonisingly slow progress along the solution path (the valley floor), and convergence is bottlenecked by the high condition number $\kappa = \lambda_{\max}/\lambda_{\min}$.
    
---

### 5.3 Gradient Descent in Vector Spaces

> **Summary:** In high-dimensional vector spaces, the **anisotropy** of the loss surface creates narrow "ravines" or "canyons". The gradient tends to point *across* the ravine (perpendicular to the valley floor) rather than along it, causing the optimizer to **zigzag** inefficiently. The degree of difficulty is quantified by the **condition number ($\kappa$)** of the Hessian ($H$), which is the ratio $\lambda_{\max}/\lambda_{\min}$. **Preconditioning** aims to solve this by linearly transforming the coordinate system to make the landscape appear isotropic (spherical).

#### Quiz Questions

!!! note "Quiz"
    **1. The primary structural issue that causes the gradient descent path to exhibit a severe "zigzagging" behavior is:**
    
    * **A.** A noisy gradient estimate.
    * **B.** **Anisotropic curvature (ravines)**. (**Correct**)
    * **C.** A zero gradient norm.
    * **D.** A very large learning rate $\eta$.
    
!!! note "Quiz"
    **2. For an anisotropic loss landscape, the difficulty of the optimization is numerically quantified by the **condition number ($\kappa$)**, defined as:**
    
    * **A.** The step size $\eta$ divided by the gradient $\nabla L$.
    * **B.** The mean $\mu$ divided by the standard deviation $\sigma$.
    * **C.** **The ratio of the largest to the smallest eigenvalue of the Hessian, $\lambda_{\max}/\lambda_{\min}$**. (**Correct**)
    * **D.** The learning rate multiplied by the iteration count.
    
---

!!! question "Interview Practice"
    **Question:** The concept of **preconditioning** seeks to normalize the geometry of the optimization landscape. Explain this process using the analogy of a parameter space ruler.
    
    **Answer Strategy:** Preconditioning is the process of finding a linear transformation that converts the anisotropic ravine geometry into a perfect, isotropic (spherical) bowl. The analogy is that we are **renormalizing the parameter space ruler**. In the stiff directions (large $\lambda_k$), we use a shorter, slower ruler (small effective $\eta$); in the sloppy directions (small $\lambda_k$), we use a longer, faster ruler (large effective $\eta$). The goal is to make a standard unit step move the same "effective distance" in all directions, making the solution path direct and eliminating zigzagging.
    
---

### 5.4 Stochastic Gradient Descent (SGD)

> **Summary:** **Batch Gradient Descent (BGD)** is computationally infeasible for large datasets ($N$). **Stochastic Gradient Descent (SGD)** solves this by approximating the full gradient with the gradient from a single randomly selected sample or **mini-batch** $B$. This stochastic gradient is an **unbiased estimator** of the true gradient. The resulting high **gradient noise** acts as an **effective temperature ($T>0$)** that allows the optimizer to **escape shallow local minima** (Brownian motion) and find better solutions.

#### Quiz Questions

!!! note "Quiz"
    **1. The BGD algorithm requires computing the gradient over the entire dataset ($N$). Why is SGD's noisy gradient estimate, based on a single sample, still statistically valid?**
    
    * **A.** Because the single step always points directly to the global minimum.
    * **B.** **Because the expected value of the stochastic gradient is equal to the true full-batch gradient**. (**Correct**)
    * **C.** Because the Hessian matrix is zero.
    * **D.** Because it is only used on convex functions.
    
!!! note "Quiz"
    **2. In the SGD analogy, the primary benefit of the **gradient noise** is that it provides the optimizer with:**
    
    * **A.** Reduced variance near the minimum.
    * **B.** **Effective thermal energy to jump over small energy barriers**. (**Correct**)
    * **C.** Guaranteed convergence to the global minimum.
    * **D.** A lower condition number $\kappa$.
    
---

!!! question "Interview Practice"
    **Question:** Gradient Descent (BGD) is a deterministic relaxation, analogous to a system at $T=0$. SGD is a stochastic relaxation, analogous to a system at $T>0$. Describe the major functional consequence of the **zero-temperature** environment for BGD in the non-convex landscapes of Chapter 4.
    
    **Answer Strategy:** In a non-convex landscape, a $T=0$ (zero noise) BGD optimizer has **no thermal energy** to overcome barriers. Consequently, it is deterministically guaranteed to **get permanently stuck** in the very first shallow local minimum it rolls into, preventing it from exploring the landscape to find the deeper, better quality minima that often lead to better generalization.
    
---

### 5.5 Mini-Batch and Variance Trade-Off

> **Summary:** **Mini-Batch Gradient Descent ($1 < B \ll N$)** is the practical compromise between the stability of BGD and the speed of SGD. The batch size $B$ acts as a **thermostat**, controlling the **effective temperature ($T$)** of the optimization. **Small batch size ($B$)** leads to high variance, acting as high $T$ that encourages **exploration** and finds flatter, more generalizable minima. **Large batch size ($B$)** leads to low variance, acting as low $T$, which risks getting trapped in sharp, local minima.

#### Quiz Questions

!!! note "Quiz"
    **1. In the physical analogy where batch size $B$ controls the effective temperature $T$ of optimization, which characteristic is associated with a **low $T$ (large $B$)** optimization?**
    
    * **A.** High variance and better exploration.
    * **B.** **Low variance and risk of getting trapped in sharp minima**. (**Correct**)
    * **C.** The ability to use a very small learning rate $\eta$.
    * **D.** Very high gradient noise.
    
!!! note "Quiz"
    **2. The primary reason practitioners often prefer to use a small mini-batch size ($B$) over the full batch ($N$) is because the noise acts as a regularizer that helps the optimizer find solutions that:**
    
    * **A.** Converge faster along the valley floor.
    * **B.** **Generalize better to unseen data**. (**Correct**)
    * **C.** Have a lower condition number $\kappa$.
    * **D.** Are mathematically guaranteed to be the global minimum.
    
---

!!! question "Interview Practice"
    **Question:** The noise variance in SGD scales roughly as $\text{Var}(\nabla L_B) \propto 1/B$. Explain why this relationship justifies calling the mini-batch size $B$ the **thermostat** for the optimization process.
    
    **Answer Strategy:** The noise variance $\text{Var}(\nabla L_B)$ dictates the magnitude of the random "thermal kicks" the optimizer receives. Since thermal energy ($T$) is proportional to fluctuations, **$\text{Var}(\nabla L_B)$ directly controls the effective temperature $T$**. Therefore, by simply increasing $B$, the system becomes "colder" (less noise, less exploration); by decreasing $B$, the system becomes "hotter" (more noise, more exploration), allowing $B$ to function as a precise control knob for the energy dynamics of the system.
    
---

### 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core concepts of gradient dynamics and stability.

### Project 1: Testing Learning Rate Stability (1D)

* **Goal:** Numerically demonstrate the stability constraint ($\eta < 1/a$) and the effect of oscillation.
* **Setup:** Use the quadratic loss $L(\theta) = \frac{1}{2}a\theta^2$ with $a=2$ (True stability bound is $\eta < 1/a = 0.5$).
* **Steps:**
    1.  Run the deterministic gradient descent loop ($\theta_{t+1} = \theta_t (1 - a\eta)$) for three cases: $\eta_{\text{safe}}=0.2$, $\eta_{\text{oscillating}}=0.8$, and $\eta_{\text{divergent}}=1.2$.
    2.  Plot the trajectory $\theta(t)$ for all three cases.
* ***Goal***: Show that $\eta_{\text{safe}}$ converges smoothly, $\eta_{\text{oscillating}}$ oscillates but converges, and $\eta_{\text{divergent}}$ oscillates with rapidly growing amplitude, confirming the stability boundary.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt

# ====================================================================

## 1. Setup Loss and Gradient Functions

## ====================================================================

## L(theta) = 0.5 * theta^2 (Convex Bowl)

def loss_func(theta):
    return 0.5 * theta**2

## Gradient: dL/d(theta) = theta

def gradient(theta):
    return theta

## ====================================================================

## 2. Gradient Descent Simulation

## ====================================================================

def run_gd(theta_start, learning_rate, max_steps=50):
    """Runs deterministic GD for the simple quadratic loss."""
    theta_history = np.zeros(max_steps)
    theta = theta_start

    for t in range(max_steps):
        # Law of Motion: theta_new = theta_old - eta * gradient(theta_old)
        grad = gradient(theta)
        theta_new = theta - learning_rate * grad

        theta_history[t] = theta
        theta = theta_new

        # Stop early if converged
        if np.abs(theta) < 1e-6:
            theta_history[t+1:] = theta
            break

    return theta_history

## --- Simulation Scenarios ---

THETA_START = 4.0
MAX_STEPS = 10

## Scenario A: Optimal Learning Rate (Fast Convergence)

ETA_A = 0.5
THETA_A = run_gd(THETA_START, ETA_A, MAX_STEPS)

## Scenario B: High Learning Rate (Divergence)

ETA_B = 2.1
THETA_B = run_gd(THETA_START, ETA_B, MAX_STEPS)

## Scenario C: Critical Learning Rate (Oscillation)

ETA_C = 2.0
THETA_C = run_gd(THETA_START, ETA_C, MAX_STEPS)

## ====================================================================

## 3. Visualization

## ====================================================================

t_steps = np.arange(MAX_STEPS)

fig, ax = plt.subplots(figsize=(8, 5))

## Plot the three scenarios

ax.plot(t_steps, THETA_A, 'o-', color='darkgreen', label=f'Optimal $(\\eta={ETA_A})$: Convergence')
ax.plot(t_steps, THETA_B, 's--', color='darkred', label=f'Too High $(\\eta={ETA_B})$: Divergence')
ax.plot(t_steps, THETA_C, '^:', color='purple', label=f'Critical $(\\eta={ETA_C})$: Oscillation')

## Annotate the minimum

ax.axhline(0, color='k', linestyle='-', lw=0.8)

## Labeling and Formatting

ax.set_title(r'Gradient Descent Dynamics: Effect of Learning Rate $\eta$ on $L(\theta) = \frac{1}{2}\theta^2$')
ax.set_xlabel('Iteration Step $t$')
ax.set_ylabel(r'Parameter Value $\theta_t$')
ax.set_ylim(-10, 10)
ax.legend()
ax.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- Gradient Descent Dynamics Summary ---")
print("Scenario A (\u03b7=0.5): The parameter decays exponentially to the minimum (\u03b8=0).")
print("Scenario B (\u03b7=2.1): The parameter overshoots the minimum and diverges (numerical instability).")
print("Scenario C (\u03b7=2.0): The parameter oscillates between +4.0 and -4.0 (critical stability limit).")
```
**Sample Output:**
```python
--- Gradient Descent Dynamics Summary ---
Scenario A (η=0.5): The parameter decays exponentially to the minimum (θ=0).
Scenario B (η=2.1): The parameter overshoots the minimum and diverges (numerical instability).
Scenario C (η=2.0): The parameter oscillates between +4.0 and -4.0 (critical stability limit).
```


### Project 2: Simulating Anisotropic Zigzagging

* **Goal:** Demonstrate how the anisotropic geometry of a ravine causes inefficient convergence.
* **Setup:** Use the 2D quadratic ravine function $L(\theta_1, \theta_2) = \frac{1}{2}\theta_1^2 + 5\theta_2^2$ (Hessian $\lambda_{\max}=10, \lambda_{\min}=1$). Start at $\mathcal{\theta}_0 = [10, 10]$ and use a stable learning rate (e.g., $\eta=0.1$).
* **Steps:**
    1.  Implement the full 2D deterministic gradient descent: $\mathcal{\theta}_{t+1} = \mathcal{\theta}_t - \eta \nabla L$.
    2.  Plot the optimization trajectory in the ($\theta_1, \theta_2$) parameter space.
* ***Goal***: Show that the trajectory spends most of its time making small steps along the $\theta_1$ (sloppy) axis and makes large, oscillating steps along the $\theta_2$ (stiff) axis, illustrating the inefficient "zigzag" pattern.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

## ====================================================================

## 1. Setup Loss and Gradient Functions

## ====================================================================

## Loss Function (Anisotropic Bowl): L = 0.5*theta1^2 + 5*theta2^2

## Note: The gradient for L = 0.5*theta1^2 + 5*theta2^2 is (theta1, 10*theta2)

def L_aniso(t1, t2):
    return 0.5 * t1**2 + 5 * t2**2

## Gradient: dL/d(theta) = (theta1, 10*theta2)

def grad_L_aniso(t1, t2):
    dL_dt1 = t1
    dL_dt2 = 10 * t2
    return np.array([dL_dt1, dL_dt2])

## ====================================================================

## 2. Gradient Descent Simulation (with Zigzagging)

## ====================================================================

MAX_STEPS = 20
ETA = 0.09 # High enough to cause visible zigzagging but not divergence

## Starting position (off-axis to demonstrate anisotropy)

THETA_START = np.array([3.0, 0.8])

## Store trajectory

theta_history = np.zeros((MAX_STEPS, 2))
theta = THETA_START.copy()

for t in range(MAX_STEPS):
    theta_history[t] = theta

    # Calculate gradient
    grad = grad_L_aniso(theta[0], theta[1])

    # Update rule
    theta_new = theta - ETA * grad
    theta = theta_new

## Final minimum point

theta_min = np.array([0.0, 0.0])

## ====================================================================

## 3. Visualization

## ====================================================================

fig, ax = plt.subplots(figsize=(8, 6))

## Plot 1: Contour Map of the Anisotropic Loss

t1_plot, t2_plot = np.meshgrid(np.linspace(-3.5, 3.5, 100), np.linspace(-1, 1, 100))
L_surface = L_aniso(t1_plot, t2_plot)
levels = np.logspace(0, np.log10(L_surface.max()), 15) # Log-spaced contours for better visibility
plt.contour(t1_plot, t2_plot, L_surface, levels=levels, colors='gray', alpha=0.6)

## Plot 2: The Gradient Descent Trajectory

plt.plot(theta_history[:, 0], theta_history[:, 1], 'r-', lw=2, label='GD Trajectory')
plt.plot(theta_history[:, 0], theta_history[:, 1], 'bo', markersize=5, label='GD Steps')

## Highlight the start and end

plt.plot(THETA_START[0], THETA_START[1], 'go', markersize=8, label='Start')
plt.plot(theta_min[0], theta_min[1], 'r*', markersize=12, label='Minimum')

## Labeling and Formatting

ax.set_title(f'Gradient Descent Path in an Anisotropic Loss Landscape ($\\eta={ETA}$)')
ax.set_xlabel(r'$\theta_1$ (Sloppy Direction)')
ax.set_ylabel(r'$\theta_2$ (Stiff Direction)')
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- Anisotropic Dynamics Summary ---")
print(f"Starting Point: ({THETA_START[0]}, {THETA_START[1]})")
print(f"Final Point: ({theta_history[-1, 0]:.4f}, {theta_history[-1, 1]:.4f})")
print("Observation: The GD path exhibits characteristic **zigzagging** (oscillation) across the steep (\u03b8_2) direction while making slow, steady progress along the shallow (\u03b8_1) direction. This confirms that GD's local, first-order rule is highly inefficient on anisotropic surfaces, motivating the use of advanced techniques that incorporate second-order information (curvature).")
```
**Sample Output:**
```python
--- Anisotropic Dynamics Summary ---
Starting Point: (3.0, 0.8)
Final Point: (0.4999, 0.0000)
Observation: The GD path exhibits characteristic **zigzagging** (oscillation) across the steep (θ_2) direction while making slow, steady progress along the shallow (θ_1) direction. This confirms that GD's local, first-order rule is highly inefficient on anisotropic surfaces, motivating the use of advanced techniques that incorporate second-order information (curvature).
```


### Project 3: Visualizing SGD as Thermal Motion

* **Goal:** Simulate the SGD model to visually confirm the "stochastic equilibrium" around the minimum.
* **Setup:** Use the noisy quadratic loss from the demo: $\nabla L_t = 2\theta_t + \xi_t$, $\eta=0.05$.
* **Steps:**
    1.  Run the SGD simulation for 1000 steps (longer run to stabilize the distribution).
    2.  Plot a **histogram** of the final 500 parameter values $\theta(t)$ recorded during the simulation.
* ***Goal***: Show that the distribution of $\theta$ is centered near the true minimum ($\theta=0$) but has a finite, measurable variance, confirming that the noise creates a **thermal ensemble** of parameter states rather than converging to a single point.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## ====================================================================

## 1. Setup Loss and Gradient (Quadratic Loss with Stochastic Noise)

## ====================================================================

## True Loss: L_true = 0.5 * theta^2 (Minimum at theta_true = 0)

THETA_TRUE = 0.0

## SGD Gradient: g_SGD = theta + xi (where xi is the sampling noise)

def gradient_sgd(theta, sigma_noise):
    # True gradient: theta
    grad_true = theta
    # Sampling noise: xi_t ~ N(0, sigma_noise^2)
    noise = np.random.normal(0, sigma_noise)
    return grad_true + noise

## ====================================================================

## 2. SGD Simulation (Tracking the Ensemble)

## ====================================================================

MAX_STEPS = 1000
ETA = 0.05 # Learning rate
SIGMA_NOISE = 1.0 # Standard deviation of the stochastic gradient noise

## Start point

THETA_START = 4.0

## Store trajectory

theta_history = np.zeros(MAX_STEPS)
theta = THETA_START

for t in range(MAX_STEPS):
    theta_history[t] = theta

    # Calculate noisy gradient
    grad = gradient_sgd(theta, SIGMA_NOISE)

    # SGD Update Rule
    theta_new = theta - ETA * grad
    theta = theta_new

## Use the last 500 steps as the stationary ensemble

ENSEMBLE_SIZE = 500
theta_ensemble = theta_history[-ENSEMBLE_SIZE:]

## Ensemble Statistics

MU_ENSEMBLE = np.mean(theta_ensemble)
VAR_ENSEMBLE = np.var(theta_ensemble)

## ====================================================================

## 3. Visualization and Analysis

## ====================================================================

## Plot 1: Ensemble Distribution (Histogram)

plt.figure(figsize=(8, 5))

## Plot histogram of final states (the thermal ensemble)

plt.hist(theta_ensemble, bins=30, density=True, color='purple', alpha=0.6,
         label='Final Parameter Ensemble')

## Plot the theoretical distribution center

plt.axvline(THETA_TRUE, color='red', linestyle='--', label='True Minimum $(\\theta^*=0)$')

## Overlay a Gaussian with the ensemble's calculated mean and variance

x_plot = np.linspace(theta_ensemble.min(), theta_ensemble.max(), 100)
pdf_ensemble = norm.pdf(x_plot, MU_ENSEMBLE, np.sqrt(VAR_ENSEMBLE))
plt.plot(x_plot, pdf_ensemble, 'k-', lw=2, label='Fitted Thermal Distribution')

## Labeling and Formatting

plt.title('SGD as a Thermal Ensemble: Parameter Distribution at Steady State')
plt.xlabel(r'Parameter Value $\theta$')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- SGD Thermal Ensemble Analysis ---")
print(f"True Minimum (Analytic): \u03b8* = {THETA_TRUE}")
print(f"Ensemble Mean (Simulated): \u03bc \u2248 {MU_ENSEMBLE:.4f}")
print(f"Ensemble Variance (\u03c3\u00b2): Var(\u03b8) \u2248 {VAR_ENSEMBLE:.4f}")

print("\nConclusion: The simulation shows that the parameters under SGD do not converge to a single point but form a **statistical ensemble** (a distribution) centered near the true minimum. This is the result of the thermal noise (\u03be_t) preventing the system from reaching absolute zero temperature, confirming that SGD operates as a high-dimensional physical system in a **non-equilibrium steady state**.")
```
**Sample Output:**
```python
--- SGD Thermal Ensemble Analysis ---
True Minimum (Analytic): θ* = 0.0
Ensemble Mean (Simulated): μ ≈ -0.0170
Ensemble Variance (σ²): Var(θ) ≈ 0.0202

Conclusion: The simulation shows that the parameters under SGD do not converge to a single point but form a **statistical ensemble** (a distribution) centered near the true minimum. This is the result of the thermal noise (ξ_t) preventing the system from reaching absolute zero temperature, confirming that SGD operates as a high-dimensional physical system in a **non-equilibrium steady state**.
```


### Project 4: Energy Dissipation Check

* **Goal:** Numerically verify the Lyapunov property of gradient descent: that energy (loss) must monotonically decrease over time.
* **Setup:** Use the anisotropic quadratic loss from Project 2 ($L(\theta_1, \theta_2) = \frac{1}{2}\theta_1^2 + 5\theta_2^2$) and run a stable deterministic simulation ($\eta=0.1$).
* **Steps:**
    1.  Implement the loss function $L(\mathcal{\theta})$.
    2.  Track and record the loss $L_t$ at every step.
* ***Goal***: Plot the loss $L_t$ versus iteration $t$. The curve must be **monotonically decreasing** (never increase), visually verifying the mathematical proof that $\frac{dL}{dt} = -\gamma \|\nabla L\|^2 \le 0$.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

## ====================================================================

## 1. Setup Loss and Gradient Functions (Anisotropic Loss)

## ====================================================================

## Use the anisotropic loss from Project 2

def L_aniso(t1, t2):
    # L = 0.5*theta1^2 + 5*theta2^2
    return 0.5 * t1**2 + 5 * t2**2

def grad_L_aniso(t1, t2):
    dL_dt1 = t1
    dL_dt2 = 10 * t2
    return np.array([dL_dt1, dL_dt2])

## ====================================================================

## 2. Gradient Descent Simulation with Loss Tracking

## ====================================================================

MAX_ITER = 50
ETA = 0.05 # Stable and low learning rate
THETA_START = np.array([3.0, 0.8])

## Store cost (Energy)

J_history = []
theta = THETA_START.copy()

for t in range(MAX_ITER):
    # Calculate current Loss (Energy)
    J_history.append(L_aniso(theta[0], theta[1]))

    # Calculate gradient
    grad = grad_L_aniso(theta[0], theta[1])

    # Update rule
    theta_new = theta - ETA * grad
    theta = theta_new

    # Stop condition
    if np.linalg.norm(grad) < 1e-4:
        break

## ====================================================================

## 3. Visualization and Convergence Check

## ====================================================================

plt.figure(figsize=(8, 5))

## Plot the Monotonic Descent of the Objective Function J (Energy)

plt.plot(J_history, 'r-', lw=2, markersize=5)

plt.title(f'Energy Dissipation in Deterministic Gradient Descent ($\\eta={ETA}$)')
plt.xlabel('Iteration Number')
plt.ylabel('Loss $L_t$ (Energy)')
plt.grid(True)
plt.show()

## --- Analysis Summary ---

## Check for monotonicity (loss never increases)

is_monotonic = np.all(np.diff(J_history) <= 1e-9)

print("\n--- Energy Dissipation Check ---")
print(f"Initial Loss (Energy): L0 = {J_history[0]:.4f}")
print(f"Final Loss (Energy): L_final = {J_history[-1]:.4f}")
print(f"Loss Monotonically Decreasing? {is_monotonic}")

print("\nConclusion: The plot shows a smooth, monotonically decreasing loss function, confirming the **Lyapunov stability** of Gradient Descent. This behavior is the direct numerical evidence of **energy dissipation**—the system constantly sheds energy (loss) as it follows the negative gradient to find the stable equilibrium state (the minimum).")
```
**Sample Output:**
```python
--- Energy Dissipation Check ---
Initial Loss (Energy): L0 = 7.7000
Final Loss (Energy): L_final = 0.0295
Loss Monotonically Decreasing? True

Conclusion: The plot shows a smooth, monotonically decreasing loss function, confirming the **Lyapunov stability** of Gradient Descent. This behavior is the direct numerical evidence of **energy dissipation**—the system constantly sheds energy (loss) as it follows the negative gradient to find the stable equilibrium state (the minimum).
```