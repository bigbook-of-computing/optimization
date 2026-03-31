# **Chapter 5: Gradient Methods: The Workhorses (Codebook)**

## Project 1: Visualizing Learning Rate and Dynamics ($\mathcal{L(\theta) = \frac{1}{2}\theta^2}$)

---

### Definition: Visualizing Learning Rate and Dynamics

The goal of this project is to implement the core **deterministic Gradient Descent (GD)** algorithm on the simplest convex loss function, $L(\theta) = \frac{1}{2}\theta^2$, and observe how the crucial control parameter, the **learning rate ($\eta$)**, affects the stability and convergence speed of the optimization path.

### Theory: Gradient Descent as the Law of Motion

Gradient Descent is the fundamental law of motion for optimization, analogous to an **overdamped physical system** where velocity ($\propto \Delta\mathcal{\theta}$) is proportional to the force ($\propto -\nabla L$).

The update rule is a first-order difference equation:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

For the **quadratic loss** $L(\theta) = \frac{1}{2}\theta^2$, the analytic gradient is $\nabla L(\theta) = \theta$. The update simplifies to:

$$\theta_{t+1} = \theta_t (1 - \eta)$$

The trajectory of the system is governed entirely by the parameter $\eta$:

  * **Stable Convergence:** Requires $0 < \eta < 2$. The system spirals inward toward the minimum ($\theta^*=0$).
  * **Critical Stability:** Occurs at $\eta=2$. The system oscillates but remains bounded.
  * **Divergence:** Occurs when $\eta \ge 2$. The system overshoots and moves exponentially away from the minimum.

---

### Extensive Python Code and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt

## ====================================================================

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

---

## Project 2: Tracking Dynamics in an Anisotropic Landscape

---

### Definition: Tracking Dynamics in an Anisotropic Landscape

The goal is to simulate deterministic Gradient Descent on an **anisotropic** (non-spherical) quadratic loss surface, $L(\theta_1, \theta_2) = \frac{1}{2}\theta_1^2 + 5\theta_2^2$. The objective is to visualize the **inefficient, zigzagging path** caused by the landscape's unequal curvature (stiffness/sloppiness).

### Theory: Anisotropy and Zigzagging

The anisotropic landscape is an **elliptical bowl** where the gradient is steep in the $\theta_2$ direction ($8\theta_2$) and shallow in the $\theta_1$ direction ($2\theta_1$). This difference in curvature defines the **condition number** of the Hessian.

The **Gradient Descent (GD) path** is locally perpendicular to the contour lines (path of steepest descent).

  * If the path is far from the minimum, the gradient vector is dominated by the steep $\theta_2$ term, causing the path to rapidly overshoot the minimum along the steep axis.
  * The path then corrects, overshoots again, and **zigzags** inefficiently toward the minimum.

This failure motivates the need for advanced optimization methods (like momentum) that use the Hessian's second-order information to account for the landscape's curvature.

---

### Extensive Python Code and Visualization

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

---

## Project 3: SGD as a Thermal Ensemble (Noise and Distribution)

---

### Definition: SGD as a Thermal Ensemble

The goal is to implement **Stochastic Gradient Descent (SGD)** and demonstrate that the inherent **sampling noise ($\xi_t$)** in the gradient prevents the parameters from converging to a single point. Instead, the final parameter values ($\theta$) form a **statistical ensemble** around the true minimum.

### Theory: Stochastic Gradient Descent (SGD)

SGD uses a **mini-batch** of data (rather than the full dataset) to estimate the true gradient. The calculated gradient $\nabla L_{\text{mb}}$ is a noisy approximation of the true gradient $\nabla L_{\text{true}}$:

$$\nabla L_{\text{mb}} = \nabla L_{\text{true}} + \xi_t$$

Where $\xi_t$ is the **stochastic noise** (sampling error).

The **SGD update rule** is analogous to the **Langevin equation** in statistical physics, where $\xi_t$ acts as **thermal energy**:

$$\theta_{t+1} = \theta_t - \eta (\nabla L_{\text{true}} + \xi_t)$$

The system settles into a **non-equilibrium steady state** where the optimizer's drift toward the minimum is balanced by the thermal diffusion, resulting in a **statistical ensemble** (a distribution with finite variance) centered at the minimum.

---

### Extensive Python Code and Visualization

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
Ensemble Mean (Simulated): μ ≈ 0.0202
Ensemble Variance (σ²): Var(θ) ≈ 0.0258

Conclusion: The simulation shows that the parameters under SGD do not converge to a single point but form a **statistical ensemble** (a distribution) centered near the true minimum. This is the result of the thermal noise (ξ_t) preventing the system from reaching absolute zero temperature, confirming that SGD operates as a high-dimensional physical system in a **non-equilibrium steady state**.
```

---

## Project 4: Energy Dissipation Check

---

### Definition: Energy Dissipation Check

The goal is to numerically verify the **Lyapunov stability property** of deterministic Gradient Descent: that the loss (energy) must **monotonically decrease** over time. Tracking the loss function $L_t$ validates that the algorithm is a physical process of **energy dissipation**.

### Theory: The Lyapunov Condition

For a continuous system evolving under deterministic Gradient Descent, the time derivative of the loss function must be negative (or zero at the minimum):

$$\frac{dL}{dt} = \frac{\partial L}{\partial \mathcal{\theta}} \cdot \frac{d\mathcal{\theta}}{dt}$$

Since the motion is defined by the negative gradient, $\frac{d\mathcal{\theta}}{dt} = -\eta \nabla L$, the change in loss is:

$$\frac{dL}{dt} = \nabla L \cdot (-\eta \nabla L) = -\eta \| \nabla L \|^2$$

As $\|\nabla L\|^2$ is always non-negative, the loss must always decrease unless the system is at a critical point ($\nabla L = 0$). Thus, the discrete loss must be **monotonically non-increasing** for a stable learning rate ($\eta \le \eta_{\max}$).

$$L_{t+1} \le L_t$$

---

### Extensive Python Code and Visualization

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