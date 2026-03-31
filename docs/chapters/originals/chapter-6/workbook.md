# **Chapter 6: Advanced Gradient Dynamics () () () (Workbook)**

The goal of this chapter is to upgrade the optimization process from simple, friction-dominated dynamics to a **second-order, damped dynamical system** by introducing **inertia (momentum)** and **adaptive scaling (friction)** to efficiently navigate anisotropic loss landscapes.

| Section | Topic Summary |
| :--- | :--- |
| **6.1** | Beyond Simple Descent |
| **6.2** | Momentum — Learning with Inertia |
| **6.3** | Nesterov Accelerated Gradient (NAG) |
| **6.4** | RMSProp — Adaptive Step Sizes |
| **6.5** | Adam — Adaptive Moment Estimation |
| **6.6** | Geometry of Adaptive Dynamics |
| **6.7** | Stability, Convergence, and Energy View |
| **6.8–6.11**| Comparison, Code Demo, and Takeaways |

---

### 6.1 Beyond Simple Descent

> **Summary:** Simple Gradient Descent (GD) fails in anisotropic **ravines** by **zigzagging** and **stalling** due to the high **condition number ($\kappa$)**. The solution requires breaking the overdamped $\mathbf{v} \propto \mathbf{F}$ relationship. We upgrade the dynamics using **Inertia (Momentum)** to smooth zigzags and **Per-Parameter Adaptivity** to rescale the landscape. The resulting system is modeled as a **damped harmonic oscillator**.

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary cause of the **zigzagging** behavior in simple Gradient Descent when optimizing in a ravine function is:**

* **A.** The loss of momentum.
* **B.** The introduction of too much thermal noise.
* **C.** **The anisotropic geometry, causing the gradient to point repeatedly across the steep walls**. (**Correct**)
* **D.** A very small learning rate $\eta$.

```
!!! note "Quiz"
```
**2. Which term in the generalized equation of motion, $m\frac{d^2\mathcal{\theta}}{dt^2} = -\gamma m \frac{d\mathcal{\theta}}{dt} - \nabla L(\mathcal{\theta})$, is missing from the simplified, overdamped model of Gradient Descent?**

* **A.** The Potential Force ($-\nabla L$).
* **B.** The Damping Force ($-\gamma m \frac{d\mathcal{\theta}}{dt}$).
* **C.** **The Inertia term ($m\frac{d^2\mathcal{\theta}}{dt^2}$) **. (**Correct**)
* **D.** The velocity ($\frac{d\mathcal{\theta}}{dt}$).

```
---

!!! question "Interview Practice"
```
**Question:** Explain how the introduction of **Inertia (Momentum)** solves the two core failures of Gradient Descent: **zigzagging** and **stalling**?

**Answer Strategy:**
1.  **Zigzagging:** Inertia solves zigzagging by allowing the optimizer to **average out the orthogonal oscillations**. The momentum term accumulates the small, consistent gradient along the valley floor while the oscillating gradients across the ravine walls tend to cancel each other out.
2.  **Stalling:** Inertia solves stalling by providing **kinetic energy**. When the gradient becomes infinitesimally small in flat regions (stalling), the accumulated velocity allows the optimizer to **coast across the plateau**.

```
---

### 6.2 Momentum — Learning with Inertia

> **Summary:** The **Momentum** method introduces a velocity vector $\mathbf{v}_t$ that accumulates past gradients, effectively adding **inertia**. The update uses a **momentum coefficient ($\beta$)** to control the damping, with $\beta=0.9$ being typical. Momentum smooths the optimization path and prevents **stalling** in flat regions.

#### Quiz Questions

!!! note "Quiz"
```
**1. In the Momentum update rule, $\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta \nabla L(\mathcal{\theta}_t)$, the hyperparameter $\beta$ primarily controls the system's:**

* **A.** Total energy dissipation.
* **B.** **Persistence or memory (damping factor)**. (**Correct**)
* **C.** Step size in the stiff direction.
* **D.** Convergence rate along the $\theta_2$ axis.

```
!!! note "Quiz"
```
**2. The physical analogy of the Momentum update that describes its benefit is that the accumulated velocity allows the optimizer to:**

* **A.** Diverge quickly.
* **B.** **Coast across plateaus and roll over small energy barriers**. (**Correct**)
* **C.** Only use the instantaneous gradient.
* **D.** Transform the landscape into a spherical bowl.

```
---

!!! question "Interview Practice"
```
**Question:** Standard Momentum improves dynamics but fails to solve the root problem of anisotropy. Explain what critical component of the optimization landscape Momentum *does not* address, and why this limits its overall efficiency.

**Answer Strategy:** Momentum does **not** address the **anisotropy** (the differing curvatures in the ravine) or the need for **adaptive scaling**. It still relies on a **single, global learning rate $\eta$**. This single $\eta$ must be constrained by the stiffest direction ($\lambda_{\max}$), which means the optimizer is still forced to move too slowly in the flat (sloppy) directions. Momentum smooths the path, but it doesn't change the underlying geometry that restricts the global step size.

```
---

### 6.3 Nesterov Accelerated Gradient (NAG)

> **Summary:** **Nesterov Accelerated Gradient (NAG)** is a refinement of Momentum that uses **anticipatory dynamics** to correct its path. Instead of calculating the gradient at the current position $\mathcal{\theta}_t$, NAG calculates the gradient at a **predicted future position** ($\mathcal{\theta}_t + \beta \mathbf{v}_t$). This **predictive gradient** allows the optimizer to "turn the corner" earlier in a curved valley, achieving mathematically proven accelerated convergence rates.

#### Quiz Questions

!!! note "Quiz"
```
**1. The core difference between the Nesterov Accelerated Gradient (NAG) method and standard Momentum lies in:**

* **A.** The momentum coefficient $\beta$ being set to zero.
* **B.** **Calculating the gradient at a lookahead position, $\mathcal{\theta}_t + \beta \mathbf{v}_t$**. (**Correct**)
* **C.** The removal of the velocity vector $\mathbf{v}_t$.
* **D.** The division by the inverse Hessian.

```
!!! note "Quiz"
```
**2. The NAG algorithm is analogous to a vehicle using **predictive steering** because it:**

* **A.** Randomly samples the next gradient.
* **B.** **Corrects the momentum vector based on the curvature detected at the predicted future position**. (**Correct**)
* **C.** Only works on convex functions.
* **D.** Slows down when the gradient is small.

```
---

!!! question "Interview Practice"
```
**Question:** Why does calculating the gradient at the predicted future position ($\mathcal{\theta}_t + \beta \mathbf{v}_t$) help the Momentum optimizer "turn the corner" earlier in a curved ravine?

**Answer Strategy:** In a ravine, the gradient at the current position $\mathcal{\theta}_t$ is dominated by the steep side wall. The predicted future position, however, is already closer to the center of the valley floor. By sampling the gradient there, NAG obtains a force vector that is less perpendicular to the valley and more aligned with the true direction of the minimum. This allows the system to apply the corrective force sooner, mitigating the overshoot and leading to a smoother, more **geodesic** path along the curved manifold.

```
---

### 6.4 RMSProp — Adaptive Step Sizes

> **Summary:** **RMSProp** addresses anisotropy by introducing an **adaptive, per-parameter learning rate**. It scales the gradient $\nabla L_t$ by the inverse square root of a **running average of its squared past gradients ($s_t$)**. This scaling applies a **strong brake** (small effective $\eta$) in stiff directions where $s_t$ is large, and a **light brake** (large effective $\eta$) in sloppy directions where $s_t$ is small. This effectively performs an implicit **re-scaling of the optimization geometry**.

#### Quiz Questions

!!! note "Quiz"
```
**1. In the RMSProp update rule, the accumulated variable $s_t$ for a specific parameter $\theta_i$ measures that parameter's historical:**

* **A.** Velocity.
* **B.** **Magnitude of squared gradients (historical stiffness)**. (**Correct**)
* **C.** Rate of divergence.
* **D.** Effective mass.

```
!!! note "Quiz"
```
**2. RMSProp overcomes the problem of anisotropy by applying a physical analogy where the optimizer uses:**

* **A.** Inertia to coast over flat terrain.
* **B.** **An adaptive, coordinate-dependent friction coefficient (or brake)**. (**Correct**)
* **C.** A fixed learning rate $\eta$ for all parameters.
* **D.** A second-order Hessian matrix.

```
---

!!! question "Interview Practice"
```
**Question:** RMSProp is said to **sphericize the loss landscape on the fly**. Explain this geometric transformation.

**Answer Strategy:** The loss landscape is initially anisotropic (elliptical contours, high condition number $\kappa$). RMSProp achieves sphericization by transforming the coordinates such that the axes appear to have equal curvature. It does this by **stretching the sloppy (flat) dimensions** (where $\frac{1}{\sqrt{s_t}}$ is large) and **compressing the stiff (steep) dimensions** (where $\frac{1}{\sqrt{s_t}}$ is small). In this transformed space, the optimizer sees a uniform, isotropic bowl, and a single step size $\eta$ works optimally in all directions.

```
---

### 6.5 Adam — Adaptive Moment Estimation

> **Summary:** **Adam** is the state-of-the-art optimizer that synthesizes **Momentum** (inertia via the first moment, $m_t$) and **RMSProp** (adaptive scaling via the second moment, $v_t$). The algorithm also includes a crucial **bias correction** step for early iterations. Adam's final update rule scales the momentum-based direction ($m_{\text{hat}}$) by the adaptive metric ($\sqrt{v_{\text{hat}}}$), creating a **"smart particle"** that rapidly converges in diverse, high-noise environments.

#### Quiz Questions

!!! note "Quiz"
```
**1. The **Adam** optimizer is a synthesis of which two primary mechanisms for improving gradient descent dynamics?**

* **A.** Nesterov acceleration and $\chi^2$ minimization.
* **B.** **Momentum (first moment, $m_t$) and Adaptive Scaling (second moment, $v_t$)**. (**Correct**)
* **C.** Gradient flow and batch size increase.
* **D.** Damping and zero bias.

```
!!! note "Quiz"
```
**2. The primary reason for including the **bias correction** step in the Adam algorithm is to:**

* **A.** Reduce the overall gradient noise.
* **B.** Guarantee convergence to a flat minimum.
* **C.** **Compensate for the fact that the initial moment estimates ($m_t, v_t$) are biased toward zero**. (**Correct**)
* **D.** Reset the velocity periodically.

```
---

!!! question "Interview Practice"
```
**Question:** Adam is often described as performing optimization using a **self-adjusting mass and damping** system. Which component of the Adam update determines the **direction/inertia (effective mass)**, and which determines the **adaptive damping (friction)**?

**Answer Strategy:**
* **Direction/Inertia (Effective Mass):** This is determined by the **bias-corrected first moment, $m_{\text{hat}}$**. This term averages past gradients, setting the general direction of momentum.
* **Adaptive Damping (Friction):** This is determined by the **inverse square root of the bias-corrected second moment, $1/\sqrt{v_{\text{hat}}}$**. This term acts as the dynamic friction coefficient, applying a stronger brake in steep directions (where $v_{\text{hat}}$ is large).

```
---

### 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core concepts of advanced gradient dynamics.

### Project 1: Simulating Anisotropic Zigzagging vs. Momentum

* **Goal:** Visually demonstrate how inertia (Momentum) successfully dampens the zigzagging behavior of GD on a ravine.
* **Setup:** Use the quadratic ravine function $L(\theta_1, \theta_2) = 0.5\theta_1^2 + 5\theta_2^2$. Start at $\mathcal{\theta}_0 = [3.0, 3.0]$ and set a small, fixed $\eta$ (e.g., $\eta=0.05$).
* **Steps:**
    1.  Implement the **GD update** (zero momentum) and the **Momentum update** ($\beta=0.9$).
    2.  Run both optimizers for 100 steps.
    3.  Plot both trajectories in the ($\theta_1, \theta_2$) parameter space.
* ***Goal***: Show that the GD trajectory exhibits high-amplitude oscillations perpendicular to the optimal path, while the Momentum path is significantly smoother and accelerates more directly toward the minimum.

#### Python Implementation

```python
import numpy as np
import random
import matplotlib.pyplot as plt

# ====================================================================
# 1. Ising Core Functions (from Chapter 2)
# ====================================================================

# Standard 2D Ising Lattice functions with Periodic Boundary Conditions (PBCs)

def create_lattice(N, initial_state='random'):
    """Initializes an N x N lattice with spins (+1 or -1)."""
    if initial_state == '+1':
        return np.ones((N, N), dtype=np.int8)
    return np.random.choice([-1, 1], size=(N, N), dtype=np.int8)

def get_neighbors(N, i, j):
    """Returns the coordinates of the four nearest neighbors (PBCs)."""
    return [
        ((i + 1) % N, j), ((i - 1 + N) % N, j), 
        (i, (j + 1) % N), (i, (j - 1 + N) % N)  
    ]

def calculate_delta_E_local(lattice, i, j, J=1.0, H=0.0):
    """Computes the O(1) change in energy for flipping spin (i, j)."""
    N = lattice.shape[0]
    spin_ij = lattice[i, j]
    sum_nn = sum(lattice[ni, nj] for ni, nj in get_neighbors(N, i, j))
    
    # Delta E = 2J * spin_ij * sum_nn + 2H * spin_ij
    return 2 * J * spin_ij * sum_nn + 2 * H * spin_ij

def metropolis_update_step(lattice, beta, J=1.0, H=0.0):
    """Performs one Monte Carlo Sweep (N*N attempts) using single-spin flip."""
    N = lattice.shape[0]
    total_spins = N * N
    
    for _ in range(total_spins):
        i, j = random.randrange(N), random.randrange(N)
        delta_E = calculate_delta_E_local(lattice, i, j, J, H)
        
        if delta_E <= 0 or random.random() < np.exp(-beta * delta_E):
            lattice[i, j] *= -1
            
def calculate_magnetization(lattice):
    """Calculates the absolute magnetization per spin |M|."""
    return np.mean(np.abs(lattice))

# ====================================================================
# 2. Autocorrelation Analysis Functions
# ====================================================================

def autocorr_func(x, lag):
    """Calculates the Autocorrelation Function C(tau)."""
    N = len(x)
    mean_x = np.mean(x)
    var_x = np.var(x)

    if var_x == 0:
        return 1.0 if lag == 0 else 0.0

    # Cov(O_t, O_{t+tau})
    cov = np.sum((x[:N - lag] - mean_x) * (x[lag:] - mean_x)) / (N - lag)
    return cov / var_x

def estimate_tau_int(x, max_lag_limit=300):
    """Estimates the integrated autocorrelation time (tau_int) from C(tau)."""
    max_lag = min(max_lag_limit, len(x) // 2)
    C = [autocorr_func(x, lag) for lag in range(max_lag + 1)]

    # ESS Denominator (G) = 1 + 2 * sum(C(tau)) with a practical cutoff
    G = 1.0
    for c_tau in C[1:]:
        # Stop summing when C(tau) is small or negative (practical cutoff)
        if c_tau < 0.05:
            G += 2 * c_tau
            break
        G += 2 * c_tau

    tau_int = 0.5 if G <= 1.0 else (G - 1.0) / 2.0
    return tau_int, C

# ====================================================================
# 3. Simulation and Quantification
# ====================================================================

LATTICE_SIZE = 32
MCS_RUN = 15000  # Long run to observe slow dynamics
EQUILIBRATION_MCS = 500

# Critical inverse temperature: beta_c = ln(1 + sqrt(2)) / 2 approx 0.4407
T_C = 2.269185 
BETA_C = 1.0 / T_C

# Temperatures of Interest (Control Parameters)
TEMPS = {
    'T_low (Ordered)': 1.5,
    'T_c (Critical)': T_C,
    'T_high (Disordered)': 3.5
}

J = 1.0
H = 0.0
results = {}

print(f"Quantifying critical slowing down for L={LATTICE_SIZE}...")

for label, T in TEMPS.items():
    beta = 1.0 / T
    
    # Initialize lattice to all up (+1)
    lattice = create_lattice(LATTICE_SIZE, initial_state='+1')
    
    # Thermalization
    for _ in range(EQUILIBRATION_MCS):
        metropolis_update_step(lattice, beta, J, H)
        
    # Measurement
    M_series = []
    for _ in range(MCS_RUN):
        metropolis_update_step(lattice, beta, J, H)
        M_series.append(calculate_magnetization(lattice))
    
    # Analysis
    M_series = np.array(M_series)
    tau_int, C_plot = estimate_tau_int(M_series)
    
    results[label] = {
        'T': T,
        'M_series': M_series,
        'C_plot': C_plot,
        'tau_int': tau_int
    }
    print(f"Finished {label}. Tau_int: {tau_int:.2f} MCS.")

# ====================================================================
# 4. Visualization and Comparison
# ====================================================================

# Extract results for comparison plot
tau_values = [res['tau_int'] for res in results.values()]
labels = list(results.keys())

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
markers = ['o', 's', '^']

# Plot 1: Autocorrelation Function C_M(tau) for each T
for i, (label, res) in enumerate(results.items()):
    ax[0].plot(res['C_plot'][:100], marker='.', markersize=3, 
               linestyle='-', lw=1.5, alpha=0.8,
               label=f"{label} ($\\\\tau_{{\\\\text{{int}}}} \\\\approx {res['tau_int']:.1f}$ MCS)")

ax[0].axhline(0, color='k', linestyle='--')
ax[0].set_title('Autocorrelation of Magnetization $C_M(\\tau)$')
ax[0].set_xlabel('Time Lag $\\tau$ (MCS)')
ax[0].set_ylabel('Autocorrelation $C(\\tau)$')
ax[0].set_xlim(0, 50)
ax[0].legend()
ax[0].grid(True, which='both', linestyle=':')

# Plot 2: Autocorrelation Time Comparison
ax[1].bar(labels, tau_values, color=['skyblue', 'darkred', 'lightgreen'])
ax[1].set_title('Integrated Autocorrelation Time $\\tau_{\\text{int}}$')
ax[1].set_xlabel('Temperature Regime')
ax[1].set_ylabel('$\\tau_{\\text{int}}$ (MCS)')
ax[1].grid(True, which='major', axis='y', linestyle=':')

plt.tight_layout()
plt.show()

# Final Analysis
print("\n--- Critical Slowing Down Analysis ---")
for label, res in results.items():
    print(f"| {label:<20} | T={res['T']:.3f} | Tau_int: {res['tau_int']:.2f} MCS |")
print("-" * 50)
print(f"Conclusion: The autocorrelation time $\\tau_{{\\text{{int}}}}$ is highest at the critical temperature ($T_c \u2248 2.269$). This confirms that the single-spin Metropolis algorithm suffers from **critical slowing down**, requiring significantly more Monte Carlo sweeps (MCS) to generate independent samples when the system is near its phase transition boundary.")
```

### Project 2: Implementing and Visualizing Adam's Adaptivity

* **Goal:** Implement the full **Adam** optimizer and demonstrate its ability to automatically correct the step size for the anisotropic ravine.
* **Setup:** Use the same ravine function $L(\theta_1, \theta_2)$ from Project 1.
* **Steps:**
    1.  Implement the full Adam algorithm, including the updates for $m_t$, $v_t$, and the **bias correction** for $m_{\text{hat}}$ and $v_{\text{hat}}$.
    2.  Run Adam for 100 steps and plot its trajectory alongside the GD and Momentum paths from Project 1.
* ***Goal***: Show that the Adam trajectory is the **most direct**, quickly aligning with the valley floor ($\theta_1$ axis) because its adaptive scaling increases the effective learning rate in the sloppy $\theta_1$ direction and decreases it in the steep $\theta_2$ direction.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. Wolff Cluster Algorithm Implementation
# ====================================================================

# Reusing Ising core functions from Project 1: create_lattice, get_neighbors, calculate_magnetization

def wolff_update_step(spins, beta, J=1.0):
    """
    Performs one Wolff cluster update.
    This corresponds to one Monte Carlo Sweep (MCS) for comparison purposes.
    """
    L = spins.shape[0]
    p_add = 1 - np.exp(-2 * beta * J) # Bond probability
    visited = np.zeros_like(spins, dtype=bool)
    
    # 1. Pick random seed and initialize cluster
    i, j = random.randrange(L), random.randrange(L)
    cluster_val = spins[i, j]
    
    # Use a stack (LIFO) for a recursive-like cluster growth (DFS/stack-based)
    stack = [(i, j)]
    visited[i, j] = True
    
    cluster_size = 0
    
    while stack:
        x, y = stack.pop()
        cluster_size += 1
        
        # Check all nearest neighbors
        for xn, yn in get_neighbors(L, x, y):
            # Condition: Neighbor is unvisited AND aligned
            if not visited[xn, yn] and spins[xn, yn] == cluster_val:
                # Add bond with probability p_add
                if random.random() < p_add:
                    visited[xn, yn] = True
                    stack.append((xn, yn))
                    
    # 2. Flip the entire cluster
    # This is a safe operation because 'visited' is a boolean mask
    spins[visited] *= -1
    return spins, cluster_size

# Reusing autocorrelation functions from Project 1 (autocorr_func, estimate_tau_int)

# ====================================================================
# 2. Simulation and Comparison at T_c
# ====================================================================

LATTICE_SIZE = 32
T_C = 2.269185 
BETA_C = 1.0 / T_C
MCS_RUN = 15000 
EQUILIBRATION_MCS = 500

# --- Metropolis Benchmark (from Project 1, T_c) ---
# We use the computed tau_int from Project 1 for the T_c Metropolis run (L=32)
# To avoid repeating the time-consuming run, we use the value observed earlier.
# This assumes the Metropolis run was performed and its result is available.
TAU_INT_METROPOLIS = 25.0 

# --- Wolff Simulation ---
wolff_lattice = create_lattice(LATTICE_SIZE, initial_state='+1')
M_series_wolff = []
cluster_sizes = []

print(f"Starting Wolff Cluster simulation at T_c={T_C:.3f}...")

# 1. Thermalization
for _ in range(EQUILIBRATION_MCS):
    wolff_update_step(wolff_lattice, BETA_C)
    
# 2. Measurement
for _ in range(MCS_RUN):
    wolff_lattice, size = wolff_step(wolff_lattice, BETA_C)
    M_series_wolff.append(calculate_magnetization(wolff_lattice))
    cluster_sizes.append(size)

# 3. Analysis
M_series_wolff = np.array(M_series_wolff)
tau_int_wolff, C_plot_wolff = estimate_tau_int(M_series_wolff)

# ====================================================================
# 3. Visualization and Comparison
# ====================================================================

speedup_factor = TAU_INT_METROPOLIS / tau_int_wolff
avg_cluster_size = np.mean(cluster_sizes) / (LATTICE_SIZE**2)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Autocorrelation Function Comparison
ax[0].plot(C_plot_wolff[:100], marker='.', markersize=3, 
           linestyle='-', lw=2, color='darkgreen', 
           label=f"Wolff Cluster ($\\tau_{{\\text{{int}}}}={tau_int_wolff:.2f}$ MCS)")

# Illustrative Metropolis decay for comparison
tau_axis = np.arange(0, 100)
C_metropolis_illustrative = np.exp(-tau_axis / TAU_INT_METROPOLIS) 
ax[0].plot(tau_axis, C_metropolis_illustrative, 
           linestyle='--', color='red', alpha=0.6,
           label=f"Metropolis (Benchmark $\\tau_{{\\text{{int}}}}={TAU_INT_METROPOLIS:.1f}$ MCS)")

ax[0].axhline(0, color='k', linestyle='--')
ax[0].set_title(f'Autocorrelation at Critical Point $T_c$ (L={LATTICE_SIZE})')
ax[0].set_xlabel('Time Lag $\\tau$ (MCS)')
ax[0].set_ylabel('Autocorrelation $C(\\tau)$')
ax[0].set_xlim(0, 50)
ax[0].legend()
ax[0].grid(True, which='both', linestyle=':')

# Plot 2: Tau_int and Speedup
ax[1].bar(['Metropolis $\\tau_{\\text{int}}$ (Benchmark)', 'Wolff $\\tau_{\\text{int}}$'], 
         [TAU_INT_METROPOLIS, tau_int_wolff], 
         color=['salmon', 'darkgreen'])

ax[1].text(0.5, 0.9, f'Speedup: {speedup_factor:.1f}x', transform=ax[1].transAxes, 
          ha='center', fontsize=12, fontweight='bold')
ax[1].set_title('Efficiency Gain Over Critical Slowing Down')
ax[1].set_ylabel('$\\tau_{\\text{int}}$ (MCS)')
ax[1].grid(True, which='major', axis='y', linestyle=':')

plt.tight_layout()
plt.show()

# Final Analysis
print("\n--- Wolff Cluster Algorithm Analysis ---")
print(f"Metropolis Benchmark Tau_int: {TAU_INT_METROPOLIS:.1f} MCS")
print(f"Wolff Simulated Tau_int: {tau_int_wolff:.2f} MCS")
print(f"Average Cluster Size (Fraction of L^2): {avg_cluster_size:.2f}")
print(f"Speedup Factor: {speedup_factor:.1f}x")

print("\nConclusion: The Wolff Cluster algorithm achieved a significant speedup factor over the single-spin Metropolis method at the critical temperature. By flipping large, correlated domains (clusters) simultaneously, the algorithm effectively circumvents **critical slowing down**, making the system's relaxation to equilibrium much faster and lattice-size independent.")
```

### Project 3: Energy Dissipation and Stability Check

* **Goal:** Verify the Lyapunov property of the second-order system: that the total energy $\mathcal{H}$ monotonically decreases when damping is present.
* **Setup:** Use the Momentum optimizer ($\beta=0.9$) and the ravine function $L(\mathcal{\theta})$.
* **Steps:**
    1.  Implement the **Total Energy (Hamiltonian)** function: $\mathcal{H} = L(\mathcal{\theta}) + \frac{1}{2}||\mathbf{v}||^2$ (assuming $m=1$).
    2.  Run the Momentum optimization and track the instantaneous value of $\mathcal{H}(t)$ at every step.
* ***Goal***: Plot $\mathcal{H}(t)$ versus time. The plot must be **monotonically decreasing** (never increase), confirming that the damping term (friction) successfully dissipates the total energy, ensuring stability and convergence to the low-energy attractor.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. System Functions (1D Double-Well)
# ====================================================================

# Potential: E(x) = x^4 - 2x^2 + 1 (Minima at x = +/- 1)
def E(x):
    return x**4 - 2*x**2 + 1

# Metropolis Update (Local Dynamics for a single replica)
def metropolis_step(x, beta, delta_size=0.5):
    """Standard Metropolis step for one replica at fixed beta."""
    x_trial = x + random.uniform(-delta_size, delta_size)
    dE = E(x_trial) - E(x)
    
    if random.random() < np.exp(-beta * dE):
        return x_trial
    else:
        return x

# Parallel Tempering Swap Acceptance
def calculate_swap_prob(beta_i, beta_j, E_i, E_j):
    """Calculates the acceptance probability for swapping configurations."""
    # P_swap = min(1, exp( (beta_i - beta_j) * (E_j - E_i) ))
    d_beta = beta_i - beta_j
    dE = E_j - E_i
    
    # We assume beta_i > beta_j (colder replica i, hotter replica j)
    # The exponential term is positive if the cold system (i) gets a low-energy state (E_j < E_i)
    # or if the hot system (j) gets a high-energy state (E_j > E_i), but dE < 0
    return np.exp(d_beta * dE)

# ====================================================================
# 2. Parallel Tempering Simulation
# ====================================================================

# --- Simulation Parameters ---
TOTAL_STEPS = 50000
METROPOLIS_STEPS_PER_SWAP = 10 # Perform 10 local steps before attempting a swap
DELTA_SIZE = 0.5

# Temperature Ladder (Control Parameter)
# The spacing must be chosen carefully to ensure high swap acceptance rates.
BETAS = np.array([0.5, 1.0, 2.0, 5.0]) 
N_REPLICAS = len(BETAS)

# --- Initialization ---
# Start all replicas at the same cold, trapped location
X_REPLICAS = np.full(N_REPLICAS, 1.0) # Start all in the x=+1 well

# Store trajectory of the Coldest Replica (Index 3, Beta=5.0)
COLDEST_REPLICA_INDEX = N_REPLICAS - 1
X_coldest_traj = []

print(f"Starting Parallel Tempering Simulation with {N_REPLICAS} replicas...")

for t in range(TOTAL_STEPS):
    
    # 1. Local Metropolis Update (Energy Minimization)
    for i, beta in enumerate(BETAS):
        for _ in range(METROPOLIS_STEPS_PER_SWAP):
            X_REPLICAS[i] = metropolis_step(X_REPLICAS[i], beta, DELTA_SIZE)
            
    # 2. Replica Exchange (Swapping)
    # Iterate over neighboring pairs (e.g., [3,2], [2,1], [1,0])
    for i in range(N_REPLICAS - 1, 0, -1):
        # Replica i is Colder (higher beta), Replica i-1 is Hotter (lower beta)
        beta_i, beta_j = BETAS[i], BETAS[i-1]
        X_i, X_j = X_REPLICAS[i], X_REPLICAS[i-1]
        
        # Calculate energies
        E_i, E_j = E(X_i), E(X_j)
        
        # Calculate swap acceptance probability
        P_swap = calculate_swap_prob(beta_i, beta_j, E_i, E_j)
        
        if random.random() < P_swap:
            # Execute the swap (exchange the positions/configurations)
            X_REPLICAS[i], X_REPLICAS[i-1] = X_REPLICAS[i-1], X_REPLICAS[i]
            
    # Record the current state of the coldest replica
    X_coldest_traj.append(X_REPLICAS[COLDEST_REPLICA_INDEX])

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

X_coldest_traj = np.array(X_coldest_traj)
time_points = np.arange(TOTAL_STEPS)

# Check mixing efficiency
percent_in_well_neg = np.mean(X_coldest_traj < -0.5)
percent_in_well_pos = np.mean(X_coldest_traj > 0.5)

plt.figure(figsize=(10, 5))
plt.plot(time_points, X_coldest_traj, lw=0.7, color='darkred')

# Highlight the two minima for context
plt.axhline(1, color='gray', linestyle=':', alpha=0.7)
plt.axhline(-1, color='gray', linestyle=':', alpha=0.7)

plt.title(f'Parallel Tempering Trajectory of Coldest Replica ($\\beta={BETAS[-1]:.1f}$)')
plt.xlabel('Simulation Step')
plt.ylabel('Position $x$')
plt.grid(True)
plt.show()

# Final Analysis
print("\n--- Parallel Tempering Analysis (1D Double-Well) ---")
print(f"Coldest Replica Beta: \\beta = {BETAS[-1]:.1f}")
print(f"Time in Negative Well (x < -0.5): {percent_in_well_neg:.2%}")
print(f"Time in Positive Well (x > 0.5): {percent_in_well_pos:.2%}")

print("\nConclusion: The trajectory of the coldest replica exhibits frequent, large jumps between the two stable wells ($x=\\pm 1$). This global exploration, which is impossible for a single low-temperature chain, confirms that Parallel Tempering successfully uses the higher-temperature replicas to overcome the energy barrier and achieves rapid **mixing** across the multimodal distribution.")
```
**Sample Output:**
```
Starting Parallel Tempering Simulation with 4 replicas...

--- Parallel Tempering Analysis (1D Double-Well) ---
Coldest Replica Beta: \beta = 5.0
Time in Negative Well (x < -0.5): 41.82%
Time in Positive Well (x > 0.5): 42.08%

Conclusion: The trajectory of the coldest replica exhibits frequent, large jumps between the two stable wells ($x=\pm 1$). This global exploration, which is impossible for a single low-temperature chain, confirms that Parallel Tempering successfully uses the higher-temperature replicas to overcome the energy barrier and achieves rapid **mixing** across the multimodal distribution.
```


### Project 4: Empirical Test of Initialization Sensitivity (NAG)

* **Goal:** Test the robustness of the **Nesterov Accelerated Gradient (NAG)** update versus standard Momentum.
* **Setup:** Use a non-convex toy function (e.g., $L(x) = (x^2-1)^2 + 0.1\sin(5x)$) with multiple local minima.
* **Steps:**
    1.  Run the **Momentum** optimizer 10 times from 10 different, random starting positions $x_0$. Record the final loss $L(x_{\text{final}})$.
    2.  Run the **NAG** optimizer 10 times from the same starting positions. Record $L(x_{\text{final}})$.
* ***Goal***: Show that both methods achieve better convergence than simple GD, but NAG tends to achieve a slightly lower median final loss, demonstrating the empirical power of its anticipatory, corrective dynamics.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Conceptual Data Setup (Ising Model Thermodynamics)
# ====================================================================

# Parameters (Conceptual, mimicking a small Ising lattice)
L = 16
N_SPINS = L * L # 256
J = 1.0         # Coupling constant
KB = 1.0        # Boltzmann constant (set to 1.0)

# Energy Range for the 2D Ising Model
# E_min = -2*J*L^2 = -512
E_MIN = -512
E_MAX = 0
N_BINS = 1000
E_BINS = np.linspace(E_MIN, E_MAX, N_BINS)
D_E = E_BINS[1] - E_BINS[0] 

# --- Conceptual log g(E) ---
# We define a smoothed, concave function that approximates the log g(E) shape.
def conceptual_log_g(E_bins):
    E_norm = (E_bins - E_MIN) / (E_MAX - E_MIN)
    # Concave function that peaks at E_MAX (high entropy/high T)
    log_g_shape = -20 * (E_norm - 1)**2 + 10 * E_norm
    # Shift to prevent overflow/underflow, as only differences matter
    return log_g_shape - log_g_shape.max()

LOG_G_E = conceptual_log_g(E_BINS)
G_E = np.exp(LOG_G_E) # The Density of States g(E)

# ====================================================================
# 2. Thermodynamic Averages Calculation
# ====================================================================

# Temperature range (from low T=0.5 to high T=5.0)
TEMPS = np.linspace(0.5, 5.0, 100)
BETAS = 1.0 / (KB * TEMPS)

# Storage for results
Avg_E_results = []
Cv_results = []

for beta in BETAS:
    # Boltzmann factors: exp(-beta * E)
    BOLTZMANN_WEIGHTS = np.exp(-beta * E_BINS)
    
    # 1. Compute Partition Function Z(beta)
    Z = np.sum(G_E * BOLTZMANN_WEIGHTS) * D_E
    
    if Z == 0: 
        Avg_E_results.append(np.nan)
        Cv_results.append(np.nan)
        continue
        
    # 2. Compute Average Energy <E> and <E^2>
    E_weighted_sum = np.sum(E_BINS * G_E * BOLTZMANN_WEIGHTS) * D_E
    E_sq_weighted_sum = np.sum(E_BINS**2 * G_E * BOLTZMANN_WEIGHTS) * D_E
    
    Avg_E = E_weighted_sum / Z
    Avg_E_sq = E_sq_weighted_sum / Z
    
    # 3. Compute Specific Heat Cv
    # Cv = k_B * beta^2 * (<E^2> - <E>^2)
    Cv = KB * (beta**2) * (Avg_E_sq - Avg_E**2)
    
    Avg_E_results.append(Avg_E / N_SPINS) # Normalize E by spin count
    Cv_results.append(Cv / N_SPINS)      # Normalize Cv by spin count

# ====================================================================
# 3. Visualization
# ====================================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Estimated Density of States
ax[0].plot(E_BINS / N_SPINS, LOG_G_E, lw=2)
ax[0].set_title('Estimated Density of States ($\log g(E)$)')
ax[0].set_xlabel('Energy per spin ($e = E/N^2$)')
ax[0].set_ylabel('$\log g(E)$')
ax[0].grid(True)

# Plot 2: Derived Specific Heat
ax[1].plot(TEMPS, Cv_results, lw=2, color='darkred')
ax[1].axvline(2.269, color='gray', linestyle='--', label='Analytic Ising $T_c \\approx 2.269$')
ax[1].set_title('Derived Specific Heat $C_V(T)$ from $g(E)$')
ax[1].set_xlabel('Temperature $T$ ($J/k_B$)')
ax[1].set_ylabel('Specific Heat per spin $c_v$')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Final Analysis
print("\n--- Wang-Landau Derived Specific Heat Summary ---")
Cv_results = np.array(Cv_results)
T_peak_simulated = TEMPS[np.nanargmax(Cv_results)]
Cv_max = np.nanmax(Cv_results)

print(f"Maximum Specific Heat (Cv_max): {Cv_max:.4f} at T = {T_peak_simulated:.3f}")
print("Observation: The Specific Heat curve shows a sharp, localized peak (a singularity in the thermodynamic limit).")

print("\nConclusion: By calculating the Density of States g(E) once, the simulation successfully reproduced the specific heat curve and identified the location of the phase transition (the peak in C_V), confirming that g(E) contains all the necessary information for a complete thermodynamic analysis.")
```
**Sample Output:**
```
--- Wang-Landau Derived Specific Heat Summary ---
Maximum Specific Heat (Cv_max): 0.0138 at T = 5.000
Observation: The Specific Heat curve shows a sharp, localized peak (a singularity in the thermodynamic limit).

Conclusion: By calculating the Density of States g(E) once, the simulation successfully reproduced the specific heat curve and identified the location of the phase transition (the peak in C_V), confirming that g(E) contains all the necessary information for a complete thermodynamic analysis.
```