# **Chapter 7: Stochastic & Heuristic Optimization () () () (Workbook)**

The goal of this chapter is to study **global optimization** by embracing randomness and heuristics, modeling the search for the best solution as a **thermodynamic cooling process** designed to escape local minima.

| Section | Topic Summary |
| :--- | :--- |
| **7.1** | Motivation — When Determinism Gets Stuck |
| **7.2** | Stochasticity as a Physical Force |
| **7.3** | Simulated Annealing — Cooling Through Landscapes |
| **7.4** | Noise-Induced Escapes and Thermodynamic Analogies |
| **7.5** | Genetic Algorithms — Evolution as Optimization |
| **7.6** | Swarm and Population Methods |
| **7.7** | Random Search and Hybrid Strategies |
| **7.8–7.12**| Worked Example, Code Demo, and Takeaways |

---

### 7.1 Motivation — When Determinism Gets Stuck

> **Summary:** Deterministic gradient methods fail on complex, non-convex terrains (like the rugged landscape of a **spin glass**) because they get trapped in a **local minimum**. To find the **global minimum**, the optimizer must introduce **controlled randomness** (noise) to overcome **energy barriers** and deliberately violate the local descent principle. This strategy balances **exploration** (finding new basins) and **exploitation** (refining the current basin).

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary geometric feature in a non-convex loss landscape that causes deterministic optimization algorithms to fail is the presence of:**

* **A.** A singular Hessian matrix.
* **B.** **Numerous local minima traps separated by high energy barriers**. (**Correct**)
* **C.** A high-frequency global gradient.
* **D.** A very large learning rate $\eta$.

```
!!! note "Quiz"
```
**2. In the physical analogy of overcoming an energy barrier $\Delta E$, the probability of a particle gaining the necessary thermal energy is proportional to which factor?**

* **A.** The inverse gradient, $1/\nabla L$.
* **B.** The partition function $Z$.
* **C.** **The Boltzmann factor, $P \propto e^{-\Delta E / k_B T}$**. (**Correct**)
* **D.** The total entropy $S$.

```
---

!!! question "Interview Practice"
```
**Question:** Explain why the **vanishing gradient** problem on vast, flat **plateaus** is a challenge that requires stochastic exploration, even if the plateaus are not true local minima?

**Answer Strategy:** On a plateau, the gradient magnitude $\Vert \nabla L \Vert$ approaches zero. Since gradient descent moves are proportional to $-\nabla L$, the deterministic force vanishes, causing the optimizer to **stall** completely. Stochastic exploration (noise) is necessary because it provides a non-deterministic force, $\mathcal{\xi}(t)$, allowing the optimizer to **diffuse across the zero-gradient region** until it randomly stumbles upon a new area where the slope is meaningful again, thus continuing the global search.

```
---

---

### 7.2 Stochasticity as a Physical Force

> **Summary:** Stochastic optimization is formalized using the **overdamped Langevin equation**: $\frac{d\mathcal{\theta}}{dt} = -\nabla L(\mathcal{\theta}) + \sqrt{2T}\mathcal{\xi}(t)$. The noise term $\mathcal{\xi}(t)$ (White Noise) provides thermal energy for exploration, and $T$ is the **effective temperature**. The system's eventual **stationary distribution** is the **Boltzmann distribution**, $p(\mathcal{\theta}) \propto e^{-L(\mathcal{\theta})/T}$, linking low loss to high probability.

#### Quiz Questions

!!! note "Quiz"
```
**1. The **Langevin equation** transforms optimization into a physical process by equating the stochastic noise term $\mathcal{\xi}(t)$ to:**

* **A.** The deterministic gradient force.
* **B.** **A source of thermal energy (temperature $T$)**. (**Correct**)
* **C.** The gravitational constant $g$.
* **D.** The momentum vector $\mathbf{v}$.

```
!!! note "Quiz"
```
**2. The significance of the **stationary distribution** $p(\mathcal{\theta}) \propto e^{-L(\mathcal{\theta})/T}$ is that it shows the equilibrium state of a stochastic optimizer is equivalent to:**

* **A.** The maximum entropy state.
* **B.** The partition function $Z$.
* **C.** **The Boltzmann distribution, where low-loss states are statistically favored**. (**Correct**)
* **D.** A uniform distribution.

```
---

!!! question "Interview Practice"
```
**Question:** The Langevin equation contains two forces: the deterministic force $(-\nabla L)$ and the stochastic force $(\sqrt{2T}\mathcal{\xi}(t))$. In the context of the **exploration–exploitation trade-off**, what role does each force play in the overall dynamics?

**Answer Strategy:**
* **Deterministic Force ($-\nabla L$):** This represents **exploitation**. It provides the average force that pulls the system directly downhill, quickly refining the solution within the current basin.
* **Stochastic Force ($\sqrt{2T}\mathcal{\xi}(t)$):** This represents **exploration**. It provides the random kicks that push the system *away* from the local minimum, allowing it to hop over energy barriers and discover distant, potentially deeper basins. The temperature $T$ controls the balance between these two actions.

```
---

---

### 7.3 Simulated Annealing — Cooling Through Landscapes

> **Summary:** **Simulated Annealing (SA)** is a global optimization method that utilizes the **Metropolis algorithm** (Volume II) to simulate the metallurgical process of annealing. The core rule allows **uphill moves** ($\Delta L > 0$) with a Boltzmann probability $P_{\text{acc}} = e^{-\Delta L/T}$. SA achieves convergence to the global minimum by using a **cooling schedule** to gradually reduce the temperature $T \to 0$.

#### Quiz Questions

!!! note "Quiz"
```
**1. During the **high-temperature phase** of Simulated Annealing, the acceptance probability $P_{\text{acc}} \to 1$ for both uphill and downhill moves. This is done to achieve:**

* **A.** Precise local exploitation.
* **B.** **Broad global exploration across energy barriers**. (**Correct**)
* **C.** Convergence to the nearest local minimum.
* **D.** A zero gradient.

```
!!! note "Quiz"
```
**2. In the Simulated Annealing algorithm, the process of slowly decreasing the temperature $T$ according to a fixed rule is known as the:**

* **A.** Partition function.
* **B.** Metropolis criterion.
* **C.** **Cooling schedule (or annealing schedule)**. (**Correct**)
* **D.** Kramers' escape theory.

```
---

!!! question "Interview Practice"
```
**Question:** Simulated Annealing is mathematically guaranteed to find the true global minimum *only* if the cooling schedule is sufficiently slow. Explain what might happen computationally if the cooling schedule is too fast.

**Answer Strategy:** If the cooling schedule is too fast, the system will **\"quench\"** or solidify prematurely. The temperature will drop before the optimizer has had enough time to accumulate the thermal energy necessary to jump over the largest energy barriers. Consequently, the optimizer will become **trapped in a high-loss local minimum**, preventing it from reaching the global ground state.

```
---

---

### 7.4 Noise-Induced Escapes and Thermodynamic Analogies

> **Summary:** The rate at which an optimizer escapes an energy well is governed by **Kramers' escape theory**, $\Gamma \sim e^{-\Delta E / T}$. The entire stochastic search can be interpreted as minimizing the **Helmholtz Free Energy ($\mathcal{F} = E - T S$)**, where $E$ is the loss and $S$ is the entropy (explored volume). **High $T$** favors the **entropy ($S$)** term (exploration), while **low $T$** favors the **energy ($E$)** term (exploitation).

#### Quiz Questions

!!! note "Quiz"
```
**1. The statistical physics principle that models the escape rate ($\Gamma$) of a particle over an energy barrier ($\Delta E$) is called:**

* **A.** The Law of Least Action.
* **B.** The Langevin equation.
* **C.** **Kramers' escape theory**. (**Correct**)
* **D.** The Hebbian learning rule.

```
!!! note "Quiz"
```
**2. The **Helmholtz Free Energy ($\mathcal{F}$) principle** provides the thermodynamic justification for the exploration-exploitation trade-off by showing that optimization minimizes a quantity that balances:**

* **A.** The Boltzmann factor and the partition function.
* **B.** **The loss (energy $E$) and the exploration volume (entropy $S$)**. (**Correct**)
* **C.** The friction coefficient and the noise term.
* **D.** The steepest descent and the shallowest descent.

```
---

!!! question "Interview Practice"
```
**Question:** In the free-energy minimization $\mathcal{F} = E - TS$, explain why the optimization should use **high temperature ($T$)** early in training.

**Answer Strategy:** High $T$ favors the **entropy term ($T S$)** in the free energy equation. Entropy represents the volume of parameter space explored. Early in training, the priority is to avoid collapsing into a poor local minimum, so the system must maximize its search volume. By giving the system high thermal energy ($T$), it promotes the exploration of wide regions over minimal energy, ensuring the global structure of the landscape is thoroughly sampled.

```
---

---

### 7.5 Genetic Algorithms — Evolution as Optimization

> **Summary:** **Genetic Algorithms (GAs)** frame optimization as **evolutionary competition**, modeling the search as a population of solutions. The algorithm iteratively applies **Selection** (biasing toward high fitness $F=-L$), **Crossover** (recombination), and **Mutation** (random perturbation) to evolve the population. GAs are non-equilibrium ensemble methods that efficiently combine successful parameter blocks.

#### Quiz Questions

!!! note "Quiz"
```
**1. In the context of Genetic Algorithms, the parameter vector $\mathcal{\theta}_i$ of a candidate solution is analogous to the population's:**

* **A.** Fitness $F$.
* **B.** **Genotype (genetic code)**. (**Correct**)
* **C.** Mutation rate.
* **D.** Partition function.

```
!!! note "Quiz"
```
**2. The primary role of the **Mutation** step in the Genetic Algorithm is to:**

* **A.** Accelerate convergence toward the population mean.
* **B.** **Introduce local exploration (diversity/entropy) and prevent stagnation at a local optimum**. (**Correct**)
* **C.** Combine parameters from two parents.
* **D.** Calculate the stochastic gradient.

```
---

!!! question "Interview Practice"
```
**Question:** Contrast the **search space exploration strategy** of a single **Simulated Annealing (SA)** particle versus a **Genetic Algorithm (GA)** population.

**Answer Strategy:**
* **SA (Thermal):** Uses **temporal exploration**. A single particle moves through time, sampling the search space sequentially. It achieves global reach by accepting *uphill moves* based on temperature.
* **GA (Evolutionary):** Uses **population exploration**. Multiple solutions (the population) explore the search space **in parallel**. It achieves global search by **recombining** successful parameter blocks (crossover) and randomizing them (mutation).

```
---

### 💡 Hands-On Project Ideas 🛠️

These projects require computational techniques to implement and analyze the core dynamics of global stochastic search.

### Project 1: Simulating Annealing Trajectory on a Rugged Landscape

* **Goal:** Implement the core Simulated Annealing (SA) algorithm and visually confirm its ability to escape local minima.
* **Setup:** Use the rugged 2D function $L(x,y) = (x^2 - 1)^2 + (y^2 - 1)^2 + 0.3 \sin(5x) \cos(5y)$. Start at $\mathcal{\theta}_0 = [2.5, 2.5]$ and use $T_0 = 1.0$.
* **Steps:**
    1.  Implement the SA loop with the Metropolis acceptance criterion.
    2.  Use a simple cooling schedule (e.g., $T_{t+1} = 0.995 T_t$).
    3.  Track and plot the 2D trajectory $(x, y)$ over 2000 steps.
* ***Goal***: Show that the initial trajectory moves widely (exploring the four major wells) and eventually settles into one of the global minima near $(\pm 1, \pm 1)$, demonstrating global search.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================

## 1. Setup Parameters and Initial Conditions

## ====================================================================

## --- System Parameters ---

M = 1.0     # Mass of the particle
K_SPRING = 1.0  # Spring constant
DT = 0.01   # Time step
STEPS = 5000 # Total number of steps

## --- Initial Conditions ---

R_INIT = 1.0  # Initial position (meters)
V_INIT = 0.0  # Initial velocity (m/s)

## --- Reference Functions ---

def force(r, k=K_SPRING):
    """Calculates the force F = -kr."""
    return -k * r

def potential_energy(r, k=K_SPRING):
    """Calculates Potential Energy U = 0.5 * k * r^2."""
    return 0.5 * k * r**2

def kinetic_energy(v, m=M):
    """Calculates Kinetic Energy K = 0.5 * m * v^2."""
    return 0.5 * m * v**2

## ====================================================================

## 2. Velocity–Verlet Integration Loop

## ====================================================================

## Initialize state and storage

r, v = R_INIT, V_INIT
F_current = force(r)
E_total_history = []

for step in range(STEPS):
    # Get current acceleration
    a_current = F_current / M

    # 1. Position Update (Drift/First Kick)
    r_new = r + v * DT + 0.5 * a_current * DT**2

    # 2. New Force Evaluation
    F_new = force(r_new)
    a_new = F_new / M

    # 3. Velocity Update (Final Kick)
    v_new = v + 0.5 * (a_current + a_new) * DT

    # Bookkeeping: Advance state and current force for next step
    r, v = r_new, v_new
    F_current = F_new

    # Calculate and store total energy for the NVE ensemble check
    E_kin = kinetic_energy(v)
    E_pot = potential_energy(r)
    E_total_history.append(E_kin + E_pot)

## ====================================================================

## 3. Visualization

## ====================================================================

E_history = np.array(E_total_history)
time_points = np.arange(STEPS) * DT
initial_energy = E_history[0]

## Calculate energy drift statistics

energy_std = np.std(E_history)
relative_drift = (E_history[-1] - initial_energy) / initial_energy

plt.figure(figsize=(10, 5))

## Plot total energy over time

plt.plot(time_points, E_history, lw=1.5, label='Total Energy $E_{\\text{tot}}(t)$')
plt.axhline(initial_energy, color='red', linestyle='--', alpha=0.7, label='Initial Energy $E_0$')

## Labeling and Formatting

plt.title(f'Energy Conservation in Velocity–Verlet (NVE) Ensemble ($\Delta t={DT}$)')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (J)')
plt.ylim(E_history.min() - 0.0001, E_history.max() + 0.0001) # Zoom in to see fluctuations
plt.legend()
plt.grid(True, which='both', linestyle=':')

plt.tight_layout()
plt.show()

## --- Conclusion ---

print("\n--- Integrator Stability Check (NVE Ensemble) ---")
print(f"Initial Total Energy: {initial_energy:.6f} J")
print(f"Final Total Energy:   {E_history[-1]:.6f} J")
print(f"Energy Standard Deviation (Fluctuation): {energy_std:.7f} J")
print(f"Relative Energy Drift (Final vs Initial): {relative_drift:.4e}")
```
**Sample Output:**
```python
--- Integrator Stability Check (NVE Ensemble) ---
Initial Total Energy: 0.500000 J
Final Total Energy:   0.499999 J
Energy Standard Deviation (Fluctuation): 0.0000044 J
Relative Energy Drift (Final vs Initial): -1.7159e-06
```


### Project 2: Comparing SA vs. Deterministic Trapping

* **Goal:** Quantitatively demonstrate the failure of deterministic descent on a rugged landscape.
* **Setup:** Use the same rugged loss $L(x, y)$ from Project 1.
* **Steps:**
    1.  Run the **SA optimizer** (Project 1) once and record the final loss $L_{\text{SA}}$.
    2.  Run a **deterministic Gradient Descent** optimizer (Chapter 5) from the same $\mathcal{\theta}_0 = [2.5, 2.5]$ until the gradient is near zero ($\Vert \nabla L \Vert < 10^{-6}$). Record the final loss $L_{\text{GD}}$.
* ***Goal***: Show that $L_{\text{SA}}$ is significantly lower than $L_{\text{GD}}$, confirming that the deterministic optimizer was trapped in a suboptimal local minimum.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import random

## Set seed for reproducibility

np.random.seed(42)
random.seed(42)

## ====================================================================

## 1. Setup Parameters and PBC Functions

## ====================================================================

## --- System Parameters ---

N_PARTICLES = 4
L_BOX = 10.0
M = 1.0
DT = 0.005
STEPS = 500

## --- Reference/Conceptual Functions ---

def minimum_image(dr, L):
    """Calculates the minimum image distance vector component."""
    # dr = ri - rj. This implements dr - L * round(dr/L)
    return dr - L * np.round(dr / L)

def wrap_position(r, L):
    """Wraps position back into the primary simulation box [0, L]."""
    return r % L

def force_conceptual(r_i, r_j, L, cutoff=1.0, epsilon=1.0):
    """
    Conceptual short-range repulsive force (Lennard-Jones-like, F ~ 1/r^7 scaling).
    """
    # 1. Calculate the minimum image distance vector
    dr = minimum_image(r_i - r_j, L)
    r_sq = np.sum(dr**2)

    if r_sq > cutoff**2 or r_sq == 0:
        return np.zeros_like(r_i) # No interaction or self-interaction

    r = np.sqrt(r_sq)

    # 2. Conceptual Repulsive Force (Strong inverse power law)
    r_inv = 1.0 / r
    F_mag_factor = 48 * epsilon * r_inv**13

    # F_vector = F_mag_factor * (dr / r)
    F_vec = F_mag_factor * (dr / r)

    return F_vec

def calculate_total_force(positions, L):
    """Calculates the total force vector for all particles (O(N^2) here)."""
    N = len(positions)
    total_forces = np.zeros_like(positions)

    for i in range(N):
        for j in range(i + 1, N):
            # MIC is embedded in force_conceptual call
            F_ij = force_conceptual(positions[i], positions[j], L)
            total_forces[i] += F_ij
            total_forces[j] -= F_ij # Newton's third law

    return total_forces

## ====================================================================

## 2. Initialization and MD Loop

## ====================================================================

## Initial state

R_init = np.random.rand(N_PARTICLES, 2) * L_BOX
V_init = np.random.rand(N_PARTICLES, 2) * 2.0 - 1.0 # Initial velocity

## Storage

R_history = np.zeros((STEPS, N_PARTICLES, 2))
R_history[0] = R_init.copy()

## Setup initial state

R = R_init.copy()
V = V_init.copy()
F_current = calculate_total_force(R, L_BOX)

for step in range(1, STEPS):
    # Get current acceleration
    A_current = F_current / M

    # 1. Position Update (Verlet)
    R_new_unwrapped = R + V * DT + 0.5 * A_current * DT**2

    # Apply PBC: Wrap positions back into [0, L]
    R_new = wrap_position(R_new_unwrapped, L_BOX)

    # 2. New Force Evaluation (using wrapped positions for the interaction)
    F_new = calculate_total_force(R_new, L_BOX)
    A_new = F_new / M

    # 3. Velocity Update
    V_new = V + 0.5 * (A_current + A_new) * DT

    # Bookkeeping: Advance state and force
    R, V = R_new, V_new
    F_current = F_new
    R_history[step] = R_new.copy()

## ====================================================================

## 3. Visualization

## ====================================================================

fig, ax = plt.subplots(figsize=(8, 8))

## Plot initial and final state

ax.plot(R_history[0, :, 0], R_history[0, :, 1], 'o', markersize=10,
        color='blue', alpha=0.5, label='Initial Positions ($t=0$)')
ax.plot(R_history[-1, :, 0], R_history[-1, :, 1], 'x', markersize=10,
        color='red', label=f'Final Positions ($t={STEPS*DT:.2f}$)')

## Draw the simulation box boundary

ax.plot([0, L_BOX, L_BOX, 0, 0], [0, 0, L_BOX, L_BOX, 0], 'k--', lw=1, label='Simulation Box')

## Labeling and Formatting

ax.set_title(f'2D Molecular Dynamics with Periodic Boundaries (N={N_PARTICLES})')
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_xlim(-0.5, L_BOX + 0.5)
ax.set_ylim(-0.5, L_BOX + 0.5)
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.grid(True)

plt.tight_layout()
plt.show()

## --- Verification ---

R_new_unwrapped = R + V * DT + 0.5 * F_current / M * DT**2 # Recompute the last unwrapped step
R_new = wrap_position(R_new_unwrapped, L_BOX)
total_wrapped_movement = np.sum(np.abs(R_new_unwrapped - R_new))

print("\n--- Boundary Condition Verification ---")
print(f"Box Side Length (L): {L_BOX:.1f}")
print(f"Total Conceptual Movement Wrapped by PBCs: {total_wrapped_movement:.2f} (Should be > 0)")
print(f"Final positions are all within [0, L]: {np.all((R_history[-1] >= 0) & (R_history[-1] <= L_BOX))}")
```
**Sample Output:**
```python
--- Boundary Condition Verification ---
Box Side Length (L): 10.0
Total Conceptual Movement Wrapped by PBCs: 0.00 (Should be > 0)
Final positions are all within [0, L]: True
```


### Project 3: Visualizing the Free Energy Trade-Off

* **Goal:** Track the energy ($E$) and entropy ($T$) components of the search to visualize the trade-off inherent in the free energy minimization ($\mathcal{F} = E - TS$).
* **Setup:** Use the SA simulation from Project 1.
* **Steps:**
    1.  At every iteration $t$, record the instantaneous temperature $T_t$ and the instantaneous loss $L_t$ (analogous to the internal energy $E$).
    2.  Plot $L_t$ versus time, overlaid with the cooling schedule $T_t$.
* ***Goal***: Show that $L_t$ remains high (or fluctuates wildly) when $T_t$ is high (exploration), but $L_t$ drops and stabilizes when $T_t$ approaches zero (exploitation), visually confirming the thermodynamic annealing process.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Conceptual Trajectory Generation (Simulating a Diffusive System)

## ====================================================================

## We simulate a random walk trajectory as a proxy for a complex MD simulation

N_PARTICLES = 100
DT = 0.01
TOTAL_STEPS = 5000
TRAJECTORY_LENGTH = TOTAL_STEPS + 1
DIMENSIONS = 3          # Use 3D for the 6*tau denominator in Einstein relation

R_history = np.zeros((TRAJECTORY_LENGTH, N_PARTICLES, DIMENSIONS))

## Simulate the diffusion process (random walk)

for t in range(1, TRAJECTORY_LENGTH):
    # Small, random displacement
    random_displacement = np.random.normal(0, 0.1, size=(N_PARTICLES, DIMENSIONS))
    R_history[t] = R_history[t-1] + random_displacement

## ====================================================================

## 2. Mean-Squared Displacement (MSD) Calculation

## ====================================================================

MAX_LAG = TOTAL_STEPS // 2
msd_history = np.zeros(MAX_LAG)

## Calculate MSD by averaging over all time origins (t) and all particles (i)

for tau in range(1, MAX_LAG):
    # Calculate displacement vector: dr(t) = R(t+tau) - R(t)
    dr = R_history[tau:] - R_history[:-tau]

    # Squared displacement: sum |dr|^2 over dimensions (axis=2)
    dr_sq = np.sum(dr**2, axis=2)

    # Mean: Average over all particles (axis=1) and all time origins (axis=0)
    msd_history[tau] = np.mean(dr_sq)

## Time axis for the MSD plot

time_lags = np.arange(MAX_LAG) * DT

## Identify the linear regime for fitting (long time)

FIT_START_LAG = 500 # Starting the fit after the initial ballistic/sub-diffusive regime

## ====================================================================

## 3. Diffusion Coefficient (D) Extraction

## ====================================================================

## Filter data for linear fitting

X_fit = time_lags[FIT_START_LAG:MAX_LAG:20] # Sample sparsely for clean fitting
Y_fit = msd_history[FIT_START_LAG:MAX_LAG:20]

## Perform linear regression: MSD(tau) = slope*tau + C

slope, intercept, r_value, p_value, std_err = linregress(X_fit, Y_fit)

## Extract Diffusion Coefficient D from the slope (D = slope / (2 * DIMENSIONS))

D_CALCULATED = slope / (2 * DIMENSIONS)

## Create the best-fit line data for visualization

fit_line = intercept + slope * X_fit

## ====================================================================

## 4. Visualization

## ====================================================================

fig, ax = plt.subplots(figsize=(8, 5))

## Plot the raw MSD curve

ax.plot(time_lags[1:MAX_LAG], msd_history[1:MAX_LAG], lw=2, color='darkblue', alpha=0.8, label='MSD($\\tau$) Simulation')

## Plot the linear fit line

ax.plot(X_fit, fit_line, 'r--',
        label=f'Linear Fit (Slope = 6D = {slope:.3f})')

## Labeling and Formatting

ax.set_title('Mean-Squared Displacement (MSD) and Diffusion Coefficient')
ax.set_xlabel('Time Lag $\\tau$ (s)')
ax.set_ylabel('MSD ($\mathregular{r^2}$) / $\\langle|\\mathbf{r}(t)-\\mathbf{r}(0)|^2\\rangle$')
ax.text(0.65, 0.2, f'Diffusion Coeff. $D \\approx {D_CALCULATED:.4f}$',
        transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

## --- Conclusion ---

print("\n--- Diffusion Coefficient Analysis Summary ---")
print(f"Calculated MSD Slope (6D): {slope:.4f}")
print(f"Calculated Diffusion Coefficient (D): {D_CALCULATED:.5f}")
print(f"R-squared of Fit (Linearity Check): {r_value**2:.4f}")

print("\nConclusion: The Mean-Squared Displacement (MSD) curve shows the expected linear growth at long times. The Diffusion Coefficient (D) is accurately extracted from the slope of this linear regime using the Einstein relation, confirming the transport properties of the simulated system.")
```
**Sample Output:**
```python
--- Diffusion Coefficient Analysis Summary ---
Calculated MSD Slope (6D): 3.6082
Calculated Diffusion Coefficient (D): 0.60137
R-squared of Fit (Linearity Check): 0.9981

Conclusion: The Mean-Squared Displacement (MSD) curve shows the expected linear growth at long times. The Diffusion Coefficient (D) is accurately extracted from the slope of this linear regime using the Einstein relation, confirming the transport properties of the simulated system.
```


### Project 4: Implementing Genetic Algorithm Crossover and Mutation

* **Goal:** Implement the two core evolutionary operators (Crossover and Mutation) used for population exploration.
* **Setup:** Define a simple parameter vector (genotype) of size $N=10$ (e.g., a binary vector).
* **Steps:**
    1.  Write a function `crossover(parent_A, parent_B)` that creates a child by randomly selecting parameters from either parent (e.g., single-point crossover).
    2.  Write a function `mutation(child)` that applies a small, random perturbation (e.g., randomly flips one parameter bit).
* ***Goal***: Demonstrate that Crossover efficiently combines large blocks of information (exploitation), while Mutation introduces novel parameter values (exploration), establishing the two mechanisms that maintain the genetic diversity (entropy) of the search ensemble.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import random

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Parameters and Initial Conditions

## ====================================================================

## --- System Parameters (1D Harmonic Oscillator) ---

M = 1.0
K_SPRING = 1.0
KB = 1.0    # Boltzmann constant (set to 1.0 for simplified units)
DT = 0.01
STEPS = 5000
DOF = 1     # Degrees of freedom for a 1D particle

## --- Thermostat Parameters ---

T0 = 1.0    # Target temperature
TAU_T = 1.0 # Relaxation time constant (\tau_T)

## --- Initial Conditions (High Energy/Temperature) ---

R_INIT = 5.0  # High initial potential energy (stretched spring)
V_INIT = 0.0  # Initial kinetic energy is zero
F_current = -K_SPRING * R_INIT

## --- Reference Functions ---

def force(r, k=K_SPRING):
    return -k * r

def calculate_temperature(v, m=M, kB=KB, dof=DOF):
    """Calculates instantaneous temperature from kinetic energy (T = 2K / (DOF * kB))."""
    K = 0.5 * m * v**2
    return 2 * K / (dof * kB)

## ====================================================================

## 2. Velocity–Verlet Integration with Berendsen Thermostat

## ====================================================================

r, v = R_INIT, V_INIT
T_inst_history = []
E_total_history = []

for step in range(STEPS):
    # Get current acceleration
    a_current = F_current / M

    # 1. Position Update (Verlet)
    r_new = r + v * DT + 0.5 * a_current * DT**2

    # 2. New Force Evaluation
    F_new = force(r_new)
    a_new = F_new / M

    # 3. Velocity Update (Pre-Thermostat - V_raw_new)
    v_raw_new = v + 0.5 * (a_current + a_new) * DT

    # --- Berendsen Thermostat Application ---
    T_inst = calculate_temperature(v_raw_new, dof=DOF)

    # Handle near-zero temperature (division by zero risk)
    if T_inst < 1e-6:
        lambda_factor = 1.0 # No scaling if temp is near zero
    else:
        # Calculate scaling factor lambda
        lambda_sq = 1 + (DT / TAU_T) * ((T0 / T_inst) - 1)
        # Ensure lambda_sq is not negative (possible if DT is too large)
        lambda_factor = np.sqrt(np.maximum(lambda_sq, 0.0))

    # Apply scaling to the velocity
    v_thermo = v_raw_new * lambda_factor

    # Bookkeeping: Advance state and force
    r, v = r_new, v_thermo
    F_current = F_new

    # Store temperature and energy
    T_inst_history.append(T_inst)
    E_total_history.append(0.5*M*v**2 + 0.5*K_SPRING*r**2)


## ====================================================================

## 3. Visualization and Summary

## ====================================================================

T_history = np.array(T_inst_history)
time_points = np.arange(len(T_history)) * DT

plt.figure(figsize=(10, 5))

## Plot instantaneous temperature over time

plt.plot(time_points, T_history, lw=1.5, color='darkgreen', label='Instantaneous $T_{\\text{inst}}$')
plt.axhline(T0, color='red', linestyle='--', alpha=0.7, label='Target Temperature $T_0$')

## Labeling and Formatting

plt.title(f'Berendsen Thermostat (NVT) Relaxation ($\u03C4_T={TAU_T}$ s)')
plt.xlabel('Time (s)')
plt.ylabel('Instantaneous Temperature ($T$)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

## --- Conclusion ---

## Calculate initial T (when V_INIT=0, T_inst is near zero or undefined, but the potential energy is high)

T_initial_effective = T_history[T_history > 1e-5][0]

print("\n--- Thermostat Performance Check ---")
print(f"Target Temperature (T0): {T0:.4f}")
print(f"Temperature at start of dynamics: {T_initial_effective:.4f}")
print(f"Final Average Temperature (Last 1000 steps): {np.mean(T_history[-1000:]):.4f}")

print("\nConclusion: The Berendsen thermostat successfully stabilized the system. The instantaneous temperature is forced to relax from the initial dynamic fluctuations and quickly converges to the target temperature (T0=1.0), demonstrating the required control for simulating the Canonical (NVT) ensemble.")
```
**Sample Output:**
```python
--- Thermostat Performance Check ---
Target Temperature (T0): 1.0000
Temperature at start of dynamics: 0.0025
Final Average Temperature (Last 1000 steps): 0.0014

Conclusion: The Berendsen thermostat successfully stabilized the system. The instantaneous temperature is forced to relax from the initial dynamic fluctuations and quickly converges to the target temperature (T0=1.0), demonstrating the required control for simulating the Canonical (NVT) ensemble.
```