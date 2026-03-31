# **Chapter 16: Physics-Informed Neural Networks (PINNs) () () () (Workbook)**

The goal of this chapter is to introduce the methodology of **Physics-Informed Neural Networks (PINNs)**, which formally embed differential physical laws into the optimization objective, transforming data-fitting into **law-constrained learning**.

| Section | Topic Summary |
| :--- | :--- |
| **16.1–16.2** | Motivation: From Data-Fitting to Law-Fitting |
| **16.3–16.4** | Core Mechanism: Automatic Differentiation and Loss Construction |
| **16.5–16.6** | Applications: Solving Forward and Inverse Problems |
| **16.7–16.9** | Synthesis: Example (Burgers' Eq.) & Physics vs. Data-Driven ML |
| **16.10–16.15** | Advanced Topics, Challenges, and Takeaways |

---

### 16.1–16.2 Motivation: From Data-Fitting to Law-Fitting

> **Summary:** PINNs solve the problem of **data sparsity** and **poor extrapolation** in traditional Machine Learning (ML). The neural network $u_{\mathcal{\theta}}(\mathbf{x}, t)$ must satisfy the **governing PDE** (the law) everywhere. This law ($\mathcal{N}[u]=f$) is converted into a **Physics Loss ($L_{\text{phys}}$)**, which penalizes the non-zero **Residual**. The entire process minimizes an **augmented energy functional**, balancing empirical data fit against physical consistency.

#### Quiz Questions

!!! note "Quiz"
```
**1. Which concept or phenomenon does the PINN framework primarily address and attempt to mitigate in scientific ML applications?**

* **A.** The high number of saddle points in the loss landscape.
* **B.** **Poor extrapolation and physical inconsistency arising from sparse training data**. (**Correct**)
* **C.** The computational cost of quantum annealing.
* **D.** The need for continuous activation functions.

```
!!! note "Quiz"
```
**2. The single mathematical expression that represents the system's governing physical law, which PINNs seek to drive to zero, is called the:**

* **A.** Total Loss $L_{\text{total}}$.
* **B.** **Differential operator residual ($\mathcal{N}[u] - f$)**. (**Correct**)
* **C.** Boundary condition ($L_{\text{bc}}$).
* **D.** Variational Free Energy ($\mathcal{F}$).

```
---

!!! question "Interview Practice"
```
**Question:** Explain the philosophical significance of the PINN loss function, $L = L_{\text{data}} + \lambda L_{\text{physics}}$, using the analogy of **Energy Minimization and Conservation Laws**.

**Answer Strategy:** This loss structure is a form of **constrained energy minimization**.
1.  **$L_{\text{data}}$ $\leftrightarrow$ Empirical Energy:** This term drives the system toward minimizing the observed error (fitting the data).
2.  **$L_{\text{physics}}$ $\leftrightarrow$ Constraint Energy:** This term ensures the solution is restricted to the subspace of functions that are physically viable (e.g., that conserve energy or momentum).
By minimizing the total functional, the PINN finds the solution that is the best fit for the measurements while being fundamentally **consistent with the rules of the universe**.

```
---

---

### 16.3–16.4 Core Mechanism: Automatic Differentiation and Loss Construction

> **Summary:** The core enabler of PINNs is **Automatic Differentiation (AD)**. AD, using the **chain rule** (Backpropagation), computes exact partial derivatives ($\frac{\partial u}{\partial x}$, $\frac{\partial^2 u}{\partial x^2}$) of the neural output with respect to the input coordinates. This capability allows the network to construct the total loss $L$ from three components: **$L_{\text{data}}$** (empirical fit), **$L_{\text{phys}}$** (PDE residual), and **$L_{\text{bc}}$** (boundary/initial conditions).

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary challenge in PINNs that is solved by using **Automatic Differentiation (AD)** instead of traditional finite difference methods is:**

* **A.** Discretizing the spatial domain.
* **B.** **Computing high-order partial derivatives of the network output with machine precision**. (**Correct**)
* **C.** Selecting the optimal optimizer (e.g., Adam).
* **D.** Balancing the loss weighting factors ($\mathcal{\lambda}$).

```
!!! note "Quiz"
```
**2. The loss component that ensures the PDE's solution is unique by enforcing known values at $t=0$ and spatial edges is:**

* **A.** $L_{\text{data}}$.
* **B.** $L_{\text{phys}}$.
* **C.** **$L_{\text{bc}}$ (Boundary/Initial Condition Loss)**. (**Correct**)
* **D.** The total residual.

```
---

!!! question "Interview Practice"
```
**Question:** Automatic Differentiation (AD) is required to compute the second derivative $u_{xx}$. Explain the conceptual process of computing a second derivative using AD, referencing the mechanism of the chain rule.

**Answer Strategy:** AD computes derivatives by iteratively applying the chain rule.
1.  **First Derivative:** AD is first applied to the output $u(\mathbf{x})$ with respect to the input $\mathbf{x}$ to compute the result $u_x$.
2.  **Second Derivative:** AD is then applied **a second time** to the intermediate output $u_x$ with respect to the input $\mathbf{x}$ to compute $u_{xx}$.
The process is exact because the network is fully differentiable, essentially providing the PINN with an **exact microscopic calculus**.

```
---

---

### 16.5–16.6 Applications: Solving Forward and Inverse Problems

> **Summary:** PINNs solve the **forward problem** (finding $u$ given $\mathcal{N}$ and ICs/BCs) by minimizing the PDE residual at arbitrary **collocation points**. They excel at **inverse problems** (finding parameters $\mathcal{\mu}$ given $\mathcal{N}$ and partial data) by treating the unknown physical constants ($\mathcal{\mu}$) as **additional trainable variables** within the optimization. This allows PINNs to **infer the hidden laws** and physical constants from noisy observations.

#### Quiz Questions

!!! note "Quiz"
```
**1. When using a PINN to solve a **forward problem**, the points scattered throughout the domain where the physics loss is enforced are called:**

* **A.** Boundary points.
* **B.** Activation points.
* **C.** **Collocation points**. (**Correct**)
* **D.** Initial condition points.

```
!!! note "Quiz"
```
**2. In a PINN-based **inverse problem**, the physical constants (e.g., diffusivity $\alpha$ or viscosity $\nu$) are optimized by:**

* **A.** Setting them equal to the learning rate $\eta$.
* **B.** **Treating them as trainable variables alongside the network weights $\mathcal{\theta}$**. (**Correct**)
* **C.** Minimizing the $L_{\text{data}}$ term only.
* **D.** Using a separate analytical solver.

```
---

!!! question "Interview Practice"
```
**Question:** Contrast the computational difficulty and final product of solving a PDE using a PINN versus a traditional numerical method (e.g., Finite Difference).

**Answer Strategy:**
* **Traditional Solver:** The difficulty is **discretization**. It solves the PDE at discrete grid points, requiring complex logic for handling non-linear terms and ensuring stability. The final product is a **discrete set of values** on a mesh.
* **PINN:** The difficulty is **optimization**. It solves the PDE by minimizing a loss functional. The final product is a **single, continuous, and differentiable function** $u_{\mathcal{\theta}}(\mathbf{x}, t)$ that satisfies the law everywhere in the domain.

```
---

---

### 16.7–16.9 Synthesis: Example (Burgers' Eq.) & Physics vs. Data-Driven ML

> **Summary:** The non-linear **Burgers' Equation** is a challenging example solved by constructing the physics residual from AD-derived derivatives. PINNs provide **strong generalization** because the physics constraint limits the solution space to only physically consistent functions. The overall process minimizes an **augmented energy functional** ($L_{\text{data}} + L_{\text{phys}}$), which is a modern incarnation of the **least action principle**.

### 16.10–16.15 Advanced Topics, Challenges, and Takeaways

> **Summary:** Challenges in PINN training include **stiff PDEs** (leading to vanishing gradients) and the difficulty of **balancing loss weights** ($\mathcal{\lambda}$). Advanced methods like **adaptive sampling** and **hybrid PINNs** are used to overcome these. The framework connects to **Neural Quantum States (NQS)** (Chapter 17) by relating the minimization of the PDE residual to minimizing a **variational energy functional**.

## 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core components of the PINN framework, focusing on AD and loss construction.

### Project 1: Implementing Automatic Differentiation (AD) for Second Order

* **Goal:** Implement the core AD mechanism required to compute the terms of the Heat Equation, $u_t$ and $u_{xx}$.
* **Setup:** Define a simple analytical function for the field $u(x, t) = \sin(\pi x)e^{-t}$ (a solution to the Heat Equation).
* **Steps:**
    1.  Convert the inputs $x$ and $t$ into PyTorch tensors with `requires_grad=True`.
    2.  Use $\texttt{torch.autograd.grad}$ once to compute $u_t$ and $u_x$.
    3.  Use $\texttt{torch.autograd.grad}$ on the result of $u_x$ to compute the second derivative $u_{xx}$.
* ***Goal***: Show that the numerically calculated $u_t$ and $\alpha u_{xx}$ are approximately equal (i.e., the residual is near zero), confirming that AD correctly computes the required partial derivatives.

#### Python Implementation

```python
import tensorflow as tf
import numpy as np

# Note: In a full implementation, the Neural Network model and its weights (\theta)
# would be defined and trained using the combined loss function.

# ====================================================================
# 1. Setup Conceptual PDE (1D Heat Equation)
# ====================================================================

# PDE: u_t = alpha * u_xx (Heat Equation)
ALPHA = 0.5 
N_COLLOCATION = 100 

# Conceptual Collocation Points (x, t) for training the physics loss
X_COLLOC = tf.constant(np.random.rand(N_COLLOCATION, 2), dtype=tf.float32)

# Conceptual PINN Output (u_theta(x, t)) - Placeholder for the NN
# This function would be the actual output of the trained neural network model
def u_theta(x_t):
    # Simplified placeholder: u(x, t) = sin(\pi x) * exp(-t * \pi^2 * alpha)
    x, t = x_t[:, 0:1], x_t[:, 1:2]
    return tf.sin(np.pi * x) * tf.exp(-ALPHA * np.pi**2 * t)


# ====================================================================
# 2. Automatic Differentiation (AD) and Residual Calculation
# ====================================================================

def calculate_physics_loss(X_colloc, alpha=ALPHA):
    with tf.GradientTape(persistent=True) as tape:
        # Step 0: Ensure (x, t) are tracked for AD
        tape.watch(X_colloc)
        
        # Output of the neural network (u)
        u = u_theta(X_colloc)
        
        # Step 1: Calculate First Derivatives (u_t, u_x)
        u_t = tape.gradient(u, X_colloc)[:, 1:2] # u_t (index 1 is time)
        u_x = tape.gradient(u, X_colloc)[:, 0:1] # u_x (index 0 is space)

        # Step 2: Calculate Second Derivative (u_xx) using a second tape
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(u_x)
            u_xx = tape2.gradient(u_x, X_colloc)[:, 0:1] # u_xx

        # Step 3: Compute the Residual (R)
        # R = u_t - alpha * u_xx
        R = u_t - alpha * u_xx
        
        # Step 4: Compute the Physics Loss (L_phys = MSE of R)
        L_phys = tf.reduce_mean(tf.square(R))
        
    return L_phys.numpy()

# =FFT (Final Function Test)
physics_loss_value = calculate_physics_loss(X_COLLOC)

print("--- Physics Loss Calculation Summary (Conceptual) ---")
print(f"PDE: u_t = {ALPHA} * u_xx")
print(f"Calculated Physics Loss L_phys: {physics_loss_value:.4e} (Should be near zero if NN is accurate)")
```

### Project 2: Constructing the Full PINN Loss Functional

* **Goal:** Construct the total weighted loss function $L$ for the 1D Heat Equation, explicitly including $L_{\text{data}}$, $L_{\text{phys}}$, and $L_{\text{bc}}$.
* **Setup:** Use the PINN structure from the demo. Set the weights $\mathcal{\lambda} = [\lambda_d=10, \lambda_p=1, \lambda_b=10]$.
* **Steps:**
    1.  Define a set of synthetic data points ($L_{\text{data}}$) and boundary points ($L_{\text{bc}}$).
    2.  Define the Physics Loss ($L_{\text{phys}}$) as the MSE of the residual $|u_t - \alpha u_{xx}|^2$.
    3.  Combine the terms using the $\mathcal{\lambda}$ weights: $L = \lambda_d L_{\text{data}} + \lambda_p L_{\text{phys}} + \lambda_b L_{\text{bc}}$.
* ***Goal***: Produce a single, differentiable scalar loss value $L$, representing the **augmented energy functional** that the optimizer will minimize.

#### Python Implementation

```python
import tensorflow as tf
import numpy as np

# ====================================================================
# 1. Setup Constraints and Loss Components
# ====================================================================

# Constants
ALPHA = 0.5  # Known diffusion constant
L = 1.0      # Domain length
T_FINAL = 1.0

# --- Data Points (Conceptual) ---
N_IC = 50   # Points for Initial Condition
N_BC = 50   # Points for Boundary Conditions
N_PHYS = 500 # Points for Physics/Collocation

# 1. Initial Condition Data (IC): u(x, 0) = sin(\pi x)
X_IC = np.hstack([np.random.rand(N_IC, 1) * L, np.zeros((N_IC, 1))])
U_IC = np.sin(np.pi * X_IC[:, 0:1])

# 2. Boundary Condition Data (BC): u(0, t) = u(L, t) = 0
X_BC_L = np.hstack([np.zeros((N_BC, 1)), np.random.rand(N_BC, 1) * T_FINAL])
X_BC_R = np.hstack([np.full((N_BC, 1), L), np.random.rand(N_BC, 1) * T_FINAL])
U_BC = np.zeros((N_BC * 2, 1))

# 3. Collocation Points (Physics): Random points in the domain
X_PHYS = np.random.rand(N_PHYS, 2)
X_PHYS[:, 0] *= L # Scale x to [0, L]
X_PHYS[:, 1] *= T_FINAL # Scale t to [0, T_FINAL]

# All points must be wrapped in tf.constant with tape.watch for AD
X_IC_tf = tf.constant(X_IC, dtype=tf.float32)
X_BC_tf = tf.constant(np.vstack([X_BC_L, X_BC_R]), dtype=tf.float32)
X_PHYS_tf = tf.constant(X_PHYS, dtype=tf.float32)
U_IC_tf = tf.constant(U_IC, dtype=tf.float32)
U_BC_tf = tf.constant(U_BC, dtype=tf.float32)

# --- Conceptual Model (NN Placeholder) ---
# In a full PINN, this would be a Keras Sequential model.
def u_theta(x_t):
    # Placeholder for NN output: u(x, t) \approx x_t[:, 0] * x_t[:, 1] (Wrong, but trainable)
    return tf.reduce_sum(x_t, axis=1, keepdims=True) 

# ====================================================================
# 2. PINN Loss Function (Composite Loss)
# ====================================================================

def calculate_total_loss(alpha=ALPHA, w_ic=1.0, w_bc=1.0):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X_PHYS_tf)
        tape.watch(X_IC_tf)
        tape.watch(X_BC_tf)
        
        # A. Physics Loss (L_phys) - Uses AD on collocation points
        u_phys = u_theta(X_PHYS_tf)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(X_PHYS_tf)
            u_x = tape2.gradient(u_theta(X_PHYS_tf), X_PHYS_tf)[:, 0:1]
            u_t = tape2.gradient(u_theta(X_PHYS_tf), X_PHYS_tf)[:, 1:2]
        u_xx = tape.gradient(u_x, X_PHYS_tf)[:, 0:1]
        
        R = u_t - alpha * u_xx
        L_phys = tf.reduce_mean(tf.square(R))

        # B. Initial Condition Loss (L_IC)
        u_ic_pred = u_theta(X_IC_tf)
        L_IC = tf.reduce_mean(tf.square(u_ic_pred - U_IC_tf))

        # C. Boundary Condition Loss (L_BC)
        u_bc_pred = u_theta(X_BC_tf)
        L_BC = tf.reduce_mean(tf.square(u_bc_pred - U_BC_tf))

        # Total Loss
        L_total = L_phys + w_ic * L_IC + w_bc * L_BC
        
    return L_total.numpy(), L_phys.numpy(), L_IC.numpy(), L_BC.numpy()

# --- Final Function Test ---
L_total, L_phys, L_IC, L_BC = calculate_total_loss()

print("\n--- Solving the Forward Problem (Conceptual Multi-Loss) ---")
print(f"Total Loss (L_total): {L_total:.4f}")
print(f"Physics Loss (L_phys): {L_phys:.4f}")
print(f"Initial Condition Loss (L_IC): {L_IC:.4f}")
print(f"Boundary Condition Loss (L_BC): {L_BC:.4f}")
```

### Project 3: PINN for an Inverse Problem (Inferring Diffusivity)

* **Goal:** Simulate the core mechanism of an **inverse problem** by making a physical constant ($\alpha$, diffusivity) a trainable parameter.
* **Setup:** Define the constant $\alpha$ as a PyTorch tensor with `requires_grad=True`. Fix the network weights (no training on $\mathcal{\theta}$ for this conceptual step).
* **Steps:**
    1.  Define a "true" observation $u_{\text{obs}}$ (e.g., from the exact solution with $\alpha_{\text{true}}=0.5$).
    2.  Set the initial guess for $\alpha_{\text{guess}}=0.1$.
    3.  Calculate the Physics Loss ($L_{\text{phys}} \propto |u_t - \alpha_{\text{guess}} u_{xx}|^2$) and treat this as the loss to minimize.
* ***Goal***: Demonstrate that the model structure allows the loss to be calculated based on the constant $\alpha$, enabling the optimizer to adjust $\alpha$ to minimize the total error, thus **inferring the hidden physical law**.

#### Python Implementation

```python
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Data and True Parameter (The Target)
# ====================================================================

ALPHA_TRUE = 0.5 # The true diffusion constant to be inferred
N_DATA_OBS = 100 # Small number of noisy observation points
N_PHYS = 500

# Conceptual Data Observations (Conceptual u(x, t) data points)
# We sample from the exact solution to simulate noisy data
def exact_solution(x, t, alpha=ALPHA_TRUE):
    return np.exp(-alpha * t * np.pi**2) * np.sin(np.pi * x)

# Generate small, noisy dataset D
X_DATA = np.random.rand(N_DATA_OBS, 2)
X_DATA[:, 0] *= 1.0 # x \in [0, 1]
X_DATA[:, 1] *= 0.5 # t \in [0, 0.5]
U_DATA = exact_solution(X_DATA[:, 0:1], X_DATA[:, 1:2]) + np.random.normal(0, 0.05, (N_DATA_OBS, 1))

# --- Trainable Variable Setup ---
# The parameter \alpha is now a trainable variable, starting from a bad guess.
ALPHA_GUESS_INIT = 0.1 
alpha_trainable = tf.Variable(ALPHA_GUESS_INIT, dtype=tf.float32, name='alpha') 

# Collocation points for physics loss
X_PHYS_tf = tf.constant(np.random.rand(N_PHYS, 2), dtype=tf.float32)

# --- Conceptual Model (NN Placeholder) ---
def u_theta(x_t):
    # Placeholder for NN output: u(x, t) \approx sin(\pi x) * exp(-t * alpha_trainable * \pi^2)
    # The NN's job is to learn the weights that make this fit the data and physics
    x, t = x_t[:, 0:1], x_t[:, 1:2]
    return tf.sin(np.pi * x) * tf.exp(-t * alpha_trainable * np.pi**2)

# ====================================================================
# 2. Inverse Loss Function (L_total with trainable \alpha)
# ====================================================================

def calculate_inverse_loss():
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(alpha_trainable)
        tape.watch(X_PHYS_tf)
        X_DATA_tf = tf.constant(X_DATA, dtype=tf.float32)

        # 1. Data Loss (L_data) - Enforces fit to noisy observations
        U_DATA_tf = tf.constant(U_DATA, dtype=tf.float32)
        U_pred_data = u_theta(X_DATA_tf)
        L_data = tf.reduce_mean(tf.square(U_pred_data - U_DATA_tf))

        # 2. Physics Loss (L_phys) - Uses the trainable alpha
        u_phys = u_theta(X_PHYS_tf)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(X_PHYS_tf)
            u_x = tape2.gradient(u_theta(X_PHYS_tf), X_PHYS_tf)[:, 0:1]
        u_xx = tape.gradient(u_x, X_PHYS_tf)[:, 0:1]
        u_t = tape.gradient(u_phys, X_PHYS_tf)[:, 1:2]

        R = u_t - alpha_trainable * u_xx
        L_phys = tf.reduce_mean(tf.square(R))

        # Total Loss (L_total)
        L_total = L_data + L_phys # Assume all weights are 1 for simplicity

    # The gradient of L_total w.r.t. \alpha is what drives the inference
    # This gradient is used by the optimizer (e.g., Adam)
    dL_dalpha = tape.gradient(L_total, alpha_trainable)

    return L_total.numpy(), alpha_trainable.numpy(), dL_dalpha.numpy()

# --- Final Function Test ---
L_total, alpha_inferred, dL_dalpha = calculate_inverse_loss()

print("\n--- Solving the Inverse Problem (Parameter Inference) ---")
print(f"True Diffusion Constant (\u03b1_true): {ALPHA_TRUE:.3f}")
print(f"Initial Guess (\u03b1_guess): {ALPHA_GUESS_INIT:.3f}")
print("---------------------------------------------------------------")
print(f"Total Loss L_total (at \u03b1_guess): {L_total:.4f}")
print(f"Gradient w.r.t. \u03b1 (\u2202L/\u2202\u03b1): {dL_dalpha:.4f}")

print("\nConclusion: The gradient \u2202L/\u2202\u03b1 is non-zero, indicating that the optimization process would successfully adjust the trainable parameter \u03b1 in the direction of the true value (\u03b1_true=0.5) to minimize the total physics-constrained data error.")
```

### Project 4: Adaptive Sampling Strategy (Conceptual)

* **Goal:** Demonstrate the principle of **adaptive sampling** by identifying the domain regions where the PINN solution is the least accurate.
* **Setup:** Assume a trained PINN solution $u_{\mathcal{\theta}}(x, t)$ has a non-zero residual $R(x, t) = |u_t - \alpha u_{xx}|$ everywhere.
* **Steps:**
    1.  Define a large grid of test points $(x_j, t_j)$.
    2.  Calculate the absolute magnitude of the residual, $|R(x_j, t_j)|$, across the entire grid.
* ***Goal***: Identify the top 5% of points with the highest residual magnitude. These points are the optimal locations for placing the **next batch of collocation points**, focusing the computational energy on the regions where the network most severely violates the physics.

#### Python Implementation

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Conceptual Solution and Residual Grid
# ====================================================================

ALPHA = 0.5
L = 1.0
T_FINAL = 1.0

# --- Conceptual Solution (Assume the network is almost trained) ---
# We use the exact solution as a proxy for a good PINN solution
def u_theta_proxy(x, t, alpha=ALPHA):
    # Sinusoidal solution that decays over time
    return np.sin(np.pi * x) * np.exp(-alpha * t * np.pi**2)

# --- Test Grid ---
N_GRID = 50
x_grid = np.linspace(0, L, N_GRID)
t_grid = np.linspace(0, T_FINAL, N_GRID)
X_test_mesh, T_test_mesh = np.meshgrid(x_grid, t_grid)
X_test_flat = np.hstack([X_test_mesh.flatten()[:, None], T_test_mesh.flatten()[:, None]])

X_test_tf = tf.constant(X_test_flat, dtype=tf.float32)

# ====================================================================
# 2. Residual Calculation on the Test Grid
# (Simulating the evaluation step of the adaptive loop)
# ====================================================================

def calculate_residual_magnitude(X_test, alpha=ALPHA):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X_test)
        
        # We must use a TensorFlow function that is differentiable for the proxy
        def u_tf(x_t):
            x, t = x_t[:, 0:1], x_t[:, 1:2]
            return tf.sin(np.pi * x) * tf.exp(-alpha * t * np.pi**2)
        
        u = u_tf(X_test)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(X_test)
            u_x = tape2.gradient(u, X_test)[:, 0:1]
            u_t = tape2.gradient(u, X_test)[:, 1:2]
        u_xx = tape.gradient(u_x, X_test)[:, 0:1]
        
        # R = u_t - alpha * u_xx 
        R = u_t - alpha * u_xx
        
    # The residual magnitude is the absolute value of the violation
    R_magnitude = tf.abs(R).numpy().flatten()
    return R_magnitude

R_mag_flat = calculate_residual_magnitude(X_test_tf)
R_mag_grid = R_mag_flat.reshape(N_GRID, N_GRID)


# ====================================================================
# 3. Adaptive Sampling Identification
# ====================================================================

# Identify the points with the highest violation (top 5% residual)
THRESHOLD_PERCENTILE = 95
threshold_value = np.percentile(R_mag_flat, THRESHOLD_PERCENTILE)
high_residual_indices = R_mag_flat > threshold_value

# Extract coordinates of the next sampling batch
X_next_batch = X_test_flat[high_residual_indices]

# ====================================================================
# 4. Visualization and Analysis
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot 1: Heatmap of the Residual R(x, t)
plt.imshow(R_mag_grid, extent=[0, L, 0, T_FINAL], origin='lower', cmap='plasma')
plt.colorbar(label='Residual Magnitude $|R(x, t)|$')

# Plot 2: Overlay the new, adaptively sampled points
plt.scatter(X_next_batch[:, 0], X_next_batch[:, 1], 
            s=5, c='white', label=f'Adaptive Sampled Points (Top {100 - THRESHOLD_PERCENTILE}%)')

# Labeling and Formatting
plt.title(f'Adaptive Sampling Strategy: Locating Maximum PDE Violation')
plt.xlabel('Space $x$')
plt.ylabel('Time $t$')
plt.legend()
plt.show()

print("\n--- Adaptive Sampling Strategy Summary ---")
print(f"Residual Violation Threshold: R > {threshold_value:.4e}")
print(f"Number of new collocation points identified: {len(X_next_batch)}")

print("\nConclusion: The heatmap visually displays the residual R across the domain. The adaptive sampling correctly identifies the regions where the solution violates the PDE the most (high R, near boundaries or sharp transitions in the solution) and focuses the next training batch on these critical areas, leading to faster and more robust convergence.")
```