## ⚛️ Chapter 17: Neural Quantum States (NQS) (Workbook)

The goal of this chapter is to establish the deepest fusion of physics and AI: representing the intractable quantum wavefunction using neural networks and solving for the ground state energy via classical optimization.

| Section | Topic Summary |
| :--- | :--- |
| **17.1** | Motivation — Quantum States as Data Distributions |
| **17.2** | The Variational Principle as Learning Objective |
| **17.3** | From Boltzmann Machines to Quantum Amplitudes |
| **17.4** | Variational Monte Carlo (VMC) |
| **17.5** | Example — Transverse-Field Ising Model |
| **17.6** | Complex-Valued Neural Networks |
| **17.7** | Neural Quantum States vs. Tensor Networks |
| **17.8** | Connection to Energy-Based Models (EBMs) |
| **17.9** | Stochastic Reconfiguration — Quantum Natural Gradient |
| **17.10–17.15**| Code Demo, Applications, and Takeaways |

---

### 17.1 Motivation — Quantum States as Data Distributions

> **Summary:** The central challenge is the **exponential scaling** of the Hilbert space with the number of particles $N$. The **wavefunction ($\psi$)** is a complex-valued function whose squared magnitude defines the observable probability distribution $P(\mathbf{s}) = |\psi(\mathbf{s})|^2$. **Neural Quantum States (NQS)** use a deep network ($\psi_{\boldsymbol{\theta}}$) to compress this exponential complexity into a polynomial number of parameters.

#### Quiz Questions

**1. The primary challenge in solving the quantum many-body problem that necessitates the use of NQS is:**

* **A.** The linearity of the Schrödinger equation.
* **B.** **The exponential scaling of the Hilbert space dimension with the number of particles ($2^N$)**. (**Correct**)
* **C.** The complexity of the classical Hamiltonian.
* **D.** The difficulty of calculating the partition function $Z$.

**2. The single variable that links the NQS neural network output $\psi(\mathbf{s})$ to the observable, classical world is:**

* **A.** The complex phase $\Phi(\mathbf{s})$.
* **B.** The total expected energy $E$.
* **C.** **The probability distribution $P(\mathbf{s}) = |\psi(\mathbf{s})|^2$**. (**Correct**)
* **D.** The local energy $E_{\text{loc}}(\mathbf{s})$.

---

#### Interview-Style Question

**Question:** The NQS approach is analogous to a powerful form of **generative modeling** (Chapter 14). Explain how the training goal of NQS differs philosophically from that of a standard GAN or VAE trained on image data.

**Answer Strategy:**
* **GAN/VAE (Data-Driven):** The network is trained on an **empirical dataset** (observed images). The goal is to maximize the likelihood of the **observed distribution**.
* **NQS (Law-Constrained):** The network is trained purely on the **theoretical law** (the Hamiltonian, $\hat{H}$). The goal is to minimize the expected energy, finding the theoretical **lowest-energy state**. The network is not fitting observed data, but the unobserved **quantum reality** dictated by the physical law.

---
***

### 17.2 The Variational Principle as Learning Objective

> **Summary:** The goal of NQS is to find the **ground state** (lowest energy $E_0$) of the system. The **Variational Principle** serves as the learning objective, stating that the expected energy $E[\psi_{\boldsymbol{\theta}}]$ of any trial wavefunction $\psi_{\boldsymbol{\theta}}$ is always an upper bound for the true energy $E_0$. The learning objective is therefore $L(\boldsymbol{\theta}) = \min E[\psi_{\boldsymbol{\theta}}]$, driven by a gradient (force) derived from the **Hamiltonian operator ($\hat{H}$)**.

#### Quiz Questions

**1. The formal learning objective for training a Neural Quantum State (NQS) network is to minimize the expected energy $E[\psi_{\boldsymbol{\theta}}]$, which is derived from which foundational quantum mechanics theorem?**

* **A.** Heisenberg Uncertainty Principle.
* **B.** **The Variational Principle**. (**Correct**)
* **C.** The Equipartition Theorem.
* **D.** The Bellman Optimality Equation.

**2. In the NQS energy functional, $E = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle}$, the role of the **Hamiltonian operator ($\hat{H}$)** is analogous to which concept in machine learning?**

* **A.** The policy $\pi$.
* **B.** The learning rate $\eta$.
* **C.** **The loss function operator**. (**Correct**)
* **D.** The normalization constant $Z$.

---

#### Interview-Style Question

**Question:** The NQS approach requires minimizing energy using gradient descent. The final equation shows that the gradient of the expected energy is proportional to the difference $E_{\text{loc}} - E$. Explain the role of the **local energy ($E_{\text{loc}}$)** term in calculating this gradient.

**Answer Strategy:** The **local energy ($E_{\text{loc}}$)**, defined as $E_{\text{loc}}(\mathbf{s}) = \frac{(\hat{H}\psi_{\boldsymbol{\theta}})(\mathbf{s})}{\psi_{\boldsymbol{\theta}}(\mathbf{s})}$, is the exact energy of a specific configuration $\mathbf{s}$ under the current trial wavefunction $\psi_{\boldsymbol{\theta}}$. The gradient term uses $E_{\text{loc}}$ as the error signal. If $E_{\text{loc}}$ is much higher than the average energy $E$, the gradient pushes the parameters to reduce the amplitude of that configuration (reduce its energy cost), driving the system toward the true energy minimum.

---
***

### 17.3 From Boltzmann Machines to Quantum Amplitudes

> **Summary:** The NQS **neural ansatz** ($\psi_{\boldsymbol{\theta}}$) must encode both the real magnitude (amplitude) and the complex **phase** of the quantum state. The **Restricted Boltzmann Machine (RBM)** (Chapter 14.4) is a common ansatz because its summation over hidden units naturally models the system's **entanglement**. The network acts as a **quantum compressor**, storing the exponential complexity of the Hilbert space in a polynomial number of weights.

### 17.4 Variational Monte Carlo (VMC)

> **Summary:** **Variational Monte Carlo (VMC)** is the stochastic optimization procedure used to train NQS. VMC estimates the expected energy $E = \mathbb{E}_{P(\mathbf{s})}[ E_{\text{loc}}(\mathbf{s}) ]$ by averaging the local energy over configurations $\mathbf{s}$ drawn from the probability distribution $P(\mathbf{s}) = |\psi_{\boldsymbol{\theta}}(\mathbf{s})|^2$. The configurations are sampled using a **classical Markov Chain Monte Carlo (MCMC)** method (e.g., Metropolis) applied to the squared amplitude.

### 17.5 Example — Transverse-Field Ising Model

> **Summary:** The **Transverse-Field Ising Model (TFIM)** provides a non-trivial testbed for NQS, studying quantum phase transitions. NQS successfully calculates the ground state energy for systems too large for traditional methods, scaling polynomially with system size $N$. The network automatically learns the correct **entanglement structure** (correlations) required by the TFIM Hamiltonian.

### 17.6 Complex-Valued Neural Networks

> **Summary:** Since the wavefunction is a **complex field**, the NQS network must represent both the real magnitude ($R$) and the imaginary phase ($\Phi$). Training requires extending the calculus of optimization using **Wirtinger derivatives** for complex gradients. This complex optimization is analogous to minimizing energy in a coupled system constrained by the **phase relationship**.

### 17.7 Neural Quantum States vs. Tensor Networks

> **Summary:** NQS are generally **globally flexible** and can handle systems with **high entanglement**. They compete with **Tensor Networks (TNs)**, which are established methods that are limited to systems with **low local entanglement**. NQS are viewed as **global generalizers** and provide a highly flexible basis set (ansatz) for the quantum state.

### 17.8 Connection to Energy-Based Models (EBMs)

> **Summary:** NQS are conceptualized as **quantum generalizations of EBMs** (Chapter 14.2). Both models use an exponential relationship between energy and probability (or amplitude). The critical difference is that NQS uses a **complex energy functional** and must model the complex phase, allowing for **quantum interference** (Section 17.6).

### 17.9 Stochastic Reconfiguration — Quantum Natural Gradient

> **Summary:** Standard gradient descent is unstable in the quantum state manifold. **Stochastic Reconfiguration (SR)** solves this by adapting the optimization direction using the **Quantum Fisher Information Matrix ($S_{ij}$)**. SR is the quantum analogue of the **Natural Gradient** and ensures the optimization follows the efficient **geodesic motion** on the manifold of normalized quantum states.

---

## 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core components of the NQS methodology using statistical and energetic analogies.

### Project 1: Simulating Wavefunction Normalization and Probability

* **Goal:** Numerically verify the quantum probability relationship and the role of the wavefunction's magnitude.
* **Setup:** Define a small 3-spin system ($N=3$, $2^3=8$ states). Define a complex trial wavefunction $\psi(\mathbf{s})$ manually (e.g., using random complex numbers for each state).
* **Steps:**
    1.  Calculate the probability distribution $P(\mathbf{s}) = |\psi(\mathbf{s})|^2$ for all 8 states.
    2.  Check the normalization condition: $\sum_{\mathbf{s}} P(\mathbf{s})$ should equal 1 (or near 1 if $\psi$ was pre-normalized).
* ***Goal***: Confirm the fundamental link: the complex amplitude determines the observable probability, demonstrating the basic physical constraint.

### Project 2: Variational Energy Estimation (VMC Conceptual)

* **Goal:** Implement the core statistical step of **Variational Monte Carlo (VMC)**: calculating the expected energy by averaging the local energy.
* **Setup:** Use a simple Hamiltonian (e.g., the 1D Ising Hamiltonian without the transverse field, $\hat{H} = -J \sum \sigma^z_i \sigma^z_{i+1}$). Use a small, fixed trial wavefunction $\psi$ (e.g., all states are equally probable: $1/\sqrt{2^N}$).
* **Steps:**
    1.  Implement the calculation of the **local energy** $E_{\text{loc}}(\mathbf{s})$ for several sample spin configurations $\mathbf{s}$.
    2.  Estimate the total expected energy $E$ by averaging the calculated $E_{\text{loc}}(\mathbf{s})$ values.
* ***Goal***: Show that $E$ is simply the mean of the local energies, confirming that VMC reduces the complex quantum expectation to a statistically tractable sample average.

### Project 3: Simulating Quantum Relaxation (Energy Descent)

* **Goal:** Track the energy minimization process, illustrating the physical analogy of **quantum relaxation**.
* **Setup:** Simulate a conceptual optimization process (e.g., 50 epochs). Start the expected energy $E_{\text{current}}$ at a high value (excited state).
* **Steps:**
    1.  Model the optimization by having the expected energy $E_{\text{current}}$ decrease monotonically at every step.
    2.  Plot the expected energy $E(t)$ vs. optimization step $t$.
* ***Goal***: Show that the energy curve is **monotonically decreasing** and stabilizes at the ground state energy $E_0$, confirming that the classical optimization process simulates the natural physical dynamic of energy minimization.

### Project 4: Comparing Entanglement Capacity (RBM vs. Simple)

* **Goal:** Illustrate the superior **entanglement capacity** of the RBM ansatz over a simple factorized (uncorrelated) ansatz.
* **Setup:** Define two different trial wavefunctions for a 4-spin system (16 states):
    1.  **Simple Ansatz:** $\psi_{\text{simple}}(\mathbf{s}) = \prod_i \psi(s_i)$ (product state, no entanglement).
    2.  **RBM Ansatz:** Use the RBM formula $\psi_{\text{RBM}}(\mathbf{s})$ (Section 17.10) (can model entanglement).
* **Steps:**
    1.  Generate a random, entangled target state $\psi_{\text{target}}$ (conceptually).
    2.  Compute the **distance** (e.g., fidelity) between the RBM ansatz and the target, and the simple ansatz and the target.
* ***Goal***: Show that the RBM ansatz achieves a significantly smaller distance (higher fidelity) to the entangled target state than the simple product state, demonstrating the necessity of hidden units for encoding complex quantum correlations.
