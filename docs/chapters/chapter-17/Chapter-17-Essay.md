# **17. Neural Quantum States (NQS)**
---

## **Introduction**

In Chapter 16, we explored **Physics-Informed Neural Networks (PINNs)**, which embed classical differential equations directly into loss functions to solve forward and inverse problems constrained by conservation laws—minimizing augmented energy functionals combining empirical data loss with physics residual penalties computed via automatic differentiation. This approach transformed PDE solving from numerical discretization to variational optimization, ensuring solutions respect fundamental laws like Navier-Stokes or the heat equation while extrapolating robustly beyond sparse training data. This chapter extends the physics-constrained learning paradigm to the ultimate frontier: **quantum mechanics**, where the exponential curse of dimensionality renders exact computation impossible for systems with even modest numbers of interacting particles. A quantum state of $N$ spins requires storing $2^N$ complex amplitudes—for just 50 spins, this exceeds $10^{15}$ parameters, vastly beyond any classical computer's memory. Traditional methods (exact diagonalization, tensor networks) either fail to scale or succeed only for systems with restricted entanglement structures obeying area laws.

At the heart of this chapter lies the **Neural Quantum State (NQS)** framework, which represents the quantum wavefunction $\psi(\mathbf{s})$ as a parameterized neural network $\psi_{\mathbf{\theta}}(\mathbf{s})$, compressing exponential Hilbert space complexity into a tractable polynomial number of weights. The training objective is not empirical data fitting but the **Variational Principle**: minimizing the expected energy $E[\psi_{\mathbf{\theta}}] = \langle \psi_{\mathbf{\theta}} | \hat{H} | \psi_{\mathbf{\theta}} \rangle / \langle \psi | \psi \rangle$ guarantees that the learned state is an upper bound to the true ground state energy, with equality achieved at the optimal parameters $\mathbf{\theta}^*$. **Variational Monte Carlo (VMC)** enables stochastic estimation of this energy by sampling configurations $\mathbf{s}_i$ from the probability distribution $P(\mathbf{s}) = |\psi_{\mathbf{\theta}}(\mathbf{s})|^2$ and averaging local energies $E_{\text{loc}}(\mathbf{s}) = (\hat{H}\psi_{\mathbf{\theta}})(\mathbf{s})/\psi_{\mathbf{\theta}}(\mathbf{s})$ via MCMC sampling. The **Restricted Boltzmann Machine (RBM)** provides the canonical neural ansatz, with visible units representing physical spins and hidden units encoding entanglement through the exponential sum $\psi_{\mathbf{\theta}}(\mathbf{s}) = \sum_{\mathbf{h}} e^{a^\top \mathbf{s} + b^\top \mathbf{h} + \mathbf{s}^\top W \mathbf{h}}$, where the marginalization over hidden states introduces the nonlocal correlations necessary to represent complex quantum states.

By the end of this chapter, you will understand NQS as implementing **quantum relaxation** through optimization—gradient descent in parameter space drives the wavefunction toward lower energy along geodesics on the Hilbert manifold, analogous to imaginary-time Schrödinger evolution. **Complex-valued neural networks** are required because wavefunctions possess both amplitude and phase ($\psi = R e^{i\Phi}$), with phase governing quantum interference; Wirtinger derivatives extend backpropagation to complex gradients. **Stochastic Reconfiguration (SR)**, the quantum Natural Gradient, preconditions standard energy gradients with the Quantum Fisher Information Matrix $S_{ij}$ to account for the non-Euclidean geometry of normalized quantum states, ensuring stable convergence by moving along optimal paths in Hilbert space. Advanced architectures include autoregressive networks for efficient sampling, CNNs exploiting translational invariance on lattices, and Transformers capturing long-range entanglement through self-attention. Applications span quantum spin systems (Transverse-Field Ising Model ground states), electronic structure (correlated electron wavefunctions in molecules), quantum chemistry (chemical accuracy for energies and densities), and hybrid quantum-classical algorithms (Variational Quantum Eigensolvers). Chapter 18 will explore **Graph Neural Networks**, which model information propagation on arbitrary discrete topologies, providing the natural architecture for systems where interactions follow specific geometric connectivity patterns rather than dense all-to-all coupling or regular lattice structures.

---

## **Chapter Outline**

| **Sec.** | **Title** | **Core Ideas & Examples** |
|:---------|:----------|:-------------------------|
| **17.1** | Motivation — Quantum States as Data Distributions | Exponential Hilbert space scaling ($2^N$ for $N$ spins); wavefunction $\psi(\mathbf{s})$ defines probability $p(\mathbf{s}) = \|\psi(\mathbf{s})\|^2$; NQS uses neural network $\psi_{\mathbf{\theta}}$ to compress exponential complexity into polynomial parameters; training minimizes expected energy (not data fitting); quantum state as probability distribution analogy |
| **17.2** | The Variational Principle as Learning Objective | Ground state energy $E[\psi] = \langle \psi \| \hat{H} \| \psi \rangle / \langle \psi \| \psi \rangle$; Variational Principle: any trial $\psi_{\mathbf{\theta}}$ gives $E[\psi_{\mathbf{\theta}}] \ge E_0$ (upper bound); optimization minimizes expected energy via gradient descent; Hamiltonian $\hat{H}$ acts as loss operator (universe defines cost surface) |
| **17.3** | From Boltzmann Machines to Quantum Amplitudes | Neural ansatz encodes complex wavefunction $\psi_{\mathbf{\theta}}(\mathbf{s}) = \sqrt{P_{\mathbf{\theta}}} e^{i\Phi_{\mathbf{\theta}}}$ (amplitude + phase); RBM ansatz $\psi_{\mathbf{\theta}}(\mathbf{s}) = \sum_{\mathbf{h}} e^{a^\top \mathbf{s} + b^\top \mathbf{h} + \mathbf{s}^\top W \mathbf{h}}$; hidden units encode entanglement (latent spins); quantum compressor (exponential Hilbert space → polynomial weights) |
| **17.4** | Variational Monte Carlo (VMC) | Energy estimation via sampling: $E = \mathbb{E}_{P(\mathbf{s})}[E_{\text{loc}}(\mathbf{s})]$ where $E_{\text{loc}} = (\hat{H}\psi_{\mathbf{\theta}})/\psi_{\mathbf{\theta}}$; MCMC samples from $P(\mathbf{s}) = \|\psi_{\mathbf{\theta}}\|^2$; gradient calculation through sampled local energies; stochastic optimization on quantum potential surface (SGD on quantum domain) |
| **17.5** | Example — Transverse-Field Ising Model | TFIM Hamiltonian $\hat{H} = -J\sum_{\langle ij\rangle}\hat{\sigma}^z_i\hat{\sigma}^z_j - h\sum_i \hat{\sigma}^x_i$ (ordering vs transverse field); quantum phase transition as $J/h$ varies; NQS (RBM or deep network) represents ground state $\psi_{\mathbf{\theta}}(\mathbf{s})$; VMC optimizes to find minimal energy; scales to $N \approx 100$ spins (exponential compression) |
| **17.6** | Complex-Valued Neural Networks | Wavefunction has complex amplitude $\psi_{\mathbf{\theta}} = R_{\mathbf{\theta}} e^{i\Phi_{\mathbf{\theta}}}$ (magnitude + phase); phase governs quantum interference; Wirtinger derivatives extend backpropagation to complex gradients; implementation: coupled real/imaginary channels; two-phase interference of gradient flow (real drives amplitude, imaginary controls phase) |
| **17.7** | Neural Quantum States vs. Tensor Networks | Comparison table: TNs (local/structured, limited entanglement, polynomial expressivity, transparent) vs NQS (global/flexible, high entanglement, exponential expressivity, black-box); TNs efficient for area-law systems, NQS handles complex nonlocal correlations; analogy: fixed polynomial fit vs deep nonlinear network |
| **17.8** | Connection to Energy-Based Models (EBMs) | EBM: $P_{\mathbf{\theta}}(\mathbf{x}) \propto e^{-E_{\mathbf{\theta}}(\mathbf{x})/T}$ (classical probability); NQS: $\psi_{\mathbf{\theta}}(\mathbf{s}) \propto e^{-E_{\mathbf{\theta}}(\mathbf{s})/2}$ (quantum amplitude); difference: amplitude vs probability (complex phase enables interference); NQS as quantum generalization of EBMs (probability becomes amplitude) |
| **17.9** | Stochastic Reconfiguration — Quantum Natural Gradient | Problem: standard gradient descent unstable on non-Euclidean quantum manifold; Stochastic Reconfiguration preconditions gradient: $S_{ij}\Delta\theta_j = -\nabla_i E$; Quantum Fisher Information $S_{ij}$ defines Hilbert space metric (statistical distinguishability of states); ensures geodesic motion on manifold of normalized states; quantum Natural Gradient analogy |
| **17.10** | Code Demo — Minimal Neural Quantum State | PyTorch RBM implementation: `NQS` class with weights $W$, biases $a,b$; forward pass computes $\psi(\mathbf{s}) = \exp(a^\top \mathbf{s} + \log\sum_h \exp(b^\top \mathbf{h} + \mathbf{s}^\top W \mathbf{h}))$ via logsumexp; entanglement encoding through hidden unit summation; input: spin configuration $\mathbf{s}$, output: complex amplitude; VMC would follow with local energy sampling |
| **17.11** | Deep Quantum Architectures | Beyond shallow RBM: autoregressive NQS (sequential modeling, exact sampling), CNN-NQS (translational invariance for lattices), Transformer-NQS (self-attention for long-range entanglement); hierarchical structures for multi-scale correlations; hybrid quantum-classical (VQE: classical optimizer + quantum circuit ansatz); specialized inductive biases match system physics |
| **17.12** | Quantum Probability and Information Geometry | Manifold of quantum states (curved Hilbert space geometry); Quantum Fisher Information Matrix $S_{ij}$ measures state distinguishability (metric tensor); SR uses $S^{-1}$ to precondition gradients (geodesic descent); information compression view: project exponential true state onto polynomial manifold; geometry dictates optimal learning dynamics |
| **17.13** | Applications | Quantum spin systems (Ising, Heisenberg ground states, quantum criticality); electronic structure (correlated electron wavefunctions, Hubbard model); quantum chemistry (molecular energies/densities with chemical accuracy); quantum dynamics (time-evolving PINNs for Schrödinger equation); VQE on quantum hardware (hybrid classical-quantum optimization); transforms high-D quantum observables to statistical inference |
| **17.14** | Physical Interpretation — Learning as Quantum Relaxation | Gradient descent as imaginary-time Schrödinger evolution (quantum relaxation toward ground state); classical relaxation: overdamped particle minimizing potential; quantum: trial wavefunction flows to minimal energy in Hilbert manifold; parameter updates $\leftrightarrow$ state relaxation driven by energy gradient force; unification: ML and quantum physics share common energetic geometry |
| **17.15** | Bridge to Quantum Information and Beyond | Extensions: Quantum Boltzmann Machines (mixed states/density matrices), learning entanglement entropy/mutual information, neural quantum dynamics (time-dependent Schrödinger); foreshadowing Chapter 18 GNNs: model information propagation on arbitrary graph topologies (discrete field theories, local interaction structures); progression: classical optimization → quantum amplitudes → graph-structured interactions |

---

## **17.1 Motivation — Quantum States as Data Distributions**

### **The Problem: Exponential Scaling**

----- 

The central challenge in computational quantum physics is the **many-body problem**. The state of a system of $N$ interacting quantum particles is described by its **wavefunction $\psi(\mathbf{s})$** (or state vector).

* **Hilbert Space:** The dimension of the space required to store this wavefunction, known as the Hilbert space, grows **exponentially** with the number of particles $N$.
* **Intractability:** For even a few dozen interacting spins (e.g., $N=50$), the state space size $2^N$ exceeds the storage capacity of any classical computer. This makes exact simulation impossible.

### **The Observation: Probability and Amplitude**

-----

Quantum mechanics provides the first link to the probabilistic world of machine learning:

* **Probability Distribution:** The wavefunction $\psi(\mathbf{s})$ is a **complex-valued function** whose squared magnitude defines the probability of observing the system in a configuration $\mathbf{s}$:

$$
p(\mathbf{s}) = |\psi(\mathbf{s})|^2
$$
* **Wavefunction as Data:** This means the quantum state *is* a probability distribution.

### **The Idea: Neural Ansatz and Optimization**

-----

The **Neural Quantum State (NQS)** approach proposes using a highly flexible, parameterized neural network ($\psi_{\mathbf{\theta}}$) to approximate the true, intractable wavefunction $\psi(\mathbf{s})$.

* **Representation:** The network weights $\mathbf{\theta}$ compress the exponential complexity of the quantum state into a tractable number of parameters.
* **Training:** The network is trained not by matching data, but by minimizing the system's fundamental **expected energy** (Section 17.2).

### **Analogy: Fitting the Distribution of the Universe**

-----

The NQS method is analogous to a powerful form of generative modeling (Chapter 14):
* **Classical ML:** We fit an energy function $E_{\mathbf{\theta}}(\mathbf{x})$ (or distribution $P_{\mathbf{\theta}}$) to an empirically observed dataset.
* **NQS:** We fit the **quantum probability amplitude** $\psi_{\mathbf{\theta}}(\mathbf{s})$ to the theoretical distribution of the system's lowest energy state.

The network learns the structural features and correlations (entanglement) required by the physical law, effectively fitting the underlying probabilistic structure of the system itself.

!!! tip "Quantum State as Probability Compressor"
    Unlike classical machine learning where we fit a model to observed data, NQS compresses the exponential Hilbert space ($2^N$ amplitudes) into a polynomial-parameter neural network ($\approx N^2$ weights). This compression preserves the essential quantum correlations (entanglement) needed to represent the ground state. Think of the network as learning the "code" that reconstructs the full quantum distribution from minimal information—analogous to how a deep generative model learns the latent structure underlying high-dimensional data.

---

## **17.2 The Variational Principle as Learning Objective**

The challenge of finding the **ground state** (lowest energy state) of a quantum system, which is the goal of **Neural Quantum States (NQS)**, is mathematically translated into a computationally solvable optimization problem by utilizing the **Variational Principle**. This principle serves as the fundamental loss function for training the neural network wavefunction.

-----

### **Quantum Ground State Energy $E[\psi]$**

-----

In quantum mechanics, the energy of a given state $\psi$ is determined by the **Hamiltonian operator ($\hat{H}$)**, which represents the total energy of the system. The expected energy $E[\psi]$ of a state is the expectation value of this operator:

$$
E[\psi] = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle}
$$

### **The Variational Principle**

-----

The Variational Principle provides a powerful, universal constraint on this energy: **The true ground state energy $E_0$ is the lowest possible expected energy a system can have**.

This means that for any arbitrary **trial wavefunction** $\psi_{\mathbf{\theta}}$ (our neural network approximation), its calculated expected energy will always be an upper bound for the true ground state energy $E_0$:

$$
E[\psi_{\mathbf{\theta}}] \ge E_0
$$

The mathematical task is thus clear: the **optimal wavefunction ($\psi_{\mathbf{\theta}}^*$) is the one that minimizes the expected energy $E[\psi_{\mathbf{\theta}}]$**.

### **Optimization: Learning via Gradient Descent**

-----

The NQS approach leverages the numerical optimization techniques of deep learning (Part II) to find this minimum.

* **Learning Objective:** The objective function $L$ for NQS training is the expected energy: $L(\mathbf{\theta}) = E[\psi_{\mathbf{\theta}}]$.
* **Gradient-Based Optimization:** The parameters $\mathbf{\theta}$ of the neural network are iteratively adjusted using gradient descent to drive the system toward the minimum energy. The gradient of the expected energy is:
$$
\nabla_{\mathbf{\theta}} E = 2 \operatorname{Re}\left[\left\langle \left(E_{\text{loc}}-E\right)\nabla_{\mathbf{\theta}} \ln \psi_{\mathbf{\theta}}\right\rangle\right]
$$
This gradient is calculated through sampling, a process known as **Variational Monte Carlo (VMC)** (Section 17.4).

### **Analogy: The Universe Defines the Loss**

-----

The Variational Principle establishes the definitive link between quantum physics and optimization:

* **Hamiltonian $\leftrightarrow$ Loss Operator:** The **Hamiltonian operator ($\hat{H}$)** plays the role of the loss function operator. It is the fundamental physical law that dictates the cost of any state.
* **Energy Minimization $\leftrightarrow$ Learning:** The process of training the NQS network is entirely equivalent to a generalized **energy minimization**. The parameters $\mathbf{\theta}$ are driven by a force (gradient) toward the global minimum defined by the quantum mechanical laws of the system.

This perspective means that the *universe itself* defines the optimization cost surface for the learning algorithm.

---

## **17.3 From Boltzmann Machines to Quantum Amplitudes**

The development of **Neural Quantum States (NQS)** relies on adapting classical neural architectures to represent the specific mathematical properties of a quantum state. This leads to the central concept of the **neural ansatz**: mapping a network designed for classical probability (like the Boltzmann Machine) to one that encodes **quantum probability amplitudes**.

-----

### **Neural Ansatz: Encoding Amplitude and Phase**

-----

A quantum wavefunction $\psi(\mathbf{s})$ is, in general, a **complex-valued field**. It must be represented in a way that allows a classical network, typically optimized with real-valued parameters, to encode both its magnitude (amplitude) and its complex phase:

$$
\psi_{\mathbf{\theta}}(\mathbf{s}) = \sqrt{P_{\mathbf{\theta}}(\mathbf{s})} e^{i \Phi_{\mathbf{\theta}}(\mathbf{s})}
$$

* **Amplitude ($R$):** $\sqrt{P_{\mathbf{\theta}}(\mathbf{s})}$ determines the probability of observing the state $\mathbf{s}$.
* **Phase ($\Phi$):** $e^{i \Phi_{\mathbf{\theta}}(\mathbf{s})}$ governs quantum interference and the time evolution of the system.

### **The Restricted Boltzmann Machine (RBM) Ansatz**

-----

The most historically significant and widely used neural ansatz for NQS is based on the **Restricted Boltzmann Machine (RBM)** (Chapter 14.4).

The RBM, a bipartite network of visible ($\mathbf{s}$) and hidden ($\mathbf{h}$) units, models the joint energy. The NQS utilizes the RBM's normalized **probability amplitude**:

$$
\psi_{\mathbf{\theta}}(\mathbf{s}) = \sum_{\mathbf{h}} e^{a^\top \mathbf{s} + b^\top \mathbf{h} + \mathbf{s}^\top W \mathbf{h}}
$$

* The summation over the hidden units ($\sum_{\mathbf{h}}$) is the key feature that introduces **non-trivial correlations** (analogous to entanglement) necessary to represent complex quantum states.
* The parameters $\{a, b, W\}$ (collectively $\mathbf{\theta}$) are the adjustable variational parameters of the network, which are often generalized to be complex numbers to explicitly encode the phase (Section 17.6).

### **Interpretation: Quantum Compressor**

-----

* **Entanglement Modeling:** The hidden units of the RBM are interpreted as **latent spins** that model the **entanglement** structure of the quantum system. The weights $W$ encode the complex, global correlation between the visible spins.
* **Spin Configuration $\leftrightarrow$ State:** The visible units $\mathbf{s}$ correspond to the physical spin configurations of the system.
* **Quantum Compressor:** The NQS network acts as a **quantum compressor**, storing the essential information of the exponential Hilbert space within a tractable, polynomial number of network weights. By minimizing the energy (Section 17.2), the network learns the most efficient encoding of the quantum state's entanglement structure.

---

## **17.4 Variational Monte Carlo (VMC)**

The optimization of a **Neural Quantum State (NQS)** (Section 17.3) requires minimizing the expected energy $E[\psi_{\mathbf{\theta}}]$ (Section 17.2). Since the full expectation integral is intractable for large, many-body systems, this energy calculation must be performed through **stochastic sampling**, a technique known as **Variational Monte Carlo (VMC)**.

### **Sampling-Based Energy Estimation**

-----

The expected energy functional involves integration over the exponentially large Hilbert space:

$$
E = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle} = \sum_{\mathbf{s}} |\psi_{\mathbf{\theta}}(\mathbf{s})|^2 \frac{(\hat{H}\psi_{\mathbf{\theta}})(\mathbf{s})}{\psi_{\mathbf{\theta}}(\mathbf{s})}
$$

VMC solves this by rewriting the expectation as an integral over the system's probability distribution $P(\mathbf{s}) = |\psi_{\mathbf{\theta}}(\mathbf{s})|^2$:

$$
E = \mathbb{E}_{P(\mathbf{s})}\left[ E_{\text{loc}}(\mathbf{s}) \right]
$$

1.  **Local Energy ($E_{\text{loc}}$):** The key quantity is the **local energy**, defined for a specific configuration $\mathbf{s}$:

$$
E_{\text{loc}}(\mathbf{s}) = \frac{(\hat{H}\psi_{\mathbf{\theta}})(\mathbf{s})}{\psi_{\mathbf{\theta}}(\mathbf{s})}
$$

    This quantity can be computed directly, as the Hamiltonian operator $\hat{H}$ usually only involves local interactions between nearby particles.
2.  **Monte Carlo Estimate:** The integral for $E$ is approximated by averaging the local energy over $N_{\text{samples}}$ configurations $\mathbf{s}_i$ drawn from the probability distribution $P(\mathbf{s})$:

$$
E \approx \frac{1}{N_{\text{samples}}} \sum_{i=1}^{N_{\text{samples}}} E_{\text{loc}}(\mathbf{s}_i)
$$

### **The Optimization Loop: Gradient through Sampling**

-----

VMC requires an iterative loop, blending the optimization of the neural network parameters with the statistical mechanics of sampling:

1.  **Sampling:** Generate configurations $\mathbf{s}_i$ from the model's probability distribution $P(\mathbf{s}) = |\psi_{\mathbf{\theta}}(\mathbf{s})|^2$. This is typically achieved using a classical **Markov Chain Monte Carlo (MCMC)** method (Chapter 2.5), like the Metropolis algorithm (Chapter 7.3), applied to the squared amplitude of the wavefunction.
2.  **Gradient Calculation:** Compute the gradient $\nabla_{\mathbf{\theta}} E$ using the sampled local energies.
3.  **Update:** Adjust the network parameters $\mathbf{\theta}$ via gradient descent or a specialized method like **Stochastic Reconfiguration** (Section 17.9).

### **Analogy: Stochastic Optimization on a Quantum Surface**

-----

VMC is the direct equivalent of **Stochastic Gradient Descent (SGD)** (Chapter 5.4) applied to the quantum domain:

* **Stochastic Optimization:** Both methods estimate the true gradient (force) using small, noisy samples (mini-batches/Monte Carlo samples).
* **Energy Minimization:** VMC is the optimization procedure for training the energy-based model on a **quantum potential surface**. The network learns the distribution that minimizes the expected energy, successfully bridging classical statistical sampling with quantum mechanical optimization.

!!! example "VMC for the Hydrogen Molecule Ground State"
    Consider computing the ground state of H$_2$. The exact wavefunction requires integrating over all electron coordinates—intractable for classical computers. VMC with an NQS ansatz samples electron configurations $\mathbf{s}_i$ from $|\psi_{\mathbf{\theta}}(\mathbf{s})|^2$ using MCMC (Metropolis sampling). For each sample, compute the local energy $E_{\text{loc}}(\mathbf{s}_i) = (\hat{H}\psi_{\mathbf{\theta}})(\mathbf{s}_i)/\psi_{\mathbf{\theta}}(\mathbf{s}_i)$ where $\hat{H}$ includes kinetic energy and Coulomb interactions. Average these local energies to estimate $E[\psi_{\mathbf{\theta}}]$, then update network weights via gradient descent. After optimization, the learned $\psi_{\mathbf{\theta}}$ predicts molecular binding energy and electron density to chemical accuracy.

---

## **17.5 Example — Transverse-Field Ising Model**

The **Transverse-Field Ising Model (TFIM)** is a foundational model in condensed matter physics, often used to study quantum phase transitions and quantum criticality. It serves as an ideal, non-trivial example for demonstrating how **Neural Quantum States (NQS)** (Section 17.3) can be employed to find the **quantum ground state** of an interacting many-body system.
### **The System and Hamiltonian**

-----

The TFIM describes a lattice of $N$ interacting spins (often simplified to $\sigma^z = \pm 1$ and $\sigma^x$ operators) subject to two competing forces:

1.  **Ising Interaction (Ordering Force):** Tries to align neighboring spins in the $z$-direction (analogous to the classical Ising model, Chapter 8.3).
2.  **Transverse Field (Disordering Force):** Tries to flip spins in the $x$-direction.

The Hamiltonian operator $\hat{H}$ for the TFIM is:

$$
\hat{H} = -J\sum_{\langle ij\rangle}\hat{\sigma}^z_i\hat{\sigma}^z_j - h\sum_i \hat{\sigma}^x_i
$$

* **$J$:** The interaction strength between nearest neighbors $\langle ij \rangle$.
* **$h$:** The strength of the transverse magnetic field.

The system undergoes a **quantum phase transition** as the ratio of $J$ to $h$ changes.

### **State Representation: The Neural Ansatz**

-----

Since the exact calculation of the wavefunction $\psi$ is impossible for large $N$, NQS represents the ground state wavefunction $\psi_{\mathbf{\theta}}(\mathbf{s})$ using a flexible neural architecture:

* **Ansatz:** A common choice is the **Restricted Boltzmann Machine (RBM)** (Section 17.3), though deep autoregressive networks or specialized CNN-based structures (Section 17.11) may also be used.
* **Input:** The input $\mathbf{s}$ is the $N$-dimensional spin configuration.
* **Output:** The complex quantum amplitude $\psi_{\mathbf{\theta}}(\mathbf{s})$ for that configuration.

### **Optimization and Observation**

-----

The network parameters $\mathbf{\theta}$ are optimized using the **Variational Monte Carlo (VMC)** method (Section 17.4) to minimize the system's expected energy $E[\psi_{\mathbf{\theta}}]$.

* **Challenge:** Traditional methods (like exact diagonalization) are limited to small systems (e.g., $N \approx 20$).
* **NQS Advantage:** The NQS approach scales effectively to much larger systems ($N \approx 100$ or more), capturing the phase transition behavior with a number of parameters that scales only polynomially with $N$. This demonstrates the network's ability to compress the **exponential complexity** of the quantum state into a tractable set of learned weights.

### **Analogy: Quantum Compressor**

-----

The neural network acts as a **quantum compressor**. The learned weights implicitly encode the complex, non-local correlations (entanglement) that define the system's quantum structure. This process is analogous to classical dimensionality reduction (Chapter 3) but applied to the Hilbert space, finding the essential, low-dimensional representation of the quantum state.

---

## **17.6 Complex-Valued Neural Networks**

The **Neural Quantum State (NQS)** approach (Section 17.3) faces a significant challenge because the quantum **wavefunction $\psi(\mathbf{s})$** is, in general, a **complex-valued field**. It has both a real magnitude (amplitude) and a phase, $e^{i\Phi_{\mathbf{\theta}}}$. This necessitates generalizing the standard real-valued neural network architecture to handle complex numbers.
### **Wavefunction as a Complex Field**

-----

The wavefunction is represented by its polar form:

$$
\psi_{\mathbf{\theta}} = R_{\mathbf{\theta}} e^{i\Phi_{\mathbf{\theta}}}
$$

* **Magnitude ($R_{\mathbf{\theta}}$):** Encodes the probability amplitude, where $R_{\mathbf{\theta}}^2 = P(\mathbf{s})$.
* **Phase ($\Phi_{\mathbf{\theta}}$):** Encodes the phase of the quantum state, which is crucial for phenomena like **quantum interference** and predicting **time evolution**.

To fully utilize the NQS framework, the neural network must be capable of learning both $R_{\mathbf{\theta}}$ and $\Phi_{\mathbf{\theta}}$.

### **Training Challenge: Complex Gradients**

-----

Training a network with complex parameters and complex outputs requires extending the calculus of optimization.

* **Standard Gradient Descent (Chapter 5):** This is defined for real functions.
* **Complex Gradients:** Complex-valued parameters $\mathbf{\theta} = \mathbf{\theta}_{\text{real}} + i \mathbf{\theta}_{\text{imag}}$ require the use of **Wirtinger derivatives**. These derivatives extend the standard chain rule to complex functions, allowing the backpropagation of the energy gradient ($\nabla E$) to the complex-valued weights.

### **Implementation: Coupled Real and Imaginary Channels**

-----

In practice, a **complex-valued neural network** is implemented by treating the complex numbers as having two coupled channels: the real and imaginary parts.

* **Weights and Activations:** All weights ($W$) and node activations are represented as pairs of real numbers. The network's linear transformations and non-linear activation functions must be adapted to operate on these coupled channels.
* **Final Output:** The NQS network ultimately outputs the complex amplitude $\psi_{\mathbf{\theta}}$.

### **Analogy: Two-Phase Interference of Flow**

-----

The optimization in this complex space is analogous to a dynamical system where two coupled vector fields (the real and imaginary components of the gradient) interfere.

* **Interference:** The dynamics of the complex weights are constrained by the **phase relationship**. The optimization must balance the real part (which drives the amplitude and energy minimization) with the imaginary part (which controls the phase).
* **Geodesic Motion:** The goal is to move the system's state along the most efficient path (a **geodesic**) on the manifold of normalized quantum states (Section 17.9), where the metric is complex.

---

## **17.7 Neural Quantum States vs. Tensor Networks**

**Neural Quantum States (NQS)** emerged as a modern approach to tackling the quantum many-body problem, building on classical machine learning's capacity for complex function approximation. They directly compete with **Tensor Networks (TNs)**, which are the established, traditional framework in physics for efficiently representing and simulating quantum states. Comparing these two approaches reveals their complementary strengths and weaknesses in modeling quantum entanglement.

-----

### **Contrasting Architectures for the Wavefunction**

-----

Both NQS and TNs attempt to find an efficient, parameterized **ansatz** (trial wavefunction, $\psi_{\mathbf{\theta}}$) that compresses the exponentially large Hilbert space into a tractable set of parameters.

| Feature | Tensor Networks (TNs) | Neural Quantum States (NQS) |
| :--- | :--- | :--- |
| **Structure** | **Local and Structured**. Defined by fixed, low-rank tensor decompositions. | **Global and Flexible**. Defined by deep, highly non-linear neural networks (e.g., RBM, CNN). |
| **Entanglement** | **Limited to low entanglement**. Efficiency depends on the system having limited local correlation (area law). | **Can handle high entanglement**. Expressivity is high, allowing for complex, global correlations. |
| **Expressivity** | **Polynomial**. Scales predictably but is inherently limited. | **Potentially Exponential**. The universal approximation theorem applies to the function. |
| **Interpretability** | **Transparent**. Parameters relate directly to physical coupling and locality. | **Black-box**. Weights encode structure non-linearly, requiring specialized probing. |
| **Geometry** | **Classical Geometry** of the Hilbert space. | **Learned Geometry**. |

### **Complementarity and Analogy**

-----

* **TNs: The Local Experts:** Tensor Networks, such as Matrix Product States (MPS), are specialized to capture **local correlations efficiently**. They rely on physical insight to define the structure of the network.
* **NQS: The Global Generalizers:** Neural Quantum States generalize this concept by offering a model capable of encoding complex, **nonlocal entanglement** without requiring explicit knowledge of the underlying physical topology.
* **Analogy:** The difference is analogous to using a **simple, fixed polynomial fit** (TN) versus a **complex, non-linear deep network** (NQS) to approximate an unknown function. NQS provides a highly flexible basis set for the quantum state, often outperforming TNs for highly entangled (disordered) systems.

---

## **17.8 Connection to Energy-Based Models (EBMs)**

The mathematical structure of **Neural Quantum States (NQS)** (Section 17.3) establishes a profound link to classical **Energy-Based Models (EBMs)** (Chapter 14). Both frameworks define their respective distributions using an **exponential relationship** with an energy-like function, solidifying the idea that the physics of probability underlies all generative modeling.
### **Similarity: The Exponential Form**

-----

EBMs and NQS model their distributions using analogous exponential functions:

* **EBMs (Classical Probability):** Model the **probability density** $P_{\mathbf{\theta}}(\mathbf{x})$ using a real-valued energy function $E_{\mathbf{\theta}}(\mathbf{x})$ and temperature $T$:

$$
P_{\mathbf{\theta}}(\mathbf{x}) \propto e^{-E_{\mathbf{\theta}}(\mathbf{x})/T}
$$

* **NQS (Quantum Probability Amplitude):** Model the **wavefunction amplitude** $\psi_{\mathbf{\theta}}(\mathbf{x})$, which is often defined as an exponential function (e.g., in the RBM ansatz):

$$
\psi_{\mathbf{\theta}}(\mathbf{x}) \propto e^{-E_{\mathbf{\theta}}(\mathbf{x})/2}
$$

    *(Note: This form simplifies the relationship, where $E_{\mathbf{\theta}}(\mathbf{x})$ now includes the complex components needed for phase.)*

### **Difference: Amplitude vs. Probability**

-----

The critical difference lies in the nature of the output:

* **EBMs output Probability ($P$):** The result is a positive, real number. This governs classical statistics.
* **NQS output Amplitude ($\psi$):** The result is a **complex number** that includes both magnitude and **phase** (Section 17.6).

This complex phase allows for **quantum interference**. While the classical probability $P(\mathbf{x})$ simply adds, quantum probability amplitudes can cancel out or reinforce each other (destructive or constructive interference). This difference is essential for modeling the unique phenomena of quantum mechanics.

### **Analogy: NQS as Quantum EBMs**

-----

The relationship is summarized by viewing NQS as **quantum generalizations of EBMs**:

* **NQS $\leftrightarrow$ Quantum EBMs:** The energy function $E_{\mathbf{\theta}}$ in NQS is now the **complex energy functional** (related to action).
* **Quantum Statistical Mechanics:** The principles of training (minimizing expected energy via VMC, Section 17.4) are derived from the same variational basis as classical energy-based learning, but adapted to the domain of complex probability.

NQS fundamentally extends the powerful EBM idea to the domain where **probability becomes amplitude**.

??? question "Why Do Quantum Amplitudes Require Complex Numbers While Classical Probabilities Don't?"
    Classical probabilities are always positive real numbers that add directly. Quantum mechanics, however, requires **interference**—the ability for probability amplitudes to cancel (destructive interference) or reinforce (constructive interference). This interference is impossible with real numbers alone. Complex numbers provide the necessary phase $e^{i\Phi}$ that allows amplitudes to point in different "directions" in the complex plane. When computing probabilities $|\psi|^2$, these phases interact: $|\psi_1 + \psi_2|^2 \neq |\psi_1|^2 + |\psi_2|^2$. This phase-dependent interference is the signature of quantum mechanics and requires the wavefunction to be fundamentally complex-valued.

---

## **17.9 Stochastic Reconfiguration — Quantum Natural Gradient**

The training of a **Neural Quantum State (NQS)** (Section 17.2) by minimizing the expected energy $E[\psi_{\mathbf{\theta}}]$ (Section 17.2) often suffers from severe instability when using standard **gradient descent**. This issue arises because the system is constrained to move on the highly non-Euclidean manifold of normalized quantum states. **Stochastic Reconfiguration (SR)**, which is the quantum analogue of the **Natural Gradient**, solves this by adapting the optimization direction to the geometry of this quantum manifold.
### **The Problem: Unstable Gradient Descent**

-----

When performing gradient descent on the energy functional, $\nabla_{\mathbf{\theta}} E$, the optimization step $\Delta \mathbf{\theta} \propto -\nabla_{\mathbf{\theta}} E$ is defined using a simple **Euclidean metric** in the parameter space.

However, the energy is defined over the Hilbert space of wavefunctions. Movement in the parameter space that looks small in Euclidean terms can lead to a massive, disruptive change in the actual quantum state, causing **unstable and slow convergence**. This instability is exacerbated because the wavefunction $\psi_{\mathbf{\theta}}$ must always remain **normalized** ($\langle \psi_{\mathbf{\theta}} | \psi_{\mathbf{\theta}} \rangle = 1$).

### **The Solution: Hilbert Space Metric**

-----

Stochastic Reconfiguration (SR) corrects this by preconditioning the standard gradient with a metric tensor derived from the quantum state's geometry. The optimization step $\Delta \mathbf{\theta}$ is found by solving a linear system:

$$
S_{ij}\Delta\theta_j = -\nabla_i E
$$

* **$\nabla_i E$:** The standard expected energy gradient.
* **$S_{ij}$ (Quantum Geometric Tensor):** This matrix, estimated through Monte Carlo sampling, defines the **metric of the Hilbert space**. It ensures that the calculated step $\Delta \mathbf{\theta}$ is the most efficient movement toward the energy minimum when distance is measured in terms of the quantum state manifold.

### **Analogy: Quantum Natural Gradient**

-----

* **Connection to Natural Gradient:** SR is the direct quantum extension of the **Natural Gradient Descent** technique used in information geometry. The Natural Gradient uses the Fisher Information Matrix ($I_{ij}$, Section 2.3) to precondition the descent, ensuring that movement is maximized in the space of statistically distinguishable models.
* **Quantum Metric:** In SR, the metric tensor $S_{ij}$ replaces $I_{ij}$. The gradient descent process now follows the **geodesic motion** (shortest, most efficient path) on the manifold of normalized quantum states.

This adaptation allows the NQS optimizer to converge quickly and stably, effectively modeling the natural **relaxation dynamics** of the quantum system in its abstract Hilbert space.

--- 

## **17.10 Code Demo — Minimal Neural Quantum State**

This code demonstration provides a simplified, schematic implementation of a **Neural Quantum State (NQS)**, using a minimal **Restricted Boltzmann Machine (RBM)** structure (Section 17.3). The purpose is to show how a neural network's forward pass can output the **quantum amplitude** $\psi_{\mathbf{\theta}}(\mathbf{s})$ for a given spin configuration $\mathbf{s}$.

-----

```python
import torch, torch.nn as nn

class NQS(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        # W: Coupling matrix between visible (spins) and hidden (entanglement) units.
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden)*0.1)
        # a, b: Bias fields for visible and hidden units, respectively.
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))
        
    def forward(self, s):
        # Implements the RBM wavefunction ansatz (Section 17.3):
        # psi(s) = sum_h exp(a^T s + b^T h + s^T W h)
        
        # 1. Linear term: a^T s
        linear_term = self.a @ s 
        
        # 2. Coupling term: b + s @ W 
        # This is the contribution to the hidden unit energies.
        coupling_term = self.b + s @ self.W
        
        # 3. Summation over hidden units (logsumexp handles the sum efficiently):
        # The exponential of the RBM energy involves a sum over all possible hidden states h.
        log_sum_exp = torch.logsumexp(coupling_term, dim=1)
        
        # 4. Final log-amplitude (unnormalized log|psi| or log psi)
        return torch.exp(linear_term + log_sum_exp)

# Example energy estimation (schematic)
# Define 100 sample spin configurations (s) for a system of 10 spins.
s = torch.randint(0,2,(100,10)).float()*2-1
psi = NQS(10,5) # N=10 spins, 5 hidden units
log_psi = psi(s)
# Gradient-based updates on <H> would follow with sampled local energies
```

### **Interpretation of the Neural Ansatz**

-----

1.  **Architecture:** The `NQS` class defines the neural network structure that acts as the **variational ansatz** for the wavefunction $\psi_{\mathbf{\theta}}$. It uses the parameters $W$, $a$, and $b$ (collectively $\mathbf{\theta}$) as the adjustable coefficients of the quantum state.
2.  **Input/Output:** The input $\mathbf{s}$ is the **spin configuration**. The output `psi(s)` is the **complex amplitude** $\psi_{\mathbf{\theta}}(\mathbf{s})$ for that specific configuration (simplified here to a real-valued amplitude for demonstration).
3.  **Entanglement Encoding:** The core of the RBM ansatz (the logic within the `logsumexp`) performs the complex summation over the hidden unit states. These hidden units model the system's **entanglement**, ensuring the network can encode the intricate quantum correlations necessary to represent the true ground state.

The entire mechanism shows how the classical function approximation power of a neural network is harnessed to encode the exponential complexity of a quantum state. The next steps in a real NQS routine would involve using **Variational Monte Carlo (VMC)** (Section 17.4) to compute the expected energy and derive the energy gradient for training.

---

## **17.11 Deep Quantum Architectures**

The development of **Neural Quantum States (NQS)** (Section 17.1) has moved beyond the simple, shallow **Restricted Boltzmann Machine (RBM)** ansatz (Section 17.3) toward more sophisticated, deep architectures. These specialized designs are necessary to efficiently capture the diverse range of local and global correlations (entanglement) found in complex quantum systems.

-----

### **Hierarchical Architectures for Entanglement**

-----

Just as hierarchical networks efficiently model complexity in classical data (Chapter 13), deep quantum architectures impose necessary **inductive biases** that align the network's structure with the system's physics:

* **Autoregressive NQS:** These networks model the wavefunction amplitude $\psi(\mathbf{s})$ sequentially, ensuring that the total probability is always correctly normalized. This architecture is well-suited for efficient and exact **sampling** from the distribution $P(\mathbf{s}) = |\psi(\mathbf{s})|^2$.
* **CNN-based NQS:** Networks built with **Convolutional Neural Network (CNN)** layers are designed to capture the **translational invariance** inherent in regular lattice systems (e.g., crystal lattices). The shared weights enforce the physical symmetry of the environment (Chapter 13.3).
* **Transformer-NQS:** These leverage the **Self-Attention** mechanism (Chapter 19) to model **long-range quantum correlations**. They can dynamically capture complex, nonlocal entanglement patterns without the connectivity limitations of local lattice models.

!!! tip "Matching Architecture to Physics: The Power of Inductive Bias"

    The best neural quantum architecture is not necessarily the most complex, but the one whose **inductive bias** (built-in structural assumptions) aligns with the physical symmetries and correlations of the quantum system. For lattice systems with local interactions, CNNs exploit translational symmetry efficiently. For ordered phases with sequential dependencies, autoregressive models excel. For systems with long-range entanglement, Transformers provide the necessary global context. This principle mirrors the lesson from PINNs (Chapter 16): embedding physical structure into the model architecture dramatically reduces the parameters needed and accelerates convergence. The lesson: design your neural ansatz to mirror the quantum Hamiltonian's structure.

### **Analogy: Multi-Scale Quantum Interactions**

-----

Deep quantum architectures mirror the **multi-scale organization of quantum interactions**:
* **Local Interactions:** CNN layers efficiently handle short-range correlations between neighboring spins.
* **Global Entanglement:** Transformer layers capture the non-local influence of distant components.

The network's depth ($L$) defines the extent of the correlations it can effectively model.

### **Hybrid Quantum–Classical Models**

-----

A significant modern extension involves **Hybrid Quantum–Classical** architectures:
* **Variational Quantum Eigensolvers (VQE):** These use classical optimizers (Part II) to minimize the energy calculated from a **quantum circuit** (the variational ansatz). The classical optimization manages the complex energy landscape, while the quantum computer handles the necessary complex computation on the quantum state.

This approach demonstrates the practical power of integrating classical AI optimization with quantum hardware, with the final goal of making quantum simulation tractable.

---

## **17.12 Quantum Probability and Information Geometry**

The training of **Neural Quantum States (NQS)** (Section 17.2) is an optimization problem that takes place on the highly constrained space of quantum states. **Information Geometry** provides the advanced mathematical language for understanding the structure of this space, establishing a rigorous connection between probability, quantum mechanics, and optimization.

-----

### **The Manifold of Quantum States**

-----

The NQS network parameters $\mathbf{\theta}$ define a **manifold**—a curved geometric surface—embedded within the vast Hilbert space. Every point on this manifold corresponds to a valid, normalized wavefunction $\psi_{\mathbf{\theta}}$.

The challenge for optimization is to measure distance and define the "steepest descent" correctly on this curved manifold, rather than using the inadequate Euclidean distance of the parameter space.

### **The Quantum Fisher Information**

-----

The metric tensor that defines distance on the quantum state manifold is the **Quantum Fisher Information Matrix ($S_{ij}$)**. This matrix plays the same role as the Fisher Information Matrix in classical statistics (Chapter 2.3).

* **Definition:** $S_{ij}$ measures how quickly the quantum state $\psi_{\mathbf{\theta}}$ changes when the parameters $\mathbf{\theta}$ are varied. It quantifies the **statistical distinguishability** between two infinitesimally close quantum states.
* **Optimization Role:** The inverse of this matrix, $S^{-1}$, is used in **Stochastic Reconfiguration (SR)** (Section 17.9) to precondition the energy gradient. This ensures the optimization moves along the geodesic (shortest, most efficient path) of the quantum manifold, stabilizing convergence.

### **Information-Theoretic View: Compression**

-----

The NQS process can be interpreted from an information-theoretic perspective:

* **Goal:** The true quantum state $\psi$ is an object of exponential complexity. The learning process $\psi_{\mathbf{\theta}} \to \psi$ is equivalent to **projecting the true quantum state** onto the lower-dimensional manifold spanned by the neural network's parameters.
* **Analogy:** Quantum learning is a highly efficient form of **information compression**. The network uses its finite capacity (polynomial number of weights) to encode the maximum possible entanglement structure (correlation) needed for the ground state.

The geometry of the quantum state space thus dictates the optimal learning dynamics.

---

## **17.13 Applications**

The capacity of **Neural Quantum States (NQS)** to compress and represent complex wavefunctions (Section 17.7) makes them a transformative tool for solving intractable problems across the quantum sciences. By turning the search for the quantum ground state into a manageable **optimization problem** (Section 17.2), NQS provide a scalable bridge between classical deep learning and quantum many-body theory.

-----

### **Impact: Scalable Quantum Inference**

-----

The primary impact of NQS is to provide an efficient, polynomial-scaling alternative to traditional methods that struggle with the **exponential complexity** of the Hilbert space (Section 17.1).

| Domain | Hamiltonian / System | Purpose |
| :--- | :--- | :--- |
| **Quantum Spin Systems** | Ising, Heisenberg models (Section 17.5). | **Ground State Estimation:** Finding the lowest energy configuration of interacting spins in disordered systems, crucial for understanding quantum criticality and phase transitions. |
| **Electronic Structure** | Hubbard, H$_2$ molecule. | **Correlated Electron Wavefunctions:** Modeling the complex behavior of interacting electrons in materials, which dictates conductivity and magnetic properties. |
| **Quantum Chemistry** | Molecular orbitals. | **Energy and Density Prediction:** Calculating the total energy and electron density of molecular configurations with chemical accuracy. |
| **Quantum Dynamics** | Time-dependent Schrödinger Equation. | **Simulation via Time-Evolving PINNs:** NQS can be combined with PINN techniques (Chapter 16) to model the evolution of the quantum state over time. |
| **Quantum Hardware** | Variational Quantum Eigensolvers (VQE). | **Hybrid Classical–Quantum Training:** The NQS serves as the classical optimizer that efficiently guides the quantum computation performed on a quantum computer. |

!!! example "Real-World Application: Solving the Hubbard Model for High-Temperature Superconductivity"

    The Hubbard model describes interacting electrons on a lattice and is central to understanding high-temperature superconductors. Traditional methods (exact diagonalization, DMRG) fail for large 2D lattices due to exponential complexity. NQS with CNN architecture (Section 17.11) has successfully computed ground state energies for 10×10 lattices with accuracy matching or exceeding tensor network methods. The CNN's translational symmetry naturally captures the lattice periodicity, while hidden units encode electron correlations (entanglement). This breakthrough demonstrates NQS can tackle problems at the frontier of condensed matter physics, providing insights into the mechanism of superconductivity that could guide material design.

The **Unifying Impact** is that NQS transforms the calculation of high-dimensional quantum observables into a problem of **statistical inference** (Variational Monte Carlo, Section 17.4), which can be scaled effectively on classical hardware.

---

## **17.14 Physical Interpretation — Learning as Quantum Relaxation**

The convergence of the **Neural Quantum State (NQS)** training process (Section 17.2) is not merely a mathematical optimization; it is a computational simulation of a fundamental physical dynamic: **quantum relaxation**. This interpretation unifies machine learning and quantum physics under a common energetic geometry.

-----

### **Analogy: Gradient Descent in Energy**

-----

The optimization of the NQS parameters $\mathbf{\theta}$ using **Variational Monte Carlo (VMC)** (Section 17.4) is analogous to driving the quantum system toward its lowest energy configuration.

* **Classical Relaxation:** In classical physics, an excited particle in a fluid undergoes **overdamped relaxation** (Chapter 5.1), minimizing its potential energy until it settles at a minimum.
* **Quantum Relaxation:** The NQS training process mimics the **imaginary-time Schrödinger evolution**. This theoretical evolution is the quantum equivalent of classical relaxation; it is a flow that is guaranteed to minimize the expected energy and drive the trial wavefunction $\psi_{\mathbf{\theta}}$ toward the true ground state $\psi_0$.
* **Unified Dynamics:** In both classical and quantum contexts, the dynamics involve performing **gradient descent in energy**. The parameter updates are controlled by a "force" (the energy gradient) that pushes the system toward lower potential.

### **The Abstract Hilbert Manifold**

-----

The optimization does not take place in the physical 3D space of the atoms, but in the abstract, high-dimensional **Hilbert manifold** of possible quantum states (Section 17.12).

* **Parameter Updates $\leftrightarrow$ Relaxation:** The iterative adjustment of the network parameters $\mathbf{\theta}$ is analogous to the quantum state relaxing within this abstract space.
* **Energy and Geometry:** The stability of this motion is governed by the specialized metric (the Quantum Fisher Information, Section 17.9), ensuring the network finds the most efficient path (geodesic) to the energy minimum.

### **Unification: A Common Energetic Geometry**

-----

The NQS framework reveals a deep **unification**:

* **Machine learning and quantum physics share a common geometry**.
* The fundamental currency is the minimization of **expected energy** (or loss). The learning algorithm is simply a computational tool that efficiently solves the universal physics problem of finding the lowest energy state.

---

## **17.15 Bridge to Quantum Information and Beyond**

The framework of **Neural Quantum States (NQS)** establishes that classical deep learning models can act as efficient variational ansätze for simulating complex quantum many-body physics. This success opens the door to the broader field of **Quantum Information Theory** and sets the stage for the final frontiers of the Physics $\leftrightarrow$ AI synthesis.

-----

### **Next Steps in Quantum Modeling**

-----

The NQS approach naturally extends to more complex quantum objects and phenomena, driving research into deeper areas of the field:

* **Quantum Boltzmann Machines (QBMs):** These are the quantum generalization of the Restricted Boltzmann Machine (RBM) (Section 17.3, 17.8). They are used to represent **mixed quantum states** (density matrices, which encode thermal or disordered states) rather than just pure ground states (wavefunctions).
* **Learning Entanglement:** NQS and related architectures are being developed to quantify and learn key informational properties of quantum systems, such as **entanglement entropy** and **mutual information** between subsystems. These are essential measures of quantum correlation.
* **Neural Quantum Dynamics:** This involves using the network to model the **time evolution** of the quantum state. By training the NQS against the time-dependent Schrödinger equation (Chapter 16.11), the network learns the trajectory of the wavefunction through Hilbert space.

### **Foreshadowing: Graph-Based Quantum Fields**

-----

The challenge of modeling quantum systems involves both the non-local nature of entanglement (Chapter 19) and the local interaction structure imposed by the physical lattice.

* The next conceptual step in the architecture is the **Graph Neural Network (GNN)** (Chapter 18). While NQS can use GNNs internally (Section 17.11), the GNN framework itself models how local information propagates across a discrete, arbitrary graph.
* The connection: **Quantum states on lattices or arbitrary graphs** (chemical molecules, condensed matter lattices) are naturally described as **graph-based functions**. NQS with GNN layers represents the integration of two frontiers—quantum simulation and geometric deep learning.

This conceptual evolution shows the broader trajectory of modern AI research: from image recognition (CNNs) → sequential data (Transformers, Chapter 19) → graph-structured data (GNNs) → and finally to **quantum physics as graph learning**.

### **The Final Insight: Physics Is the Language of Learning**

-----

The NQS framework completes the circle:

1.  **Classical ML** (Chapters 1–11) begins with the language of **optimization**: energy landscapes, gradient descent, and loss functions.
2.  **Physics-Informed AI** (Chapters 12–17) reveals that the **laws of physics themselves are optimization problems**: PDEs are energy minimization (Chapter 16), quantum states are variational principles (this chapter).
3.  **The Synthesis**: Machine learning and physics are not separate fields—they are two perspectives on the same universal geometry of **information constrained by dynamics**.

The final lesson: Neural networks are not just tools for data fitting—they are **universal compressors** for the structure of reality itself.

---

## **References**

1. Carleo, G., & Troyer, M. (2017). Solving the quantum many-body problem with artificial neural networks. *Science*, 355(6325), 602-606.
2. Torlai, G., et al. (2018). Neural-network quantum state tomography. *Nature Physics*, 14(5), 447-450.
3. Choo, K., Carleo, G., Regnault, N., & Neupert, T. (2018). Symmetries and many-body excitations with neural-network quantum states. *Physical Review Letters*, 121(16), 167204.
4. Vicentini, F., et al. (2022). NetKet 3: Machine learning toolbox for many-body quantum systems. *SciPost Physics Codebases*, 7.
5. Glasser, I., et al. (2018). Neural-network quantum states, string-bond states, and chiral topological states. *Physical Review X*, 8(1), 011006.
6. Stokes, J., et al. (2020). Quantum natural gradient. *Quantum*, 4, 269.
7. Cai, Z., & Liu, J. (2018). Approximating quantum many-body wave functions using artificial neural networks. *Physical Review B*, 97(3), 035116.
8. Hermann, J., Schätzle, Z., & Noé, F. (2020). Deep-neural-network solution of the electronic Schrödinger equation. *Nature Chemistry*, 12(10), 891-897.
9. Pfau, D., et al. (2020). Ab initio solution of the many-electron Schrödinger equation with deep neural networks. *Physical Review Research*, 2(3), 033429.
10. Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251-276.
* This message-passing dynamics directly mirrors the flow of **quantum or physical interactions** on arbitrary geometries. GNNs, therefore, serve as the **discrete counterpart of continuous field theories** and are a natural architecture for enforcing local conservation laws in quantum systems.

The progression continues: from classical optimization to **quantum amplitude mechanics**, and then to the generalized interaction rules of arbitrary graph topology.
