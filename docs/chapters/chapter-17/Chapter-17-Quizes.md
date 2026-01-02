# **Chapter 17 : Quizes**

---

!!! note "Quiz"
    **1. What is the primary motivation for using Neural Quantum States (NQS) to solve the quantum many-body problem?**

    - A. To increase the speed of classical computers.
    - B. To compress the exponentially large Hilbert space of a quantum state into a neural network with a tractable (polynomial) number of parameters.
    - C. To eliminate the need for a Hamiltonian operator.
    - D. To directly measure the quantum state without collapse.

    ??? info "See Answer"
        **Correct: B**

        *(The core challenge of quantum many-body physics is the "curse of dimensionality," where storing the $2^N$ complex amplitudes for $N$ spins is impossible. NQS offers a compact representation.)*

---

!!! note "Quiz"
    **2. What fundamental principle of quantum mechanics serves as the primary learning objective for training an NQS?**

    - A. The Heisenberg Uncertainty Principle.
    - B. The Pauli Exclusion Principle.
    - C. The Variational Principle.
    - D. The principle of superposition.

    ??? info "See Answer"
        **Correct: C**

        *(This principle states that the expected energy of any trial wavefunction is always an upper bound to the true ground state energy. Therefore, minimizing the expected energy of the NQS drives it toward the best possible approximation of the ground state.)*

---

!!! note "Quiz"
    **3. In the NQS framework, what is the relationship between the wavefunction $\psi(\mathbf{s})$ and the observable probability $P(\mathbf{s})$?**

    - A. $P(\mathbf{s}) = \psi(\mathbf{s})$
    - B. $P(\mathbf{s}) = |\psi(\mathbf{s})|^2$
    - C. $P(\mathbf{s}) = \text{Re}(\psi(\mathbf{s}))$
    - D. $P(\mathbf{s}) = \log(|\psi(\mathbf{s})|)$

    ??? info "See Answer"
        **Correct: B**

        *(The probability of observing a system in a specific configuration $\mathbf{s}$ is the squared magnitude of its complex wavefunction amplitude. This is a cornerstone of quantum mechanics known as the Born rule.)*

---

!!! note "Quiz"
    **4. What is the role of the Hamiltonian operator ($\hat{H}$) in the context of NQS training?**

    - A. It acts as the optimizer, like Adam or SGD.
    - B. It defines the architecture of the neural network.
    - C. It acts as the "loss operator," defining the energy landscape that the optimization process seeks to minimize.
    - D. It is a hyperparameter that needs to be tuned.

    ??? info "See Answer"
        **Correct: C**

        *(The expected energy, which is the loss function, is calculated as the expectation value of the Hamiltonian, $E = \langle \psi | \hat{H} | \psi \rangle$.)*

---

!!! note "Quiz"
    **5. What is Variational Monte Carlo (VMC)?**

    - A. A specific type of neural network architecture.
    - B. A method for exact diagonalization of the Hamiltonian.
    - C. A stochastic method for estimating the expected energy by sampling configurations from $|\psi_{\theta}(\mathbf{s})|^2$ and averaging their local energies.
    - D. A technique for regularizing the neural network weights.

    ??? info "See Answer"
        **Correct: C**

        *(Since integrating over the entire Hilbert space is intractable, VMC provides a statistical estimate of the energy, which is then used to compute the gradient for optimization.)*

---

!!! note "Quiz"
    **6. In the Restricted Boltzmann Machine (RBM) ansatz for an NQS, what is the physical interpretation of the hidden units?**

    - A. They represent the external magnetic field.
    - B. They represent the temperature of the system.
    - C. They are latent variables that model the non-local correlations and entanglement between the visible physical spins.
    - D. They have no physical interpretation and are purely for regularization.

    ??? info "See Answer"
        **Correct: C**

        *(Summing over the hidden units introduces the complex, all-to-all couplings necessary to represent an entangled quantum state, which cannot be captured by a simple product state.)*

---

!!! note "Quiz"
    **7. Why are complex-valued neural networks necessary for accurately representing a general quantum state?**

    - A. Because the input spin configurations are complex numbers.
    - B. To make the optimization process faster.
    - C. Because the quantum wavefunction has both a magnitude (amplitude) and a phase, and the phase is crucial for modeling quantum interference.
    - D. They are not necessary; real-valued networks are sufficient for all quantum states.

    ??? info "See Answer"
        **Correct: C**

        *(A real-valued network can only represent the amplitude, missing the critical phase information that governs quantum dynamics and interference effects.)*

---

!!! note "Quiz"
    **8. What is the "local energy," $E_{\text{loc}}(\mathbf{s})$?**

    - A. The average energy of the entire system.
    - B. The energy contribution from a single spin.
    - C. The value $(\hat{H}\psi_{\theta})(\mathbf{s})/\psi_{\theta}(\mathbf{s})$, which is the energy of a specific configuration $\mathbf{s}$ under the current trial wavefunction.
    - D. The kinetic energy of the particles.

    ??? info "See Answer"
        **Correct: C**

        *(The total expected energy is the statistical average of the local energies over all sampled configurations.)*

---

!!! note "Quiz"
    **9. What is Stochastic Reconfiguration (SR), and why is it used in NQS training?**

    - A. A method for data augmentation.
    - B. It is the quantum analogue of the Natural Gradient, which preconditions the energy gradient with the Quantum Fisher Information matrix to ensure stable optimization on the curved manifold of quantum states.
    - C. A technique for reducing the number of hidden units in an RBM.
    - D. A way to initialize the network weights.

    ??? info "See Answer"
        **Correct: B**

        *(Standard gradient descent is unstable because the parameter space is not Euclidean. SR adapts the gradient to the geometry of the Hilbert space.)*

---

!!! note "Quiz"
    **10. How does an NQS compare to a Tensor Network (TN) in terms of representing entanglement?**

    - A. NQS and TNs have identical entanglement capacity.
    - B. TNs are better at representing high, non-local entanglement, while NQS are limited to local entanglement.
    - C. NQS are generally more flexible and can represent complex, global entanglement, whereas TNs are most efficient for systems with limited, structured (area-law) entanglement.
    - D. Neither can represent entanglement.

    ??? info "See Answer"
        **Correct: C**

        *(NQS are seen as global generalizers, while TNs are local experts.)*

---

!!! note "Quiz"
    **11. The training process of an NQS is physically analogous to what phenomenon?**

    - A. A chemical reaction reaching equilibrium.
    - B. A particle undergoing Brownian motion.
    - C. Quantum relaxation, specifically imaginary-time Schrödinger evolution, where the system is driven toward its lowest energy ground state.
    - D. The expansion of the universe.

    ??? info "See Answer"
        **Correct: C**

        *(The gradient descent on the energy landscape is a computational simulation of the physical process of a system losing energy and settling into its ground state.)*

---

!!! note "Quiz"
    **12. What is the key difference between a classical Energy-Based Model (EBM) and an NQS?**

    - A. EBMs are generative, while NQS are discriminative.
    - B. EBMs model a real-valued probability distribution, while NQS model a complex-valued probability amplitude that includes a phase.
    - C. EBMs use a Hamiltonian, while NQS use a Lagrangian.
    - D. There is no fundamental difference.

    ??? info "See Answer"
        **Correct: B**

        *(This makes NQS a "quantum generalization" of EBMs, with the complex phase enabling the modeling of quantum interference.)*

---

!!! note "Quiz"
    **13. In the context of the Transverse-Field Ising Model (TFIM), what two competing effects does the Hamiltonian describe?**

    - A. Ferromagnetism and gravity.
    - B. An ordering force that aligns spins and a transverse field that flips them, causing disorder.
    - C. Kinetic energy and potential energy.
    - D. Particle creation and annihilation.

    ??? info "See Answer"
        **Correct: B**

        *(The competition between the Ising interaction term ($J$) and the transverse field term ($h$) leads to a quantum phase transition.)*

---

!!! note "Quiz"
    **14. What is the advantage of using a CNN-based NQS for a quantum system on a regular lattice?**

    - A. CNNs can handle complex-valued inputs automatically.
    - B. The inductive bias of a CNN (translational invariance) naturally matches the physical symmetry of the lattice, making the representation more efficient.
    - C. CNNs are the only architecture that can be trained with VMC.
    - D. CNNs have fewer parameters than RBMs.

    ??? info "See Answer"
        **Correct: B**

        *(By building the physical symmetries into the architecture, the network can learn the ground state with fewer parameters and better generalization.)*

---

!!! note "Quiz"
    **15. The gradient of the expected energy, $\nabla_{\theta} E$, is proportional to $\langle (E_{\text{loc}} - E) \nabla_{\theta} \ln \psi_{\theta} \rangle$. What is the role of the $(E_{\text{loc}} - E)$ term?**

    - A. It acts as a learning rate.
    - B. It is a normalization constant.
    - C. It acts as an error signal, similar to a TD error, indicating whether a sampled configuration has a higher or lower energy than the current average.
    - D. It is always zero.

    ??? info "See Answer"
        **Correct: C**

        *(This term "rewards" or "penalizes" the log-probability gradient, pushing the wavefunction's amplitude away from high-energy configurations and toward low-energy ones.)*

---

!!! note "Quiz"
    **16. What is a Variational Quantum Eigensolver (VQE)?**

    - A. A fully quantum algorithm for finding eigenvalues.
    - B. A hybrid quantum-classical algorithm where a classical optimizer trains the parameters of a quantum circuit (the ansatz) to find the ground state energy.
    - C. A type of tensor network.
    - D. A classical simulation of a quantum computer.

    ??? info "See Answer"
        **Correct: B**

        *(This is a key application of variational methods on near-term quantum hardware, where the NQS acts as the classical optimization part.)*

---

!!! note "Quiz"
    **17. What does the Quantum Fisher Information Matrix ($S_{ij}$) measure?**

    - A. The entanglement entropy of the state.
    - B. The speed of the quantum computation.
    - C. The statistical distinguishability between two infinitesimally close quantum states, defining a metric on the manifold of quantum states.
    - D. The number of qubits required for a simulation.

    ??? info "See Answer"
        **Correct: C**

        *(It is the quantum analogue of the classical Fisher Information Matrix and is used to define the Natural Gradient for stable optimization.)*

---

!!! note "Quiz"
    **18. In the code demo for the NQS gradient, why is `tf.GradientTape` used?**

    - A. To record the loss history for plotting.
    - B. To measure the execution time of the code.
    - C. To perform automatic differentiation for calculating the gradient of the log-wavefunction with respect to the trainable parameters ($\nabla_{\theta} \ln \psi$).
    - D. To tape the quantum state to a classical memory register.

    ??? info "See Answer"
        **Correct: C**

        *(This is the "feature vector" part of the energy gradient formula and requires AD for its computation.)*

---

!!! note "Quiz"
    **19. What is a key limitation of a simple product state ansatz like $\psi(\mathbf{s}) = \prod_i \psi(s_i)$?**

    - A. It can only be used for 1D systems.
    - B. It is computationally very expensive.
    - C. It cannot represent entangled states because it assumes the state of each particle is independent of the others.
    - D. It can only represent the ground state, not excited states.

    ??? info "See Answer"
        **Correct: C**

        *(The project on entanglement capacity demonstrates that such an ansatz has very low fidelity with a target entangled state, proving the need for more expressive models like RBMs.)*

---

!!! note "Quiz"
    **20. The conceptual project on "Energy Dissipation Check" aims to verify what property of the VMC optimization?**

    - A. That the energy of the system is conserved.
    - B. That the energy of the system fluctuates randomly.
    - C. That the expected energy monotonically decreases over training epochs, simulating physical relaxation to a ground state.
    - D. That the learning rate remains constant.

    ??? info "See Answer"
        **Correct: C**

        *(This check confirms that the optimization is stable and is correctly finding a minimum on the energy landscape.)*

---

!!! note "Quiz"
    **21. What is the output of a forward pass of an RBM-based NQS for a given spin configuration `s`?**

    - A. The probability of that configuration.
    - B. The local energy of that configuration.
    - C. The complex probability amplitude, $\psi_{\theta}(\mathbf{s})$.
    - D. The gradient of the energy.

    ??? info "See Answer"
        **Correct: C**

        *(The network's fundamental role is to act as a function approximator for the wavefunction itself. All other quantities (probability, energy) are derived from this output.)*

---

!!! note "Quiz"
    **22. How does the NQS framework connect to Graph Neural Networks (GNNs)?**

    - A. GNNs are used to visualize the Hilbert space.
    - B. Quantum systems with specific interaction geometries (like molecules or crystal lattices) can be modeled as graphs, making GNNs a natural architectural choice for the NQS ansatz.
    - C. GNNs are used as the classical optimizer instead of gradient descent.
    - D. There is no connection between NQS and GNNs.

    ??? info "See Answer"
        **Correct: B**

        *(The message-passing mechanism of a GNN can directly model the local interactions defined by the system's Hamiltonian on a graph.)*

---

!!! note "Quiz"
    **23. To handle complex numbers, a neural network can be implemented with coupled real and imaginary channels. What mathematical tool is needed to extend backpropagation to this architecture?**

    - A. Fourier Transforms.
    - B. Laplace Transforms.
    - C. Wirtinger derivatives.
    - D. Finite difference approximations.

    ??? info "See Answer"
        **Correct: C**

        *(This calculus extends the concept of gradients to functions of complex variables, allowing for the application of gradient-based optimization to complex-valued neural networks.)*

---

!!! note "Quiz"
    **24. In the VMC process, how are the sample configurations $\mathbf{s}_i$ generated?**

    - A. They are drawn from a uniform random distribution.
    - B. They are generated using a classical Markov Chain Monte Carlo (MCMC) method, like the Metropolis algorithm, to sample from the distribution $P(\mathbf{s}) = |\psi_{\theta}(\mathbf{s})|^2$.
    - C. All $2^N$ possible states are enumerated and used.
    - D. They are provided as a fixed dataset before training begins.

    ??? info "See Answer"
        **Correct: B**

        *(This is crucial for efficiently sampling the most probable configurations, which contribute most to the expectation value.)*

---

!!! note "Quiz"
    **25. What is the ultimate synthesis or "final insight" presented by the NQS framework?**

    - A. That quantum computers will soon replace all classical computers.
    - B. That machine learning and physics are two perspectives on the same universal geometry of information constrained by dynamics, where optimization is equivalent to energy minimization.
    - C. That all quantum systems can be solved with simple RBMs.
    - D. That physics-informed learning is only applicable to classical, not quantum, systems.

    ??? info "See Answer"
        **Correct: B**

        *(The NQS framework shows that the language of optimization and the laws of physics are deeply intertwined.)*
