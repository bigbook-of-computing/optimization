# **Chapter-8: Quizes**

---

!!! note "Quiz"
    **1. What is the fundamental reason that gradient-based optimizers are unsuitable for combinatorial optimization problems?**

    - A. The search space is too small.
    - B. The loss function is always convex.
    - C. The variables are discrete, meaning the landscape consists of isolated points with no defined gradient.
    - D. They require too much memory.

    ??? info "See Answer"
        **Correct: C**

        *(Combinatorial problems have discrete variables (e.g., 0/1), so the concept of an infinitesimal change needed to define a gradient does not exist. The landscape is not a smooth surface.)*

---

!!! note "Quiz"
    **2. The total number of possible configurations in a system with $N$ binary variables is $2^N$. This exponential growth is often referred to as:**

    - A. The manifold hypothesis.
    - B. The curse of dimensionality.
    - C. Moore's Law.
    - D. The holographic principle.

    ??? info "See Answer"
        **Correct: B**

        *(While often used for continuous spaces, the term also applies to the exponential explosion of the state space in discrete problems, making brute-force search intractable.)*

---

!!! note "Quiz"
    **3. The **Quadratic Unconstrained Binary Optimization (QUBO)** model represents the cost function as a quadratic polynomial of binary variables. What does the quadratic term $\sum_{i<j} b_{ij} x_i x_j$ represent?**

    - A. The individual cost of selecting each item.
    - B. The total number of selected items.
    - C. The interaction cost or benefit between pairs of variables.
    - D. A constant energy offset.

    ??? info "See Answer"
        **Correct: C**

        *(The quadratic term captures the pairwise relationships, which is analogous to the interaction energy between spins in the Ising model.)*

---

!!! note "Quiz"
    **4. The QUBO formalism is mathematically equivalent to which fundamental model from statistical physics?**

    - A. The Schrödinger Equation.
    - B. The Navier-Stokes Equations.
    - C. The Ising Model.
    - D. The Black-Scholes Model.

    ??? info "See Answer"
        **Correct: C**

        *(The QUBO cost function has a one-to-one mapping with the Ising Hamiltonian, allowing any combinatorial problem to be framed as finding the ground state of a spin system.)*

---

!!! note "Quiz"
    **5. What is the linear transformation used to convert a QUBO binary variable $x_i \in \{0, 1\}$ into an Ising spin variable $s_i \in \{-1, +1\}$?**

    - A. $s_i = x_i - 1$
    - B. $s_i = 2x_i$
    - C. $s_i = 2x_i - 1$
    - D. $s_i = (x_i - 0.5) / 2$

    ??? info "See Answer"
        **Correct: C**

        *(This simple transformation maps the "off" state (0) to spin down (-1) and the "on" state (1) to spin up (+1).)*

---

!!! note "Quiz"
    **6. In the Ising model, $E(\mathbf{s}) = -\sum_{i<j} J_{ij} s_i s_j - \sum_i h_i s_i$, what does a positive coupling constant ($J_{ij} > 0$) encourage?**

    - A. It encourages spins $s_i$ and $s_j$ to be anti-aligned (one up, one down).
    - B. It encourages spins $s_i$ and $s_j$ to be aligned (both up or both down).
    - C. It has no effect on the spin alignment.
    - D. It forces both spins to be zero.

    ??? info "See Answer"
        **Correct: B**

        *(A positive $J_{ij}$ (ferromagnetic coupling) means the energy is minimized when $s_i s_j = +1$, which occurs when the spins are aligned.)*

---

!!! note "Quiz"
    **7. How are strict rules, or "constraints," incorporated into the "Unconstrained" QUBO model?**

    - A. They are solved for separately after the main optimization.
    - B. They are added to the objective function as large energy penalty terms that are zero only when the constraint is satisfied.
    - C. They are ignored, as the model is unconstrained.
    - D. They are used to reduce the number of variables.

    ??? info "See Answer"
        **Correct: B**

        *(This technique converts a constrained problem into an unconstrained one by creating massive energy walls that guide the solver away from infeasible regions of the search space.)*

---

!!! note "Quiz"
    **8. For a "one-hot" constraint like "select exactly one item from a list," the penalty term is often $\lambda (\sum_i x_i - 1)^2$. What is the purpose of the large coefficient $\lambda$?**

    - A. To normalize the energy.
    - B. To act as a learning rate.
    - C. To act as a Lagrange multiplier or penalty strength, ensuring any violation incurs a massive energy cost.
    - D. To set the temperature for simulated annealing.

    ??? info "See Answer"
        **Correct: C**

        *(The penalty strength $\lambda$ must be large enough to dominate the original objective cost, making it energetically unfavorable for the solver to ever violate the constraint.)*

---

!!! note "Quiz"
    **9. Which of the following is an **exact** method for solving QUBO problems, guaranteeing a globally optimal solution but scaling poorly?**

    - A. Simulated Annealing.
    - B. Genetic Algorithms.
    - C. Brute-force enumeration (exhaustive search).
    - D. Quantum Annealing.

    ??? info "See Answer"
        **Correct: C**

        *(Brute-force search checks every single one of the $2^N$ configurations, which guarantees finding the minimum but is only feasible for very small $N$.)*

---

!!! note "Quiz"
    **10. What is the primary physical mechanism that allows **Quantum Annealing (QA)** to find solutions to QUBO problems?**

    - A. Thermal hopping over energy barriers.
    - B. Gradient descent.
    - C. Quantum tunneling through energy barriers.
    - D. Evolutionary crossover and mutation.

    ??? info "See Answer"
        **Correct: C**

        *(QA replaces classical thermal fluctuations with quantum fluctuations, which allow the system to tunnel through barriers instead of having to climb over them, potentially offering a speed advantage for certain problems.)*

---

!!! note "Quiz"
    **11. In the QUBO formulation of the Traveling Salesman Problem (TSP), what does the binary variable $x_{i,t}$ represent?**

    - A. The distance between city $i$ and city $t$.
    - B. A value of 1 if city $i$ is visited at position $t$ in the tour, and 0 otherwise.
    - C. The total number of cities in the tour.
    - D. The temperature at time $t$.

    ??? info "See Answer"
        **Correct: B**

        *(This double-indexed variable creates an $N 	imes N$ grid of decisions, where a valid tour corresponds to a specific pattern of 1s on the grid.)*

---

!!! note "Quiz"
    **12. For the Graph Partitioning (Minimum Cut) problem, assigning spins $s_i \in \{-1, +1\}$ to nodes, the energy term $\sum w_{ij}(1 - s_i s_j)$ is minimized when:**

    - A. All spins are +1.
    - B. The cut weight is maximized.
    - C. Nodes $i$ and $j$ are in the same partition ($s_i = s_j$).
    - D. Nodes $i$ and $j$ are in different partitions ($s_i 
eq s_j$).

    ??? info "See Answer"
        **Correct: C**

        *(If nodes $i$ and $j$ are in the same partition, $s_i s_j = 1$, and the term becomes $w_{ij}(1-1)=0$. If they are in different partitions, the term is $2w_{ij}$, adding to the cost. Therefore, minimizing this energy minimizes the number of edges within the same partition, which is not the standard min-cut. The correct formulation for min-cut is to minimize edges *between* partitions, often written as $\sum w_{ij}(1-s_is_j)/2$. However, based on the provided options, C is the most direct interpretation of minimizing that specific term.)* 
        
        *Correction Note: A better energy function for min-cut is $E = -\sum w_{ij} s_i s_j$. The provided workbook form is slightly different, but the principle of using spin products to represent cuts remains.*

---

!!! note "Quiz"
    **13. How can the machine learning problem of **feature selection** be framed as a QUBO problem?**

    - A. By assigning a continuous weight to each feature.
    - B. By assigning a binary variable $x_i$ to each feature, where $x_i=1$ means the feature is selected, and the objective balances model accuracy and the number of selected features.
    - C. By using Principal Component Analysis.
    - D. By training a neural network with binary weights.

    ??? info "See Answer"
        **Correct: B**

        *(The objective function becomes a trade-off between a complex term for model error and a simple linear penalty $\sum x_i$ (the $L^0$ norm) that encourages sparsity.)*

---

!!! note "Quiz"
    **14. A "spin glass" is a disordered magnetic system often used to benchmark QUBO solvers. What property makes its ground state hard to find?**

    - A. All couplings $J_{ij}$ are positive and uniform.
    - B. It has a single, deep global minimum.
    - C. The couplings $J_{ij}$ are random and conflicting ("frustration"), creating a rugged landscape with many local minima.
    - D. It has no external magnetic field ($h_i=0$).

    ??? info "See Answer"
        **Correct: C**

        *(Frustration means that no single spin configuration can satisfy all the interaction bonds simultaneously, leading to a complex and rugged energy landscape that is very difficult to navigate.)*

---

!!! note "Quiz"
    **15. In the code demo for a simple QUBO solver, why is brute-force search used instead of a heuristic like Simulated Annealing?**

    - A. Brute-force is always faster.
    - B. The problem size ($N=6$) is small enough that all $2^6=64$ configurations can be checked exhaustively to guarantee the exact global minimum.
    - C. Simulated Annealing cannot solve QUBO problems.
    - D. The QUBO matrix Q was not symmetric.

    ??? info "See Answer"
        **Correct: B**

        *(For very small N, exhaustive search is feasible and provides a useful way to verify the correctness of the problem formulation and find the true ground state energy.)*

---

!!! note "Quiz"
    **16. The process used by Quantum Annealers, which involves slowly changing the Hamiltonian from a simple initial state to the final problem state, is based on what principle?**

    - A. The Heisenberg Uncertainty Principle.
    - B. The Adiabatic Theorem.
    - C. The Metropolis-Hastings Algorithm.
    - D. The No-Free-Lunch Theorem.

    ??? info "See Answer"
        **Correct: B**

        *(The Adiabatic Theorem states that if a quantum system starts in its ground state and its Hamiltonian is changed slowly enough, it will remain in the instantaneous ground state throughout the evolution.)*

---

!!! note "Quiz"
    **17. In the context of the QUBO matrix $Q$ for the Max-Cut problem, what do the diagonal elements $Q_{ii}$ typically represent?**

    - A. The weight of the edges.
    - B. A linear bias related to the degree of node $i$.
    - C. The number of nodes in the graph.
    - D. The penalty factor $\lambda$.

    ??? info "See Answer"
        **Correct: B**

        *(The diagonal elements $Q_{ii}$ correspond to the linear terms $a_i x_i$ in the QUBO expansion. For Max-Cut, this term is derived from the sum of weights of all edges connected to node $i$.)*

---

!!! note "Quiz"
    **18. An antiferromagnetic chain is a system where the energy is minimized when adjacent spins are anti-aligned. What would the coupling constant $J_{ij}$ be for adjacent spins?**

    - A. A large positive number.
    - B. A large negative number.
    - C. Zero.
    - D. A complex number.

    ??? info "See Answer"
        **Correct: B**

        *(In the Ising formulation $E = -\sum J_{ij}s_i s_j$, a negative $J_{ij}$ makes the energy contribution positive if spins are aligned ($s_i s_j=1$) and negative if they are anti-aligned ($s_i s_j=-1$). To minimize energy, the system will choose anti-alignment.)*

---

!!! note "Quiz"
    **19. What is the primary conceptual shift when moving from Part II (Optimization) to Part III (Inference) of the book?**

    - A. From discrete variables to continuous variables.
    - B. From finding a single optimal point (minimum energy) to characterizing an entire probability distribution.
    - C. From using Python to using C++.
    - D. From linear models to non-linear models.

    ??? info "See Answer"
        **Correct: B**

        *(Optimization seeks the single best solution $\mathbf{	heta}^*$. Inference, particularly Bayesian inference, seeks to understand the entire landscape of possibilities by finding the posterior probability distribution $P(\mathbf{	heta}|\mathcal{D})$.)*

---

!!! note "Quiz"
    **20. The duality $E(\mathbf{	heta}) \leftrightarrow -\ln P(\mathbf{	heta})$ connects energy and probability. A low-energy state corresponds to a:**

    - A. Low-probability state.
    - B. High-probability state.
    - C. Zero-probability state.
    - D. State with high uncertainty.

    ??? info "See Answer"
        **Correct: B**

        *(This is the core of the Boltzmann distribution, $P \propto e^{-E/T}$. The states with the lowest energy are exponentially the most probable.)*

---

!!! note "Quiz"
    **21. Why is the QUBO matrix $Q$ often made symmetric in implementations (e.g., `Q = (Q + Q.T) / 2`)?**

    - A. To make it invertible.
    - B. To ensure the energy function is well-defined, as the terms $Q_{ij}x_i x_j$ and $Q_{ji}x_j x_i$ are combined.
    - C. To reduce the number of variables.
    - D. To satisfy the one-hot constraint.

    ??? info "See Answer"
        **Correct: B**

        *(Since $x_i x_j = x_j x_i$, the contributions from $Q_{ij}$ and $Q_{ji}$ are indistinguishable. The solver only cares about their sum, $(Q_{ij} + Q_{ji})$. Symmetrizing the matrix is a standard convention that doesn't change the underlying energy function.)*

---

!!! note "Quiz"
    **22. In a hybrid classical-quantum approach to solving a large QUBO problem, what is the typical role of the quantum annealer?**

    - A. It solves the entire problem from start to finish.
    - B. It handles the user interface and data pre-processing.
    - C. It solves small, complex, core subproblems that have been decomposed from the larger problem.
    - D. It is used to verify the solution found by the classical computer.

    ??? info "See Answer"
        **Correct: C**

        *(Current quantum devices are limited in size and connectivity. Hybrid methods leverage them as powerful co-processors for the hardest parts of a problem, while a classical computer manages the overall workflow.)*

---

!!! note "Quiz"
    **23. What is the ground state energy of a simple 2-spin antiferromagnetic system ($s_1, s_2$) with Hamiltonian $E = -J s_1 s_2$ where $J=-1$?**

    - A. +1
    - B. 0
    - C. -1
    - D. -2

    ??? info "See Answer"
        **Correct: C**

        *(The Hamiltonian is $E = -(-1)s_1 s_2 = s_1 s_2$. To minimize energy, the spins must be anti-aligned, so $s_1 s_2 = -1$. Therefore, the minimum energy is -1.)*

---

!!! note "Quiz"
    **24. If you use the one-hot penalty function $E = 100(\sum_{i=1}^4 x_i - 1)^2$ and your solver returns a configuration of $[1, 1, 0, 0]$, what would the penalty energy be?**

    - A. 0
    - B. 100
    - C. 200
    - D. 400

    ??? info "See Answer"
        **Correct: B**

        *(The sum of variables is $\sum x_i = 1+1+0+0 = 2$. The penalty is $100 	imes (2 - 1)^2 = 100 	imes 1^2 = 100$.)*

---

!!! note "Quiz"
    **25. The universality of the QUBO/Ising formalism means that a scientist with a new combinatorial problem needs to focus their effort on what primary task?**

    - A. Building a new quantum computer.
    - B. Writing a new heuristic solver from scratch.
    - C. Translating the problem's logic and constraints into the corresponding QUBO matrix $Q$.
    - D. Proving that P=NP.

    ??? info "See Answer"
        **Correct: C**

        *(The main intellectual challenge is the formulation of the problem in the universal language of QUBO. Once the matrix Q is constructed, it can be handed off to a variety of existing, powerful solvers.)*


### 1.2 Representing Simulation Outputs

> **Summary:** Raw simulation data must be **flattened** (unrolled into a single vector) to form the standard data matrix $X$ ($M$ samples $\times$ $D$ features). All features must be placed on equal footing via **normalization** (typically Z-score standardization). The resulting dataset $X$ is an **empirical distribution**, and its shape is summarized by the **mean vector ($\boldsymbol{\mu}$)** and the **covariance matrix ($\Sigma$)**.

#### Quiz Questions

**1. In the context of data preparation, the process of taking a 2D spin lattice ($16 \times 16$) and converting it into a single vector of length 256 is called: **

* **A.** Standardization.
* **B.** **Flattening**. (**Correct**)
* **C.** Ensemble averaging.
* **D.** Eigendecomposition.

**2. The purpose of **standardization (Z-score normalization)** in data preparation is to:**

* **A.** Discard the continuous time variable.
* **B.** Convert the data from time averages to ensemble averages.
* **C.** **Center each feature at zero mean and scale it to unit variance, ensuring no feature numerically dominates the analysis**. (**Correct**)
* **D.** Compute the non-linear geodesic distance.

---

#### Interview-Style Question

**Question:** The **covariance matrix ($\Sigma$)** is described as the key to geometric analysis. What specific types of information does the off-diagonal element $\Sigma_{jk}$ encode about the relationship between features $j$ and $k$?

**Answer Strategy:** The off-diagonal element $\Sigma_{jk}$ of the covariance matrix encodes the **linear correlation** between feature $j$ and feature $k$.
* If $\Sigma_{jk} > 0$, the features are positively correlated (they tend to move up or down together).
* If $\Sigma_{jk} < 0$, the features are anti-correlated (one moves up, the other moves down).
* If $\Sigma_{jk} \approx 0$, the features are linearly independent (or uncorrelated). This reveals how different parts of the physical system (e.g., two different atoms or regions of a lattice) move together.

---
***

### 1.3 The Geometry of Variability

> **Summary:** The covariance matrix $\Sigma$ is a **geometric operator**. Its decomposition via the eigenvalue equation ($\Sigma \mathbf{v}_k = \lambda_k \mathbf{v}_k$) is the core of **Principal Component Analysis (PCA)**. The **eigenvectors ($\mathbf{v}_k$)** are the **principal axes** of the data cloud, and their corresponding **eigenvalues ($\lambda_k$)** are the *variance* along those axes. The dominant eigenvectors identify the physical system's **collective variables** or **order parameters**.

#### Quiz Questions

**1. In Principal Component Analysis (PCA), the **eigenvectors ($\mathbf{v}_k$)** of the covariance matrix represent the data's:**

* **A.** Mean position.
* **B.** Total energy.
* **C.** **Principal axes or directions of greatest variance**. (**Correct**)
* **D.** Shannon entropy.

**2. When analyzing molecular dynamics (MD) data, a physicist interprets the first principal component ($\mathbf{v}_1$) as the system's dominant mode of motion. This mode is described as a **collective variable** because it:**

* **A.** Only involves a single, isolated atom.
* **B.** **Represents a highly coordinated motion (e.g., hinging of two domains) across many features**. (**Correct**)
* **C.** Is guaranteed to be non-linear.
* **D.** Is exactly equal to the total magnetization $M$.

---

#### Interview-Style Question

**Question:** A physicist performs PCA on simulation data and finds that the first three eigenvalues ($\lambda_1, \lambda_2, \lambda_3$) account for 98% of the total variance, while the remaining $D-3$ eigenvalues are near zero. What does this result tell them about the data's dimensionality, and how does it relate to the **manifold hypothesis**?

**Answer Strategy:** This tells the physicist that the data's **intrinsic dimensionality is very low** ($d=3$), even though it lives in a high-dimensional feature space $\mathbb{R}^D$. The result directly supports the **manifold hypothesis**. It means the system's complex fluctuations are constrained to a low-dimensional surface ($\mathcal{M}$), and the first three principal axes ($\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$) form the linear coordinate system that best approximates that manifold.

---
***

### 1.4 Distance, Similarity, and Metrics

> **Summary:** The standard **Euclidean ($L^2$) distance** is often physically nonsensical in high dimensions (e.g., between a structure and its rotated copy). Physically relevant metrics must be **invariant** to symmetries, such as **Root Mean Square Deviation (RMSD)** for molecular structures. The **geodesic distance** is the physically meaningful path between points **while staying on the manifold** ($\mathcal{M}$), capturing energy barriers missed by the straight-line $L^2$ norm.

#### Quiz Questions

**1. The main physical drawback of using the standard **Euclidean ($L^2$) distance** to compare two molecular snapshots is that:**

* **A.** It is too slow to compute.
* **B.** **It loses information about the original 3D rotational and translational symmetries**. (**Correct**)
* **C.** It only works for uncorrelated data.
* **D.** It requires the prior to be Gaussian.

**2. The type of distance that measures the shortest path between two states *while accounting for the physical constraints (curved surface)* of the low-dimensional manifold $\mathcal{M}$ is called the:**

* **A.** Euclidean distance ($L^2$).
* **B.** Correlation distance.
* **C.** **Geodesic distance**. (**Correct**)
* **D.** Mahalanobis distance.

---

#### Interview-Style Question

**Question:** Two time-series signals from two different simulation runs show the exact same fluctuation pattern (e.g., the same oscillations and peaks) but one has a much larger amplitude. The Euclidean distance between them is large. Propose an alternative distance metric that would correctly identify them as highly *similar* and explain why it works.

**Answer Strategy:** The best alternative is a **correlation distance**, such as $d_C(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{2(1 - r_{ij})}$, based on the Pearson correlation coefficient ($r$).
* **Why it works:** The Pearson correlation coefficient measures the **shape of the linear relationship**, not the magnitude or offset of the signals. Since the pattern is the same (high correlation), $r$ would be close to $+1$, and the distance $d_C$ would be near $0$, correctly identifying the functional similarity despite the amplitude difference.

---
***

### 1.5 From Clouds to Structure

> **Summary:** The data cloud's shape is a direct map of the **free-energy landscape**. **Dense clusters** in the data correspond to **basins** (stable or metastable states) in the energy landscape, while **empty voids** correspond to **high-energy barriers**. The cloud's overall spread can be quantified by **Shannon entropy** ($S = -k_B \sum_i p_i \ln p_i$), linking geometric disorder to physical disorder.

#### Quiz Questions

**1. In the context of mapping a potential energy landscape from simulation data, a large, empty void in the high-dimensional data cloud corresponds to a(n):**

* **A.** Time-reversible trajectory.
* **B.** **High-energy barrier**. (**Correct**)
* **C.** Low-entropy, ordered state.
* **D.** Non-linear embedding.

**2. What is the physical interpretation of a data cloud that is highly concentrated in one small, dense region (low entropy)?**

* **A.** A high-temperature, disordered state.
* **B.** A fast transition path.
* **C.** **A low-temperature, ordered state**. (**Correct**)
* **D.** A complex chemical reaction.

---

#### Interview-Style Question

**Question:** If you are analyzing a molecular dynamics trajectory of a protein that is known to exist in two distinct stable states ("open" and "closed"), what characteristic shape and topology would you expect to see when plotting the data onto its first two principal components?

**Answer Strategy:** I would expect to see a plot with **two distinct, dense clusters** of data points.
* Each cluster represents one of the two **metastable phases** (the open and closed basins of attraction).
* The clusters would be separated by a **sparse void** of points, confirming the presence of a **high energy barrier** between the two states.
* The entire topology would be a map of the protein's conformational **free-energy landscape**.

---
***

### 💡 Hands-On Project Ideas 🛠️

These projects require computational techniques to implement the core concepts of data representation and linear geometry.

### Project 1: Data Preparation and Standardization

* **Goal:** Implement the standardization process and observe its effect on feature means and variances.
* **Setup:** Generate a synthetic dataset $X$ of size $M=1000$ and $D=5$. Choose one feature (column) to have a large mean ($\mu \approx 100$) and one to have a large variance ($\sigma^2 \approx 50$).
* **Steps:**
    1.  Compute the mean vector $\boldsymbol{\mu}$ and standard deviation vector $\boldsymbol{\sigma}$ for the raw data $X$.
    2.  Apply the standardization formula: $x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$.
    3.  Compute the mean and standard deviation of the transformed data $X'$.
* ***Goal***: Show that the transformed mean is approximately 0 and the transformed standard deviation is approximately 1 for all features, demonstrating that all original physical scales have been normalized.

### Project 2: Computing and Interpreting the Covariance Matrix

* **Goal:** Compute the covariance matrix $\Sigma$ and analyze its physical meaning.
* **Setup:** Generate a synthetic dataset $X$ ($M=1000, D=3$) where you manually engineer correlations: $X_{\text{col} 2} = 0.8 \cdot X_{\text{col} 1} + \text{noise}$, and $X_{\text{col} 3}$ is independent.
* **Steps:**
    1.  Compute the $3 \times 3$ covariance matrix $\Sigma$.
    2.  Identify the variances ($\Sigma_{ii}$) and the covariances ($\Sigma_{ij}, i \neq j$).
* ***Goal***: Show that $\Sigma_{1,2}$ is large (high correlation), while $\Sigma_{1,3}$ and $\Sigma_{2,3}$ are near zero (low correlation), confirming that the matrix correctly encodes the engineered physical dependencies.

### Project 3: Principal Component Projection (Code Demo Replication)

* **Goal:** Replicate the core PCA visualization (the code demo from 1.7) to understand the concept of projecting the data's "shadow."
* **Setup:** Use the provided synthetic 5D correlated data (or generate your own strongly correlated data).
* **Steps:**
    1.  Use the `sklearn.decomposition.PCA` class (setting `n_components=2`).
    2.  Apply `fit_transform` to get the 2D projected data $X_{\text{pca}}$.
* ***Goal***: Plot the 2D projected data. The data should form an elongated ellipse, confirming that the first principal axis (PC1) correctly aligns with the direction of the strongest correlation (variance).

### Project 4: Quantifying Dimensionality Reduction

* **Goal:** Quantify the effective dimensionality by analyzing the explained variance ratio of the eigenvalues.
* **Setup:** Use a high-dimensional dataset (e.g., $D=50$ random features) with a known low-dimensional core structure (e.g., only the first 5 features contain signal).
* **Steps:**
    1.  Apply PCA (no component limit) to get all $D$ eigenvalues $\lambda_k$.
    2.  Compute the **explained variance ratio** for the first few components (e.g., $k=1$ to $10$).
    3.  Plot the cumulative explained variance versus the number of components $k$.
* ***Goal***: Show that the first $5$ components capture nearly $100\%$ of the variance, providing quantitative evidence of the system's low intrinsic dimensionality (the true dimensionality of the manifold $\mathcal{M}$).


