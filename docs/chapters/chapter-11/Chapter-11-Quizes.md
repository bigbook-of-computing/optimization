# Chapter 11: Graphical Models & Probabilistic Graphs

This chapter introduces graphical models as a framework for representing and reasoning about complex systems with structured dependencies.

| Section | Topic Summary |
| :--- | :--- |
| **11.1** | **From Single Models to Networks of Belief**: Introduction to structured dependencies and graphical factorization. |
| **11.2** | **Bayesian Networks (BNs)**: Using Directed Acyclic Graphs (DAGs) to model causal relationships. |
| **11.3** | **Markov Random Fields (MRFs)**: Using undirected graphs to model symmetric interactions, with connections to statistical physics. |
| **11.4** | **Conditional Independence & Markov Blankets**: The principles that enable local computation for global inference. |
| **11.5** | **Factor Graphs**: A unified bipartite representation for implementing inference algorithms. |
| **11.6** | **Belief Propagation (BP)**: The core message-passing algorithm for inference. |
| **11.7** | **Variational Inference (VI)**: An optimization-based approach to approximate intractable posteriors. |
| **11.8** | **Loopy and Approximate Belief Propagation**: Applying BP to graphs with cycles. |
| **11.9** | **Dynamic and Temporal Graphical Models**: Extending graphical models to handle time-series data (HMMs, Kalman Filters). |
| **11.10-14** | **Worked Examples, Code, and Takeaways**: Practical applications and summary. |

---

!!! note "Quiz"
    **1. What is the primary purpose of using a graphical model to represent a joint probability distribution?**
    
    *   A. To ensure the distribution is always Gaussian.
    *   B. To factorize a complex, high-dimensional joint distribution into a product of simpler, local conditional probabilities.
    *   C. To calculate the partition function Z directly.
    *   D. To convert all variables into a binary format.

??? info "See Answer"
    **B. To factorize a complex, high-dimensional joint distribution into a product of simpler, local conditional probabilities.** This factorization makes computation tractable by exploiting the conditional independence structure of the system.

---

!!! note "Quiz"
    **2. Which type of graphical model is best suited for representing systems with a clear causal hierarchy, where influence flows in one direction?**
    
    *   A. Markov Random Field (MRF).
    *   B. Factor Graph.
    *   C. Bayesian Network (BN).
    *   D. Ising Model.

??? info "See Answer"
    **C. Bayesian Network (BN).** BNs use Directed Acyclic Graphs (DAGs) to model causal relationships, where edges point from cause to effect.

---

!!! note "Quiz"
    **3. A Markov Random Field (MRF) uses an undirected graph to model symmetric dependencies. This makes it a direct generalization of which foundational model from statistical physics?**
    
    *   A. The Schrödinger Equation.
    *   B. The Navier-Stokes equations.
    *   C. The Ising Model.
    *   D. The Black-Scholes model.

??? info "See Answer"
    **C. The Ising Model.** An MRF is a generalization of spin networks like the Ising or Potts models, where nodes are spins and edge potentials represent coupling energies.

---

!!! note "Quiz"
    **4. In a Bayesian Network, the joint probability distribution $p(\mathbf{x})$ is factored according to what rule?**
    
    *   A. $p(\mathbf{x}) = \prod_i p(x_i | 	ext{children}(x_i))$
    *   B. $p(\mathbf{x}) = \frac{1}{Z} \prod_C \psi_C(\mathbf{x}_C)$
    *   C. $p(\mathbf{x}) = \prod_i p(x_i | 	ext{parents}(x_i))$
    *   D. $p(\mathbf{x}) = \sum_i p(x_i)$

??? info "See Answer"
    **C. $p(\mathbf{x}) = \prod_i p(x_i | 	ext{parents}(x_i))$.** This is the chain rule of probability applied to the DAG structure, where each variable is conditioned only on its direct parents.

---

!!! note "Quiz"
    **5. What is the "Markov Blanket" of a node $x_i$ in a graphical model?**
    
    *   A. The set of all nodes in the graph except $x_i$.
    *   B. The minimal set of neighboring nodes that renders $x_i$ conditionally independent of all other nodes in the graph.
    *   C. The set of all nodes that are causally descended from $x_i$.
    *   D. The clique with the largest number of nodes.

??? info "See Answer"
    **B. The minimal set of neighboring nodes that renders $x_i$ conditionally independent of all other nodes in the graph.** This is the key property that allows for local computation.

---

!!! note "Quiz"
    **6. In an undirected MRF, the Markov Blanket of a node $x_i$ consists of...**
    
    *   A. Its parents, children, and co-parents.
    *   B. Only its parents.
    *   C. Its set of direct neighbors.
    *   D. All nodes in the same clique.

??? info "See Answer"
    **C. Its set of direct neighbors.** The influence of the rest of the graph is "screened" by the immediate neighbors.

---

!!! note "Quiz"
    **7. The joint probability distribution of an MRF is given by the Gibbs distribution, $p(\mathbf{x}) = \frac{1}{Z} e^{-E(\mathbf{x})}$. What does the term $E(\mathbf{x})$ represent?**
    
    *   A. The evidence for the data.
    *   B. The entropy of the system.
    *   C. The total energy of the system configuration $\mathbf{x}$.
    *   D. The Euclidean distance.

??? info "See Answer"
    **C. The total energy of the system configuration $\mathbf{x}$.** This formulation directly links probability to energy, where low-energy states are more probable.

---

!!! note "Quiz"
    **8. What is the primary advantage of using a Factor Graph representation?**
    
    *   A. It can model systems that BNs and MRFs cannot.
    *   B. It explicitly separates variable nodes from factor (function) nodes, which simplifies the implementation of message-passing algorithms like Belief Propagation.
    *   C. It guarantees that the graph will have no loops.
    *   D. It automatically calculates the partition function.

??? info "See Answer"
    **B. It explicitly separates variable nodes from factor (function) nodes, which simplifies the implementation of message-passing algorithms like Belief Propagation.** Its bipartite structure provides algorithmic clarity.

---

!!! note "Quiz"
    **9. What is the main goal of the Belief Propagation (BP) algorithm?**
    
    *   A. To find the most complex graph structure.
    *   B. To compute the marginal probability $p(x_i)$ for each variable node in the network.
    *   C. To learn the parameters of the potential functions.
    *   D. To calculate the KL divergence.

??? info "See Answer"
    **B. To compute the marginal probability $p(x_i)$ for each variable node in the network.** This final marginal is the node's updated "belief."

---

!!! note "Quiz"
    **10. Belief Propagation is guaranteed to converge to the exact marginal probabilities only on which type of graph structure?**
    
    *   A. Graphs with many loops.
    *   B. Fully connected graphs.
    *   C. Tree-structured graphs (graphs with no cycles).
    *   D. Directed Acyclic Graphs only.

??? info "See Answer"
    **C. Tree-structured graphs (graphs with no cycles).** On loopy graphs, BP is an approximate algorithm.

---

!!! note "Quiz"
    **11. The iterative message-passing process in Belief Propagation is the computational analogue of what physical process?**
    
    *   A. Radioactive decay.
    *   B. A system undergoing a phase transition.
    *   C. A distributed system relaxing to statistical equilibrium.
    *   D. A particle moving in a uniform magnetic field.

??? info "See Answer"
    **C. A distributed system relaxing to statistical equilibrium.** The messages are like local forces, and convergence is achieved when all forces are balanced and beliefs are self-consistent.

---

!!! note "Quiz"
    **12. When the true posterior distribution $p(\mathbf{	heta}|\mathcal{D})$ is intractable, Variational Inference (VI) finds an approximate distribution $q(\mathbf{	heta})$ by minimizing what quantity?**
    
    *   A. The model's accuracy.
    *   B. The number of parameters.
    *   C. The Kullback-Leibler (KL) divergence, $D_{\mathrm{KL}}(q||p)$.
    *   D. The partition function Z.

??? info "See Answer"
    **C. The Kullback-Leibler (KL) divergence, $D_{\mathrm{KL}}(q||p)$.** This measures the "distance" between the approximate distribution and the true posterior.

---

!!! note "Quiz"
    **13. Minimizing the KL divergence in Variational Inference is equivalent to maximizing what other quantity?**
    
    *   A. The model's complexity.
    *   B. The Evidence Lower Bound (ELBO).
    *   C. The number of graph cycles.
    *   D. The learning rate.

??? info "See Answer"
    **B. The Evidence Lower Bound (ELBO).** This reframes the inference problem as a more standard optimization (maximization) problem.

---

!!! note "Quiz"
    **14. Variational Inference is the direct statistical analogue of what powerful approximation technique from physics?**
    
    *   A. The Born-Oppenheimer approximation.
    *   B. The mean-field approximation.
    *   C. The perturbation theory.
    *   D. The WKB approximation.

??? info "See Answer"
    **B. The mean-field approximation.** Both methods simplify a complex, interacting system by assuming its components can be treated with simpler, independent distributions.

---

!!! note "Quiz"
    **15. What is the primary challenge of applying Belief Propagation to "loopy" graphs?**
    
    *   A. The messages can become statistically correlated as they circulate, violating the independence assumption of BP.
    *   B. The graph becomes directed.
    *   C. The number of nodes increases exponentially.
    *   D. The potential functions become non-linear.

??? info "See Answer"
    **A. The messages can become statistically correlated as they circulate, violating the independence assumption of BP.** This is why Loopy BP is an approximate, not exact, algorithm.

---

!!! note "Quiz"
    **16. A Hidden Markov Model (HMM) is a type of Dynamic Bayesian Network (DBN) used to model sequences. What is the core assumption of an HMM?**
    
    *   A. The underlying state of the system is always directly observable.
    *   B. The true state of the system is hidden (latent), and we only see noisy observations generated by that state.
    *   C. The system has no memory of past states.
    *   D. The relationships between variables are linear and the noise is Gaussian.

??? info "See Answer"
    **B. The true state of the system is hidden (latent), and we only see noisy observations generated by that state.** This is fundamental to tasks like speech recognition and sequence tagging.

---

!!! note "Quiz"
    **17. The Kalman Filter is a DBN that is the continuous-variable counterpart to the HMM. It is specifically designed for systems with what properties?**
    
    *   A. Non-linear dynamics and discrete states.
    *   B. Linear relationships and Gaussian noise.
    *   C. Asymmetric dependencies and categorical data.
    *   D. Noisy observations but a perfectly known state.

??? info "See Answer"
    **B. Linear relationships and Gaussian noise.** It provides an optimal solution for state estimation under these assumptions, widely used in robotics and control.

---

!!! note "Quiz"
    **18. In the context of learning graphical models, what is "structure learning"?**
    
    *   A. Learning the numerical values of the Conditional Probability Tables (CPTs) for a fixed graph.
    *   B. Inferring the topology of the graph itself—that is, discovering which nodes are connected by edges.
    *   C. Finding the optimal learning rate for the model.
    *   D. Calculating the marginal probability of each node.

??? info "See Answer"
    **B. Inferring the topology of the graph itself—that is, discovering which nodes are connected by edges.** This is analogous to discovering the physical laws and connectivity of an unknown system.

---

!!! note "Quiz"
    **19. In a Bayesian Network for a "wet grass" scenario with variables (Rain, Sprinkler, Wet Grass), what is the relationship between "Rain" and "Sprinkler"?**
    
    *   A. They are children of "Wet Grass".
    *   B. They are conditionally dependent given "Wet Grass".
    *   C. They are independent causes of "Wet Grass".
    *   D. "Rain" is the parent of "Sprinkler".

??? info "See Answer"
    **C. They are independent causes of "Wet Grass".** In a simple BN for this problem, Rain and Sprinkler would both be parents of Wet Grass, but would not have an edge between them, making them marginally independent.

---

!!! note "Quiz"
    **20. The transition probability $p(\mathbf{x}_t | \mathbf{x}_{t-1})$ in a Dynamic Bayesian Network is analogous to what concept in physics?**
    
    *   A. The system's temperature.
    *   B. The system's total mass.
    *   C. The system's equations of motion.
    *   D. The system's ground state energy.

??? info "See Answer"
    **C. The system's equations of motion.** It describes the dynamics of how the system evolves from one state to the next in a probabilistic manner.

---

!!! note "Quiz"
    **21. What is the main difference between parameter learning and structure learning in graphical models?**
    
    *   A. Parameter learning infers connectivity, while structure learning infers interaction strength.
    *   B. Parameter learning infers interaction strength for a fixed connectivity, while structure learning infers the connectivity itself.
    *   C. Parameter learning is only for BNs, while structure learning is only for MRFs.
    *   D. There is no difference; they are the same process.

??? info "See Answer"
    **B. Parameter learning infers interaction strength for a fixed connectivity, while structure learning infers the connectivity itself.** Parameter learning is about "how strong" the connections are; structure learning is about "which" connections exist.

---

!!! note "Quiz"
    **22. In computer vision, an MRF is often used for image denoising. What role do the potential functions play in this context?**
    
    *   A. They ensure the image is converted to grayscale.
    *   B. They enforce a smoothness prior, penalizing configurations where neighboring pixels have different labels.
    *   C. They calculate the total number of pixels in the image.
    *   D. They apply a Fourier transform to the image.

??? info "See Answer"
    **B. They enforce a smoothness prior, penalizing configurations where neighboring pixels have different labels.** This encourages local consistency, which helps remove salt-and-pepper noise.

---

!!! note "Quiz"
    **23. The "forward algorithm" in an HMM is used for which inference task?**
    
    *   A. Finding the most likely sequence of hidden states (Viterbi algorithm does this).
    *   B. Learning the transition probabilities.
    *   C. Computing the belief over the hidden state at time $t$ given all observations up to that point, $P(\mathbf{z}_t | \mathbf{x}_{1:t})$.
    *   D. Reversing the direction of time in the model.

??? info "See Answer"
    **C. Computing the belief over the hidden state at time $t$ given all observations up to that point, $P(\mathbf{z}_t | \mathbf{x}_{1:t})$.** It is the core of state estimation or "filtering."

---

!!! note "Quiz"
    **24. Why is the acyclicity constraint crucial for a Bayesian Network?**
    
    *   A. It ensures the graph is undirected.
    *   B. It prevents infinite causal loops, ensuring a well-defined and factorizable joint probability distribution.
    *   C. It makes the graph fully connected.
    *   D. It guarantees the variables are all independent.

??? info "See Answer"
    **B. It prevents infinite causal loops, ensuring a well-defined and factorizable joint probability distribution.** A cycle like A -> B -> A would imply a variable is its own ancestor, which is logically inconsistent.

---

!!! note "Quiz"
    **25. The move from handcrafted graphical models (Part III) to deep learning (Part IV) represents a shift from...**
    
    *   A. Probabilistic models to deterministic models.
    *   B. Supervised learning to unsupervised learning.
    *   C. Explicitly designed probabilistic structures to implicitly learned hierarchical representations.
    *   D. Continuous variables to discrete variables.

??? info "See Answer"
    **C. Explicitly designed probabilistic structures to implicitly learned hierarchical representations.** Deep networks learn the features and structure autonomously, rather than having them specified by a human designer.


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


