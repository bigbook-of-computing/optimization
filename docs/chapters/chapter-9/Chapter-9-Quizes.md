!!! note "Quiz"
	**1. What fundamental principle provides the mathematical bridge between minimizing energy in physics and maximizing probability in statistical inference?**
	- A. The Principle of Least Action
	- B. The Central Limit Theorem
	- C. The Boltzmann Distribution
	- D. The Law of Large Numbers

	??? info "See Answer"
		**C. The Boltzmann Distribution.** The Boltzmann distribution, $P(\mathbf{s}) \propto e^{-E(\mathbf{s})/k_B T}$, directly links the probability of a state to its energy, making the minimization of energy equivalent to the maximization of probability.

!!! note "Quiz"
	**2. In the context of Bayesian learning, what is the primary philosophical difference between optimization and inference?**
	- A. Optimization uses gradients, while inference uses sampling.
	- B. Optimization seeks a single best point estimate, while inference seeks to characterize a full probability distribution.
	- C. Optimization is for discrete systems, while inference is for continuous systems.
	- D. Optimization minimizes loss, while inference maximizes a reward function.

	??? info "See Answer"
		**B. Optimization seeks a single best point estimate, while inference seeks to characterize a full probability distribution.** Optimization aims to find the single best parameter set ($\mathbf{	heta}^*$), whereas inference aims to model the uncertainty by finding the posterior distribution $p(\mathbf{	heta}|\mathcal{D})$.

!!! note "Quiz"
	**3. In Bayes' Theorem, $p(\mathbf{	heta}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{	heta}) p(\mathbf{	heta})}{p(\mathcal{D})}$, which term represents the "updated belief" after observing data?**
	- A. $p(\mathcal{D}|\mathbf{	heta})$ (Likelihood)
	- B. $p(\mathbf{	heta})$ (Prior)
	- C. $p(\mathcal{D})$ (Model Evidence)
	- D. $p(\mathbf{	heta}|\mathcal{D})$ (Posterior)

	??? info "See Answer"
		**D. $p(\mathbf{	heta}|\mathcal{D})$ (Posterior).** The posterior distribution represents the state of knowledge about the parameters $\mathbf{	heta}$ after the evidence $\mathcal{D}$ has been incorporated.

!!! note "Quiz"
	**4. The Maximum A Posteriori (MAP) estimation is equivalent to regularized optimization. What does the regularization term correspond to in the Bayesian framework?**
	- A. The negative log-likelihood
	- B. The negative log-prior
	- C. The model evidence
	- D. The posterior predictive distribution

	??? info "See Answer"
		**B. The negative log-prior.** The MAP objective is to minimize $[-\ln p(\mathcal{D}|\mathbf{	heta}) - \ln p(\mathbf{	heta})]$, where $-\ln p(\mathcal{D}|\mathbf{	heta})$ is the loss and $-\ln p(\mathbf{	heta})$ is the regularization term.

!!! note "Quiz"
	**5. If you use L2 regularization (weight decay) in a linear regression model, what kind of prior are you implicitly assuming for the model's weights?**
	- A. A Laplace prior
	- B. A Gaussian prior
	- C. A Uniform prior
	- D. A Bernoulli prior

	??? info "See Answer"
		**B. A Gaussian prior.** An L2 penalty term of $\|\mathbf{	heta}\|^2$ in the loss function is mathematically equivalent to assuming a zero-mean Gaussian prior, $p(\mathbf{	heta}) \propto e^{-\|\mathbf{	heta}\|^2/2\sigma^2}$, on the parameters.

!!! note "Quiz"
	**6. What is a "conjugate prior" in the context of Bayesian inference?**
	- A. A prior that is always a Gaussian distribution.
	- B. A prior that ensures the posterior distribution belongs to the same family of distributions as the prior.
	- C. A prior that is learned from the data itself.
	- D. A prior that has a mean of zero.

	??? info "See Answer"
		**B. A prior that ensures the posterior distribution belongs to the same family of distributions as the prior.** This property, like in the Beta-Binomial case, allows for an analytical, closed-form solution for the posterior, simplifying computation.

!!! note "Quiz"
	**7. The model evidence, $p(\mathcal{D}|M)$, is crucial for model comparison. What principle of scientific reasoning does it naturally enforce?**
	- A. The Falsification Principle
	- B. The Principle of Locality
	- C. Occam's Razor
	- D. The Correspondence Principle

	??? info "See Answer"
		**C. Occam's Razor.** The evidence integral $p(\mathcal{D}|M) = \int p(\mathcal{D}|\mathbf{	heta}, M) p(\mathbf{	heta}|M) d\mathbf{	heta}$ penalizes overly complex models that spread their prior probability thinly over a large parameter space, favoring simpler models that provide a good fit over a larger prior volume.

!!! note "Quiz"
	**8. Variational Inference (VI) reframes the intractable problem of finding the posterior $p(\mathbf{	heta}|\mathcal{D})$ into an optimization problem. What quantity does VI seek to minimize?**
	- A. The L2 norm of the parameters.
	- B. The KL divergence between an approximate distribution $q(\mathbf{	heta})$ and the true posterior $p(\mathbf{	heta}|\mathcal{D})$.
	- C. The number of parameters in the model.
	- D. The variance of the posterior predictive distribution.

	??? info "See Answer"
		**B. The KL divergence between an approximate distribution $q(\mathbf{	heta})$ and the true posterior $p(\mathbf{	heta}|\mathcal{D})$.** Minimizing this divergence is equivalent to minimizing the Variational Free Energy (or maximizing the ELBO), making the approximation $q$ as close as possible to the true posterior $p$.

!!! note "Quiz"
	**9. What is the key difference between a Bayesian "credible interval" and a frequentist "confidence interval"?**
	- A. A credible interval is always wider than a confidence interval.
	- B. A credible interval makes a direct probabilistic statement about the parameter itself, while a confidence interval is a statement about the long-run frequency of the procedure.
	- C. A credible interval can only be calculated for Gaussian posteriors.
	- D. A confidence interval is used for model comparison, while a credible interval is for parameter estimation.

	??? info "See Answer"
		**B. A credible interval makes a direct probabilistic statement about the parameter itself, while a confidence interval is a statement about the long-run frequency of the procedure.** A 95% credible interval means there is a 95% probability the true parameter lies within it.

!!! note "Quiz"
	**10. The posterior predictive distribution, $p(y_*|x_*, \mathcal{D})$, provides a robust prediction by averaging over the uncertainty in the parameters. What is this process of averaging called?**
	- A. Bootstrapping
	- B. Cross-validation
	- C. Marginalization
	- D. Regularization

	??? info "See Answer"
		**C. Marginalization.** The formula $p(y_*|x_*, \mathcal{D}) = \int p(y_*|x_*, \mathbf{	heta}) p(\mathbf{	heta}|\mathcal{D}) d\mathbf{	heta}$ integrates out (marginalizes) the parameters $\mathbf{	heta}$, effectively averaging the predictions of all possible models weighted by their posterior probability.

!!! note "Quiz"
	**11. What is the defining structural characteristic of a Bayesian Network?**
	- A. It is an undirected graph.
	- B. It is a fully connected graph.
	- C. It is a Directed Acyclic Graph (DAG).
	- D. It must contain cycles to model feedback.

	??? info "See Answer"
		**C. It is a Directed Acyclic Graph (DAG).** The nodes represent random variables, and the directed edges represent conditional dependencies, with the acyclic constraint enforcing a clear causal hierarchy.

!!! note "Quiz"
	**12. In the Bayesian coin toss example with a Beta prior and Binomial likelihood, what happens to the posterior distribution as more data is collected?**
	- A. The posterior mean moves towards 0.5, regardless of the data.
	- B. The posterior distribution becomes wider, reflecting increased uncertainty.
	- C. The posterior distribution becomes narrower and more concentrated around the empirical frequency of the data.
	- D. The posterior distribution converges to a uniform distribution.

	??? info "See Answer"
		**C. The posterior distribution becomes narrower and more concentrated around the empirical frequency of the data.** This demonstrates learning as entropy reduction: uncertainty decreases as evidence accumulates.

!!! note "Quiz"
	**13. The Variational Free Energy, which is minimized in Variational Inference, is analogous to what concept in statistical physics?**
	- A. The Hamiltonian
	- B. The Partition Function
	- C. The Helmholtz Free Energy
	- D. The Canonical Ensemble

	??? info "See Answer"
		**C. The Helmholtz Free Energy.** Both concepts balance an energy term (fit to data) and an entropy term (complexity/uncertainty), and the system evolves to minimize this total free energy.

!!! note "Quiz"
	**14. If you have no prior knowledge about a parameter and want to use an uninformative prior, which of the following would be a suitable choice for a probability parameter $	heta \in [0, 1]$?**
	- A. A very narrow Gaussian distribution centered at 0.
	- B. A Beta(100, 100) distribution.
	- C. A Beta(1, 1) distribution (which is a uniform distribution).
	- D. A Dirac delta function at 0.5.

	??? info "See Answer"
		**C. A Beta(1, 1) distribution (which is a uniform distribution).** A uniform prior assigns equal probability to all possible values of the parameter, representing a state of maximal ignorance before seeing data.

!!! note "Quiz"
	**15. What is the primary advantage of Maximum Likelihood (ML) estimation compared to MAP estimation?**
	- A. It is always more accurate.
	- B. It does not require the specification of a prior distribution.
	- C. It is guaranteed to avoid overfitting.
	- D. It provides a full posterior distribution.

	??? info "See Answer"
		**B. It does not require the specification of a prior distribution.** ML estimation only considers the likelihood $p(\mathcal{D}|\mathbf{	heta})$, which can be simpler if a good prior is unknown or difficult to formulate.

!!! note "Quiz"
	**16. In a Bayesian Network, the joint probability distribution $p(x_1, ..., x_N)$ can be factored into a product of local conditional probabilities. This factorization is justified by:**
	- A. The chain rule of probability and the conditional independencies encoded in the graph.
	- B. The Central Limit Theorem.
	- C. The law of total probability.
	- D. The use of conjugate priors.

	??? info "See Answer"
		**A. The chain rule of probability and the conditional independencies encoded in the graph.** The DAG structure implies that a variable is conditionally independent of its non-descendants given its parents, allowing the full joint distribution to be simplified as $p(\mathbf{x}) = \prod_i p(x_i|	ext{parents}(x_i))$.

!!! note "Quiz"
	**17. The process of Bayesian learning can be viewed as a form of "entropy reduction." What does this mean?**
	- A. The model's parameters become more random over time.
	- B. The total energy of the system increases.
	- C. The uncertainty about the model's parameters, as measured by the width of the posterior, decreases as more data is observed.
	- D. The model becomes simpler by removing parameters.

	??? info "See Answer"
		**C. The uncertainty about the model's parameters, as measured by the width of the posterior, decreases as more data is observed.** A wide prior (high entropy) is refined into a narrow posterior (low entropy) through the accumulation of evidence.

!!! note "Quiz"
	**18. When comparing two models, $M_1$ and $M_2$, using the Bayes Factor, $BF_{12} = \frac{p(\mathcal{D}|M_1)}{p(\mathcal{D}|M_2)}$, what does a value of $BF_{12} > 10$ strongly suggest?**
	- A. Model $M_1$ is more complex than Model $M_2$.
	- B. The data provides strong evidence in favor of Model $M_1$ over Model $M_2$.
	- C. The prior for Model $M_1$ was poorly chosen.
	- D. Both models are overfit to the data.

	??? info "See Answer"
		**B. The data provides strong evidence in favor of Model $M_1$ over Model $M_2$.** The Bayes Factor is the ratio of the model evidences; a value significantly greater than 1 indicates that the data is much more probable under the first model.

!!! note "Quiz"
	**19. The physical analogy for Bayesian inference suggests that the likelihood term $p(\mathcal{D}|\mathbf{	heta})$ acts as a force that:**
	- A. Increases the overall entropy of the system.
	- B. Breaks the initial symmetry defined by the prior distribution.
	- C. Ensures the posterior is always Gaussian.
	- D. Makes the model evidence easier to compute.

	??? info "See Answer"
		**B. Breaks the initial symmetry defined by the prior distribution.** The prior represents the initial, often symmetric, state of belief. The data, through the likelihood, acts as an external field that perturbs this symmetry and forces the posterior to conform to the observed reality.

!!! note "Quiz"
	**20. What is the primary output of a Bayesian Neural Network (BNN) that distinguishes it from a standard neural network?**
	- A. It always has fewer layers.
	- B. It produces a single point prediction with no uncertainty.
	- C. It produces a predictive distribution, including uncertainty estimates for its predictions.
	- D. It can only be used for classification, not regression.

	??? info "See Answer"
		**C. It produces a predictive distribution, including uncertainty estimates for its predictions.** By placing distributions over its weights, a BNN can propagate uncertainty from the parameters to the final prediction, providing a measure of confidence.

!!! note "Quiz"
	**21. In the context of Variational Inference, what is the "mean-field approximation"?**
	- A. Assuming the true posterior is a Gaussian distribution.
	- B. Approximating the true posterior by assuming the parameters are mutually independent.
	- C. Using the mean of the data as the final estimate.
	- D. Replacing the true likelihood with a simpler function.

	??? info "See Answer"
		**B. Approximating the true posterior by assuming the parameters are mutually independent.** This simplifies the problem by decoupling the interactions between parameters in the approximate distribution $q(\mathbf{	heta})$, making the optimization tractable.

!!! note "Quiz"
	**22. If your prior belief is that a coin is fair, and you observe 3 heads in 3 tosses, the MAP estimate for the coin's bias $	heta$ will be:**
	- A. Exactly 1.0 (the MLE).
	- B. Exactly 0.5 (the prior mean).
	- C. A value between 0.5 and 1.0.
	- D. A value less than 0.5.

	??? info "See Answer"
		**C. A value between 0.5 and 1.0.** The posterior is a compromise between the prior (centered at 0.5) and the likelihood (peaked at 1.0). The MAP estimate will be pulled away from the MLE towards the prior mean.

!!! note "Quiz"
	**23. The KL divergence, $D_{	ext{KL}}(Q || P)$, is used to measure the "distance" between two distributions. If $D_{	ext{KL}}(Q || P) = 0$, what does this imply?**
	- A. The distributions are orthogonal.
	- B. The distributions are identical ($Q=P$).
	- C. The mean of Q is zero.
	- D. The variance of P is infinite.

	??? info "See Answer"
		**B. The distributions are identical ($Q=P$).** The KL divergence is zero if and only if the two distributions are the same everywhere.

!!! note "Quiz"
	**24. Why is the model evidence $p(\mathcal{D})$ often called the "marginal likelihood"?**
	- A. Because it is only marginally important for inference.
	- B. Because it is calculated by marginalizing (integrating out) the parameters $\mathbf{	heta}$ from the joint distribution $p(\mathcal{D}, \mathbf{	heta})$.
	- C. Because it represents the margin of error in the model's predictions.
	- D. Because it is calculated at the margins of the parameter space.

	??? info "See Answer"
		**B. Because it is calculated by marginalizing (integrating out) the parameters $\mathbf{	heta}$ from the joint distribution $p(\mathcal{D}, \mathbf{	heta})$.** The calculation is $p(\mathcal{D}) = \int p(\mathcal{D}|\mathbf{	heta})p(\mathbf{	heta})d\mathbf{	heta}$.

!!! note "Quiz"
	**25. Linear Regression derived from a Maximum Likelihood principle assumes the errors (residuals) follow what kind of distribution?**
	- A. A Laplace distribution
	- B. A Bernoulli distribution
	- C. A Poisson distribution
	- D. A Gaussian distribution

	??? info "See Answer"
		**D. A Gaussian distribution.** Minimizing the sum of squared errors in linear regression is mathematically equivalent to maximizing the likelihood under the assumption that the data is generated from a linear model with additive Gaussian noise.


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


