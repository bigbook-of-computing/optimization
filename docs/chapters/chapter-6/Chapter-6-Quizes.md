# **Chapter-6: Quizes**

---

!!! note "Quiz"
    **1. What is the primary failure of standard Gradient Descent in high-dimensional, anisotropic "ravine" landscapes?**

    - A. It converges too quickly to a sharp minimum.
    - B. It requires computing the full Hessian matrix.
    - C. It suffers from "zigzagging" and "stalling" due to a high Hessian condition number.
    - D. It can only be used for convex functions.

    ??? info "See Answer"
        **Correct: C**

        *(The high condition number means curvature varies drastically. The gradient points across the steep ravine walls, causing oscillation (zigzagging), while a single learning rate small enough to prevent divergence is too small for the flat valley floor, causing stalling.)*

---

!!! note "Quiz"
    **2. The introduction of inertia into optimization dynamics upgrades the physical model from an overdamped relaxation to what?**

    - A. A quantum tunneling process.
    - B. A damped harmonic oscillator.
    - C. An ideal gas expansion.
    - D. A random walk.

    ??? info "See Answer"
        **Correct: B**

        *(By adding a "mass" term ($m\ddot{\mathbf{	heta}}$), the dynamics are no longer dominated by friction alone, allowing for velocity accumulation and oscillation, just like a physical damped oscillator.)*

---

!!! note "Quiz"
    **3. In the Momentum update rule, $\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta 
abla L(\mathbf{	heta}_t)$, what is the role of the hyperparameter $\beta$?**

    - A. It sets the learning rate.
    - B. It controls the amount of memory or inertia, acting as a damping factor.
    - C. It normalizes the gradient.
    - D. It is the batch size.

    ??? info "See Answer"
        **Correct: B**

        *(A typical value of $\beta=0.9$ means 90% of the previous velocity is retained, allowing the optimizer to accumulate momentum and smooth its path.)*

---

!!! note "Quiz"
    **4. How does Momentum's inertia help solve the "stalling" problem on flat plateaus?**

    - A. It increases the learning rate automatically.
    - B. It allows the optimizer to "coast" across the plateau using its accumulated velocity, even when the current gradient is near zero.
    - C. It adds random noise to escape the plateau.
    - D. It computes a second-order gradient.

    ??? info "See Answer"
        **Correct: B**

        *(The kinetic energy from the accumulated momentum prevents the optimizer from stopping just because the local force (gradient) has vanished.)*

---

!!! note "Quiz"
    **5. What is the key "anticipatory" idea behind Nesterov Accelerated Gradient (NAG)?**

    - A. It calculates the gradient at the current position $\mathbf{	heta}_t$.
    - B. It uses a much larger momentum term $\beta$.
    - C. It computes the gradient at a predicted future position ($\mathbf{	heta}_t + \beta \mathbf{v}_t$) to correct its trajectory sooner.
    - D. It eliminates the momentum term entirely.

    ??? info "See Answer"
        **Correct: C**

        *(By "looking ahead," NAG can sense the curvature of the valley and start turning earlier, preventing overshooting and following a more optimal path.)*

---

!!! note "Quiz"
    **6. RMSProp is an "adaptive" algorithm because it:**

    - A. Uses a fixed, global learning rate for all parameters.
    - B. Introduces momentum into the update.
    - C. Dynamically computes a per-parameter learning rate based on the history of squared gradients.
    - D. Periodically increases the batch size.

    ??? info "See Answer"
        **Correct: C**

        *(It adapts the step size for each parameter, applying a "strong brake" in steep directions and a "light brake" in flat directions.)*

---

!!! note "Quiz"
    **7. In the RMSProp update, $\mathbf{	heta}_{t+1} = \mathbf{	heta}_t - \frac{\eta}{\sqrt{s_t+\epsilon}}
abla L_t$, what does the term $s_t$ represent?**

    - A. The accumulated velocity.
    - B. A running average of the first moment of the gradient.
    - C. A running average of the squared gradients (historical stiffness).
    - D. The bias correction term.

    ??? info "See Answer"
        **Correct: C**

        *(A large $s_t$ for a parameter indicates it's in a steep (stiff) direction, which leads to a smaller effective learning rate for that parameter.)*

---

!!! note "Quiz"
    **8. The Adam (Adaptive Moment Estimation) optimizer is a synthesis of which two preceding methods?**

    - A. Gradient Descent and Newton's Method.
    - B. Momentum (first moment estimation) and RMSProp (second moment estimation).
    - C. Simulated Annealing and Genetic Algorithms.
    - D. Nesterov Acceleration and Gradient Clipping.

    ??? info "See Answer"
        **Correct: B**

        *(Adam combines the inertia from momentum with the per-parameter adaptive scaling from RMSProp to create a robust and widely used optimizer.)*

---

!!! note "Quiz"
    **9. What is the purpose of the "bias correction" step in the Adam algorithm?**

    - A. To prevent the learning rate from becoming too large.
    - B. To add more noise to the gradient.
    - C. To compensate for the fact that the moment estimates ($m_t, v_t$) are initialized at zero and are biased toward zero in early steps.
    - D. To ensure the loss function is convex.

    ??? info "See Answer"
        **Correct: C**

        *(Without bias correction, the initial steps would be artificially small because the moment estimates have not yet accumulated enough information.)*

---

!!! note "Quiz"
    **10. How do adaptive methods like Adam and RMSProp geometrically transform the optimization landscape?**

    - A. They make the landscape more non-convex.
    - B. They increase the condition number $\kappa$.
    - C. They act as a learned preconditioner, effectively making an anisotropic ravine appear more isotropic (spherical).
    - D. They flatten all minima.

    ??? info "See Answer"
        **Correct: C**

        *(By scaling each parameter's update, they "sphericize" the landscape, allowing the optimizer to take more direct steps toward the minimum.)*

---

!!! note "Quiz"
    **11. The update rule of adaptive methods is a diagonal approximation of which intractable second-order optimization method?**

    - A. Conjugate Gradient.
    - B. Levenberg-Marquardt.
    - C. Quasi-Newton methods (BFGS).
    - D. Newton's Method ($\mathbf{	heta} - H^{-1}
abla L$).

    ??? info "See Answer"
        **Correct: D**

        *(The adaptive scaling term $\frac{1}{\sqrt{v_{	ext{hat}}}}$ serves as a computationally cheap, diagonal approximation of the inverse Hessian $H^{-1}$, which performs the ideal geometric rescaling.)*

---

!!! note "Quiz"
    **12. In the energy view of optimization, the total energy or "Hamiltonian" ($\mathcal{H}$) of a system with momentum is the sum of:**

    - A. The loss and the learning rate.
    - B. The potential energy (the loss $L(\mathbf{	heta})$) and the kinetic energy ($\frac{1}{2m}|\mathbf{p}|^2$).
    - C. The gradient and the Hessian.
    - D. The bias and the variance.

    ??? info "See Answer"
        **Correct: B**

        *(This Hamiltonian view allows us to analyze the stability and convergence of the optimization as a physical energy-dissipating system.)*

---

!!! note "Quiz"
    **13. For a damped dynamical system like Momentum or Adam, what property of the total energy $\mathcal{H}$ guarantees stability and convergence?**

    - A. The energy is always increasing.
    - B. The energy is conserved.
    - C. The energy is continuously dissipated ($d\mathcal{H}/dt \le 0$) due to the friction/damping term.
    - D. The energy fluctuates randomly.

    ??? info "See Answer"
        **Correct: C**

        *(The damping term $-\gamma|\mathbf{v}|^2$ ensures that the total energy of the system can never increase, forcing it to settle into a low-energy minimum.)*

---

!!! note "Quiz"
    **14. While Momentum helps smooth out zigzags, what is its primary limitation that adaptive methods solve?**

    - A. It is computationally too expensive.
    - B. It can only be used on small models.
    - C. It still uses a single, global learning rate $\eta$ that is constrained by the landscape's anisotropy.
    - D. It does not work with stochastic gradients.

    ??? info "See Answer"
        **Correct: C**

        *(Momentum improves the path but doesn't fix the underlying geometric problem that a single learning rate is a poor fit for an anisotropic landscape. Adaptive methods solve this by giving each parameter its own effective learning rate.)*

---

!!! note "Quiz"
    **15. In the Adam update rule, which term is responsible for providing the direction and inertia, analogous to a "navigator"?**

    - A. The bias-corrected first moment, $m_{	ext{hat}}$.
    - B. The bias-corrected second moment, $v_{	ext{hat}}$.
    - C. The learning rate $\eta$.
    - D. The epsilon term $\epsilon$.

    ??? info "See Answer"
        **Correct: A**

        *($m_{	ext{hat}}$ is the moving average of the gradients, representing the accumulated momentum and pointing in the general direction of descent.)*

---

!!! note "Quiz"
    **16. In the Adam update rule, which term is responsible for providing the adaptive damping or friction, analogous to a "damper"?**

    - A. The bias-corrected first moment, $m_{	ext{hat}}$.
    - B. The inverse square root of the second moment, $1/\sqrt{v_{	ext{hat}}}$.
    - C. The learning rate $\eta$.
    - D. The momentum coefficient $\beta_1$.

    ??? info "See Answer"
        **Correct: B**

        *($1/\sqrt{v_{	ext{hat}}}$ scales the update, applying strong friction in steep directions and light friction in flat directions.)*

---

!!! note "Quiz"
    **17. The geometric re-scaling performed by adaptive methods is closely related to which concept from information geometry?**

    - A. The Cramer-Rao bound.
    - B. The Kullback-Leibler divergence.
    - C. Natural Gradient Descent.
    - D. The Law of Large Numbers.

    ??? info "See Answer"
        **Correct: C**

        *(Natural Gradient Descent uses the Fisher Information Matrix to perform the ideal geometric step. Adaptive methods provide a computationally cheap, diagonal approximation of this principle.)*

---

!!! note "Quiz"
    **18. When comparing optimizer trajectories on a ravine function, which method is expected to follow the most direct path to the minimum by transforming the ravine into an isotropic bowl?**

    - A. Standard Gradient Descent (GD).
    - B. GD with Momentum.
    - C. Adam.
    - D. A random search.

    ??? info "See Answer"
        **Correct: C**

        *(Adam's adaptive scaling is specifically designed to counteract the anisotropy of the ravine, resulting in the most direct trajectory.)*

---

!!! note "Quiz"
    **19. What is a potential downside of Adam, particularly concerning the types of minima it finds?**

    - A. It is extremely slow to converge.
    - B. It requires manual tuning of many hyperparameters.
    - C. It may converge to sharp minima that generalize less well than the flat minima found by well-tuned SGD with momentum.
    - D. It uses too much memory.

    ??? info "See Answer"
        **Correct: C**

        *(While excellent at finding low-loss regions quickly, the adaptive nature of Adam can sometimes cause it to settle in sharp, less robust minima compared to non-adaptive methods that are more influenced by the landscape's broader structure.)*

---

!!! note "Quiz"
    **20. The physical analogy for Nesterov Accelerated Gradient (NAG) is a skier who:**

    - A. Skis directly down the fall line without turning.
    - B. Reacts to a turn only after entering it, causing them to drift wide.
    - C. Anticipates an upcoming turn and begins to lean into it *before* reaching it, following a tighter line.
    - D. Stops at every bump.

    ??? info "See Answer"
        **Correct: C**

        *(The "look-ahead" gradient calculation allows NAG to correct its momentum vector predictively, leading to a more efficient path on the curved loss manifold.)*

---

!!! note "Quiz"
    **21. If you set the momentum parameter $\beta$ to 0 in the Momentum update rule, what algorithm do you recover?**

    - A. Adam.
    - B. Standard Gradient Descent.
    - C. Newton's Method.
    - D. RMSProp.

    ??? info "See Answer"
        **Correct: B**

        *(If $\beta=0$, the velocity term $\mathbf{v}_{t+1}$ is simply $-\eta 
abla L(\mathbf{	heta}_t)$, and the update becomes $\mathbf{	heta}_{t+1} = \mathbf{	heta}_t - \eta 
abla L(\mathbf{	heta}_t)$, which is the definition of GD.)*

---

!!! note "Quiz"
    **22. Why is the term $\epsilon$ (e.g., $10^{-8}$) added to the denominator in the Adam and RMSProp update rules?**

    - A. To increase the learning rate.
    - B. To ensure numerical stability by preventing division by zero.
    - C. To add momentum to the update.
    - D. To correct for bias.

    ??? info "See Answer"
        **Correct: B**

        *(If a parameter's gradient has been zero for a long time, its accumulated squared gradient $v_t$ could be zero. Epsilon prevents a fatal division-by-zero error.)*

---

!!! note "Quiz"
    **23. The "condition number" $\kappa = \lambda_{\max}/\lambda_{\min}$ of the Hessian quantifies what property of the loss landscape?**

    - A. The number of local minima.
    - B. The degree of anisotropy (how much steeper the steepest direction is than the flattest direction).
    - C. The height of the energy barriers.
    - D. The overall volume of the landscape.

    ??? info "See Answer"
        **Correct: B**

        *(A high condition number indicates a landscape with long, narrow ravines, which is the primary challenge that advanced optimizers are designed to solve.)*

---

!!! note "Quiz"
    **24. In the damped harmonic oscillator model, $m\ddot{\mathbf{	heta}} = -\gamma m\dot{\mathbf{	heta}} - 
abla L(\mathbf{	heta})$, which term represents the damping force or friction?**

    - A. $m\ddot{\mathbf{	heta}}$ (Inertia).
    - B. $-
abla L(\mathbf{	heta})$ (Potential Force).
    - C. $-\gamma m\dot{\mathbf{	heta}}$ (Damping Force).
    - D. The mass $m$.

    ??? info "See Answer"
        **Correct: C**

        *(This term is proportional to the velocity $\dot{\mathbf{	heta}}$ and acts to dissipate energy from the system, ensuring it eventually settles in a minimum.)*

---

!!! note "Quiz"
    **25. The empirical finding that stochastic noise (from SGD) combined with adaptive dynamics preferentially drives the system toward what kind of minima is a key insight for generalization?**

    - A. The deepest possible minima, regardless of shape.
    - B. Sharp, narrow minima that perfectly fit the training data.
    - C. Flat, wide minima that are more robust to perturbations.
    - D. The minima closest to the initialization point.

    ??? info "See Answer"
        **Correct: C**

        *(Flat, wide basins are thermodynamically more stable and correspond to solutions that are less sensitive to small changes in the input data, leading to better generalization performance.)*


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


