# **Chapter 2: 2. Statistics & Probability in High Dimensions**

# **Introduction**

Chapter 1 established a **geometric** framework for understanding simulation data: trajectories in phase space were transformed into point clouds in feature space, characterized by their center (the mean vector $\mathbf{\mu}$) and shape (the covariance matrix $\Sigma$). This geometric picture, however, is incomplete. It describes the *location* and *orientation* of the data manifold but tells us nothing about the *density* of points populating that manifold—whether the cloud is uniformly distributed, clustered into distinct phases, or concentrated along narrow pathways. To answer these questions, we must transition from **geometry** (the shape of data) to **probability** (the distribution that generates it).

This chapter provides the formal statistical and information-theoretic toolkit required to model and analyze high-dimensional probability distributions $p(\mathbf{x})$. We begin by reinterpreting our data cloud as an **empirical distribution** and establish the profound connection between probability and energy through the **Boltzmann distribution**, linking statistical inference to the physical concept of free-energy landscapes. We then confront the central obstacle to all high-dimensional analysis: the **Curse of Dimensionality**—the exponential explosion of volume and the catastrophic failure of naive density estimation methods in spaces where $D \gg 3$. To overcome this, we develop the theory of **sampling-based inference**, introducing **Monte Carlo methods**, **importance sampling**, and the foundational **Markov Chain Monte Carlo (MCMC)** framework. We formalize the concepts of **entropy**, **mutual information**, and **KL divergence** as quantitative measures of uncertainty, dependence, and distributional mismatch—tools essential for discovering order parameters and validating models.

By the end of this chapter, you will understand how to move from raw simulation trajectories to probabilistic models, compute expectation values via sampling when analytical integration is impossible, and quantify the information content and uncertainty inherent in high-dimensional data. You will recognize why the Gaussian distribution emerges as the maximum-entropy solution subject to known mean and covariance constraints, and why density estimation in the full $\mathbb{R}^D$ space is futile without first applying dimensionality reduction. These foundations prepare you for **Chapter 3**, where we develop the algorithms—**PCA**, **t-SNE**, **UMAP**, and **autoencoders**—that project high-dimensional manifolds into computationally tractable low-dimensional representations, enabling the statistical methods of this chapter to finally work.

---

# **Chapter 2: Outline**

| **Sec.** | **Title**                                                 | **Core Ideas & Examples**                                                                                                                                                                                      |
| -------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2.1**  | **From Geometry to Probability**                          | Empirical distribution $\hat{p}(\mathbf{x}) = \frac{1}{N}\sum \delta(\mathbf{x}-\mathbf{x}_i)$; Boltzmann distribution $p(\mathbf{s}) \propto e^{-E/k_BT}$; energy-probability duality $E_{\text{eff}} = -k_BT \ln p(\mathbf{x})$; likelihood as inverse energy. |
| **2.2**  | **Probability Distributions as Energy Landscapes**        | Exponential family and partition function $Z(\mathbf{\theta})$; moments (mean, covariance, skewness, kurtosis) and cumulants; Shannon entropy $S[p] = -\langle \ln p \rangle$; KL divergence $D_{\mathrm{KL}}(p||q)$ as distributional distance. |
| **2.3**  | **Likelihood and Inference**                              | Maximum Likelihood Estimation (MLE) $\mathbf{\theta}_{\text{ML}} = \arg\max \ln \mathcal{L}(\mathbf{\theta})$; Bayesian inference and MAP $\mathbf{\theta}_{\text{MAP}} = \arg\max [\ln \mathcal{L} + \ln p(\mathbf{\theta})]$; $\chi^2$ fitting as MLE under Gaussian noise; Fisher Information Matrix as metric tensor. |
| **2.4**  | **The Curse of Dimensionality**                           | Volume concentration in high-dimensional spheres; shell fraction $1-(1-\epsilon)^D \to 1$ as $D \to \infty$; distance concentration (all pairwise distances converge); data sparsity ($10^D$ samples required); failure of naive histograms and nearest-neighbor methods. |
| **2.5**  | **Sampling Strategies**                                   | Monte Carlo integration $\langle f \rangle \approx \frac{1}{N}\sum f(\mathbf{x}_i)$; importance sampling with weights $w = p/q$; Markov Chain Monte Carlo (MCMC) as correlated sampling; Metropolis-Hastings acceptance $\min(1, e^{-\beta \Delta E})$; variance reduction techniques. |
| **2.6**  | **Density Estimation**                                    | Parametric models (single Gaussian, Gaussian Mixture Models); nonparametric methods (Kernel Density Estimation with bandwidth $h$, K-Nearest Neighbors); bias-variance trade-off; curse revisited—KDE fails when $D$ large; workflow: reduce dimension first, estimate density in latent space. |
| **2.7**  | **Entropy, Information, and Uncertainty**                 | Shannon entropy as Gibbs entropy $S = -k_B \sum p_i \ln p_i$; Maximum Entropy Principle (MaxEnt) justifies Boltzmann and Gaussian distributions; mutual information $I(X;Y) = S(X) - S(X|Y)$ quantifies dependence; discovering order parameters via $\max I(O; \text{Label})$. |
| **2.8**  | **Worked Example: Sampling a High-Dimensional Gaussian**  | Ground truth $\mathcal{N}(\mathbf{0}, \Sigma_{\text{true}})$; projection effects hide correlations; empirical covariance $\hat{\Sigma}$ estimation; singularity when $N < D$; sample size requirement $N \gg D$ for stable inference; Random Matrix Theory regime. |
| **2.9**  | **Code Demo: Monte Carlo Density Estimation**             | 2D correlated Gaussian as ground truth; draw $N=5000$ samples; Kernel Density Estimation (KDE) with automatic bandwidth; visualization of learned density $\hat{p}(\mathbf{x})$ via contour plot; bridge to Chapter 3: need dimensionality reduction for $D \gg 2$. |

---

## **2.1 From Geometry to Probability**

---

### **Why Statistics?**

In Chapter 1, we established a **geometric** picture of our simulation data. We learned to transform raw, high-dimensional trajectories into a static data cloud, or manifold, $\mathcal{M} \subset \mathbb{R}^D$. We used the mean vector $\mathbf{\mu}$ to locate its center and the covariance matrix $\Sigma$ to describe its linear shape, orientation, and dominant modes of variation.

This geometric summary, however, is incomplete. It provides the first and second moments but tells us little about the *density* of the cloud. Is the cloud solid, hollow, or clustered in multiple 'blobs'? To answer this, we must move from geometry (the *shape* of the data) to **probability** (the *distribution* that populates that shape). Our goal is no longer just to describe the data, but to infer the **generative process** $P(\mathbf{x})$ from which our samples $\{\mathbf{x}_i\}$ were drawn.

!!! tip "From Shape to Distribution"
```
The mean and covariance tell us *where* the data is and *how it's oriented*, but not *how densely* it's packed. Probability density is the missing piece that transforms geometric description into predictive modeling.

```
---

### **The Empirical Distribution**

Our entire knowledge of the true, underlying distribution $P(\mathbf{x})$ is contained within the $N$ finite samples from our simulation. The most direct, unbiased representation of this knowledge is the **empirical distribution**, $\hat{p}(\mathbf{x})$.

We define $\hat{p}(\mathbf{x})$ as a sum of Dirac delta functions, one centered on each data point we observed:

$$
\hat{p}(\mathbf{x}) \approx \frac{1}{N}\sum_{i=1}^{N}\delta(\mathbf{x}-\mathbf{x}_i)
$$

This function places a "probability mass" of $1/N$ at the precise location of each sample and is zero everywhere else [1]. It is a "spiky," non-parametric model of the underlying reality. Any expectation value $\langle f(\mathbf{x}) \rangle$ computed under this empirical distribution is simply the sample mean of $f$ over our dataset:

$$
\langle f(\mathbf{x}) \rangle_{\hat{p}} = \int f(\mathbf{x})\hat{p}(\mathbf{x})d\mathbf{x} = \frac{1}{N}\sum_{i=1}^{N}f(\mathbf{x}_i)
$$
This is the formal bridge from our data samples to the language of probability. Our data cloud *is* the empirical distribution.

---

### **Physical Analogy: Energy and Log-Likelihood**

In statistical physics, the probability of a system being in a microstate $\mathbf{s}$ at thermal equilibrium is not uniform. It is governed by the **Boltzmann distribution** [2]:

$$
p(\mathbf{s}) = \frac{1}{Z} e^{-E[\mathbf{s}]/k_B T}
$$
Here, $E[\mathbf{s}]$ is the energy of the state, $k_B T$ is the thermal energy, and $Z$ is the partition function that ensures the distribution sums to one.

This gives us a profound conceptual link:
* High-probability states $\leftrightarrow$ Low-energy states (basins, stable phases)
* Low-probability states $\leftrightarrow$ High-energy states (barriers, transition states)

We can invert this relationship to *define* an **effective energy** for *any* probability distribution:

$$
E_{\text{eff}}(\mathbf{x}) \equiv -k_B T \ln p(\mathbf{x})
$$

This reframes our entire task. The goal of "learning a distribution" $p(\mathbf{x})$ from data is equivalent to the physical goal of "finding the energy landscape" $E_{\text{eff}}(\mathbf{x})$ that would produce that distribution.

!!! example "Energy-Probability Duality"
```
In the Ising model at low temperature, the two ground states (all spins up or all spins down) have $E = -JN$ and dominate the probability $p(\mathbf{s}) \propto e^{+\beta JN}$. High-energy configurations with mixed spins are exponentially suppressed. This direct mapping between energy minima and probability maxima is the foundation of statistical mechanics.

```
This also connects directly to the core concept of **log-likelihood** in statistics. The log-likelihood of observing a state $\mathbf{x}$ is $\ln p(\mathbf{x})$. Therefore, **maximizing the log-likelihood is equivalent to minimizing the energy**. This duality—between statistics and physics, between likelihood and energy—is the central theme of this volume.

---

### **Goal**

Our goal in this chapter is to build a formal toolkit for modeling $p(\mathbf{x})$. We must move beyond the simple empirical distribution $\hat{p}(\mathbf{x})$ and the low-order moments of Chapter 1. We need methods to extract statistical regularities, quantify uncertainty, and build parametric models of the underlying energy landscape, all while confronting the profound mathematical challenges that arise when our dimension $D$ is large.

---

## **2.2 Probability Distributions as Energy Landscapes**

The analogy from Section 2.1—where probability is inversely related to an effective energy—is one of the most powerful organizing principles in science. We can formalize this by defining our probability models as members of the **exponential family**, a broad class of distributions that includes the Gaussian, Poisson, Bernoulli, and, most importantly, the Boltzmann distribution [1].

---

### **The Exponential Family and the Partition Function**

A general (unnormalized) energy-based model defines the probability $p(\mathbf{x})$ of a state $\mathbf{x}$ using a set of parameters $\mathbf{\theta}$ and an energy function $E(\mathbf{x};\mathbf{\theta})$:

$$
p(\mathbf{x}) = \frac{1}{Z(\mathbf{\theta})}\exp[-E(\mathbf{x};\mathbf{\theta})]
$$

This form explicitly bridges statistics and physics [1]:

* $E(\mathbf{x};\mathbf{\theta})$ is the **energy function** (or "Hamiltonian"). It assigns a low energy to "desirable" or probable states and a high energy to "undesirable" or improbable states. The parameters $\mathbf{\theta}$ (e.g., coupling constants $J_{ij}$ in an Ising model) define the landscape.
* $Z(\mathbf{\theta})$ is the **partition function**, a normalization constant that ensures the distribution integrates to one. It is the sum (or integral) over all possible states:

$$
Z(\mathbf{\theta}) = \int \exp[-E(\mathbf{x};\mathbf{\theta})] d\mathbf{x}
$$

This single equation, $p \propto e^{-E}$, is the core of statistical mechanics, Bayesian inference, and energy-based machine learning. The central challenge of these fields is often the intractability of $Z(\mathbf{\theta})$, which requires summing over an exponentially large state space.

??? question "Why is the partition function so hard to compute?"
```
For a system with $N$ binary variables (like spins), there are $2^N$ possible states. Computing $Z$ requires summing $\exp[-E(\mathbf{x})]$ over all of them. For just $N=300$ spins, this is $2^{300} \approx 10^{90}$ terms—more than the number of atoms in the universe. This exponential complexity makes exact calculation impossible, forcing us to use approximations (mean-field theory) or sampling methods (MCMC).

```
---

### **Moments and Cumulants**

A probability distribution can be characterized by its **moments**, which describe its shape.

* **1st Moment (Mean):**

$$
\mathbf{\mu} = \langle \mathbf{x} \rangle = \int \mathbf{x} p(\mathbf{x}) d\mathbf{x}
$$

This is the center of mass.

* **2nd Central Moment (Covariance):**

$$
\Sigma_{jk} = \langle (x_j - \mu_j)(x_k - \mu_k) \rangle
$$

This is the $D \times D$ matrix (from Chapter 1) that describes the linear correlations and spread.

Higher-order moments capture more subtle, non-Gaussian features:

* **3rd Moment (Skewness):** Measures the asymmetry of the distribution.
* **4th Moment (Kurtosis):** Measures the "tailedness" or "peakedness" of the distribution compared to a Gaussian.

**Cumulants** are a closely related set of quantities (the $n$-th cumulant is a polynomial of the first $n$ moments) that are often more fundamental. For example, the first cumulant is the mean, and the second is the variance. A key property is that for a Gaussian distribution, all cumulants of order greater than two are exactly zero. Thus, the deviation of higher-order cumulants from zero provides a direct measure of non-Gaussianity.

---

### **Entropy: The Measure of Uncertainty**

While moments describe shape, **entropy** measures the total "volume," "spread," or "uncertainty" of a distribution. The **Shannon (or differential) entropy** $S[p]$ is defined as the expected value of the negative log-probability [3]:

$$
S[p] = -\int p(\mathbf{x}) \ln p(\mathbf{x}) d\mathbf{x} = \langle -\ln p(\mathbf{x}) \rangle_p
$$

This quantity is the direct information-theoretic analogue of entropy in statistical mechanics [2].

* A "delta function" distribution (perfect certainty) has $S = -\infty$ (or $S=0$ in the discrete case).
* A broad, flat, uniform distribution (total uncertainty) has the maximum possible entropy for a given domain.

Entropy provides a way to quantify what we *don't* know. The principle of maximum entropy states that the most unbiased model $p(\mathbf{x})$ consistent with a set of constraints (e.g., a known mean and variance) is the one that maximizes $S[p]$.

---

### **Relative Entropy (KL Divergence): The Measure of Difference**

Finally, we need a way to measure the "distance" between two distributions—for example, the distance between our model's guess $q(\mathbf{x})$ and the true data distribution $p(\mathbf{x})$.

This is given by the **relative entropy**, or **Kullback-Leibler (KL) divergence** [5]:

$$
D_{\mathrm{KL}}(p||q) = \int p(\mathbf{x})\ln\frac{p(\mathbf{x})}{q(\mathbf{x})}d\mathbf{x}
$$

The KL divergence measures the expected "information gain" or "surprise" when one revises their beliefs from a prior distribution $q(\mathbf{x})$ to a posterior distribution $p(\mathbf{x})$.

  * $D_{\mathrm{KL}}(p||q) \ge 0$ for all $p$ and $q$.
  * $D_{\mathrm{KL}}(p||q) = 0$ if and only if $p(\mathbf{x}) = q(\mathbf{x})$.

Critically, the KL divergence is **not a true distance metric** because it is asymmetric: $D_{\mathrm{KL}}(p||q) \neq D_{\mathrm{KL}}(q||p)$. This asymmetry is important, but for our purposes, $D_{\mathrm{KL}}$ serves as our primary tool for measuring how "wrong" our model $q$ is with respect to the truth $p$. Much of modern machine learning, particularly variational inference, is formulated as an optimization problem that seeks to minimize this divergence.

---

## **2.3 Likelihood and Inference**

Given a set of $N$ i.i.d. (independent and identically distributed) data samples $X = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ and a parametric model $p(\mathbf{x}|\mathbf{\theta})$ that defines an energy landscape (as in 2.2), how do we "fit" the model to the data? That is, how do we select the parameters $\mathbf{\theta}$ that best explain our observations? This is the central question of statistical inference.

---

### **The Likelihood Principle**

The most common approach in classical statistics is founded on the **likelihood principle**. The likelihood function, $\mathcal{L}(\mathbf{\theta})$, is not a probability distribution over $\mathbf{\theta}$. Instead, it is the probability of having observed the *specific data* $X$ *given* a particular choice of parameters $\mathbf{\theta}$.

Assuming our $N$ samples are independent, the total probability of the dataset is the product of the probabilities of each sample:

$$
\mathcal{L}(\mathbf{\theta}) = p(X|\mathbf{\theta}) = \prod_{i=1}^N p(\mathbf{x}_i|\mathbf{\theta})
$$

The **Maximum Likelihood Estimate (MLE)** is the value of $\mathbf{\theta}$ that maximizes this function. This is the set of parameters $\mathbf{\theta}_{\text{ML}}$ under which our observed data was most probable.

In practice, this product of small probabilities is numerically unstable. We instead maximize the **log-likelihood**, which converts the product into a more stable and convenient sum:

$$
\ln \mathcal{L}(\mathbf{\theta}) = \sum_{i=1}^N \ln p(\mathbf{x}_i|\mathbf{\theta})
$$

Since the logarithm is a monotonic function, $\arg\max \mathcal{L}(\mathbf{\theta}) = \arg\max \ln \mathcal{L}(\mathbf{\theta})$. This converts a difficult product-maximization problem into an equivalent, and much simpler, sum-maximization problem.

---

### **Maximum Likelihood vs. Bayesian Inference**

The MLE approach provides a single point estimate, $\mathbf{\theta}_{\text{ML}}$. The Bayesian framework offers a more complete, probabilistic view by incorporating *prior knowledge* $p(\mathbf{\theta})$, which represents our beliefs about the parameters *before* seeing any data.

Using **Bayes' theorem**, we compute the **posterior distribution** $p(\mathbf{\theta}|X)$, which is our updated belief about $\mathbf{\theta}$ *after* observing the data:

$$
p(\mathbf{\theta}|X) = \frac{p(X|\mathbf{\theta}) p(\mathbf{\theta})}{p(X)} \propto \mathcal{L}(\mathbf{\theta}) p(\mathbf{\theta})
$$

This is often stated as: **Posterior $\propto$ Likelihood $\times$ Prior**.

Instead of just maximizing the likelihood, a Bayesian may seek the **Maximum A Posteriori (MAP)** estimate, which is the peak (or mode) of the posterior distribution:

$$
\mathbf{\theta}_{\text{MAP}} = \arg\max_{\mathbf{\theta}} p(\mathbf{\theta}|X) = \arg\max_{\mathbf{\theta}} [\ln \mathcal{L}(\mathbf{\theta}) + \ln p(\mathbf{\theta})]
$$

This reveals a crucial connection: the MAP estimate is identical to the MLE, but with an added **regularization term** $\ln p(\mathbf{\theta})$. For example, choosing a Gaussian prior $p(\mathbf{\theta}) \propto \exp(-\alpha ||\mathbf{\theta}||^2)$ is equivalent to adding an $L^2$ penalty (ridge regression) to the log-likelihood objective. MLE can thus be seen as the special case of MAP inference where the prior $p(\mathbf{\theta})$ is uniform (or "flat").

---

### **Connection to Physical Fitting ($\chi^2$)**

The principle of maximum likelihood provides the formal justification for one of the most common practices in the physical sciences: **least-squares fitting**.

Assume we are fitting a set of data points $(y_i, x_i)$ to a model $f(x_i; \mathbf{\theta})$. Let's assume that our measurements $y_i$ are corrupted by independent Gaussian noise with a known standard deviation $\sigma_i$. Our model for the data is therefore:

$$
p(y_i | x_i, \mathbf{\theta}) = \mathcal{N}(y_i | f(x_i; \mathbf{\theta}), \sigma_i^2) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left[ -\frac{(y_i - f(x_i; \mathbf{\theta}))^2}{2\sigma_i^2} \right]
$$

The total log-likelihood for all $N$ data points is the sum:

$$
\ln \mathcal{L}(\mathbf{\theta}) = \sum_{i=1}^N \ln p(y_i | x_i, \mathbf{\theta}) = \sum_{i=1}^N \left( -\frac{1}{2} \ln(2\pi\sigma_i^2) -\frac{(y_i - f(x_i; \mathbf{\theta}))^2}{2\sigma_i^2} \right)
$$

To maximize $\ln \mathcal{L}(\mathbf{\theta})$, we only need to consider the terms that depend on $\mathbf{\theta}$. This is equivalent to *minimizing* the negative log-likelihood:

$$
-\ln \mathcal{L}(\mathbf{\theta}) = \frac{1}{2} \sum_{i=1}^N \frac{(y_i - f(x_i; \mathbf{\theta}))^2}{\sigma_i^2} + \text{Constant}
$$

This is precisely the definition of the **chi-squared ($\chi^2$) statistic** [7]. Therefore, **minimizing $\chi^2$ is identical to finding the maximum likelihood estimate of $\mathbf{\theta}$ under the assumption of independent Gaussian noise**.

---

### **Information Geometry**

The log-likelihood function $\ln \mathcal{L}(\mathbf{\theta})$ can be viewed as a "landscape" over the space of parameters. The MLE is the peak of this landscape. The *sharpness* of this peak tells us how much "information" the data provides about $\mathbf{\theta}$. A sharp peak (high curvature) implies high certainty, while a flat peak (low curvature) implies high uncertainty.

This concept is formalized by the **Fisher Information Matrix**, $I(\mathbf{\theta})$, which is defined as the expected value of the Hessian (matrix of second derivatives) of the *negative* log-likelihood:

$$
I(\mathbf{\theta})_{jk} = -\left\langle \frac{\partial^2 \ln p(\mathbf{x}|\mathbf{\theta})}{\partial \theta_j \partial \theta_k} \right\rangle_{p(\mathbf{x}|\mathbf{\theta})}
$$

The Fisher Information Matrix acts as a **metric tensor** $g_{jk}$ on the manifold of parameters. It defines a "distance" between different models, $ds^2 = \sum_{jk} I(\mathbf{\theta})_{jk} d\theta_j d\theta_k$. This field, known as **information geometry**, provides a powerful framework for understanding the space of statistical models, where the natural "distance" between models is not their Euclidean parameter difference but the *statistical distinguishability* of the data they generate [1].

---

## **2.4 The Curse of Dimensionality**

In low-dimensional spaces like $\mathbb{R}^2$ or $\mathbb{R}^3$, our geometric intuition is a reliable guide. We can visualize data clouds, understand "closeness," and build histograms to estimate density. In the high-dimensional spaces $\mathbb{R}^D$ where our simulation data lives ($D \gg 3$), this intuition fails spectacularly. This failure is famously known as the **"Curse of Dimensionality,"** a term coined by Richard Bellman [6] to describe the exponential scaling of problems in high dimensions.

---

### **The Volume Paradox**

The most counter-intuitive property of high-dimensional spaces concerns volume. Consider a $D$-dimensional hypersphere (or "ball") of radius $r$. Its volume scales as:

$$
\text{Vol}(r) \propto r^D
$$

Now, let's examine where this volume is located. Consider a hypersphere of radius $r=1$ and the "shell" of data that lies between $r=1-\epsilon$ and $r=1$. The volume of this thin, outer shell is:

$$
\frac{\text{Vol}_{\text{shell}}}{\text{Vol}_{\text{total}}} = \frac{\text{Vol}(r=1) - \text{Vol}(r=1-\epsilon)}{\text{Vol}(r=1)} = \frac{1^D - (1-\epsilon)^D}{1^D} = 1 - (1-\epsilon)^D
$$
In three dimensions, for $\epsilon=0.01$, this ratio is tiny: $1 - (0.99)^3 \approx 0.03$. Most of the volume is in the core.
In high dimensions, this trend reverses completely. For $D=10,000$ and $\epsilon=0.01$:

$$
1 - (0.99)^{10000} \approx 1 - (2.6 \times 10^{-44}) \to 1
$$

This is a stunning result: **virtually all of the volume of a high-dimensional sphere is concentrated in an infinitesimally thin shell near its surface.** The "core" of the sphere is, for all practical purposes, empty.

!!! example "Volume Concentration in High Dimensions"
```
For a 10,000-dimensional unit hypersphere, 99.99% of the volume lies in a shell between radius 0.99 and 1.0—a shell that is only 1% thick. This is why naive sampling fails: uniformly distributed points would almost never land in this critical shell where the probability mass is concentrated.

```
---

### **Consequences for Data**

This "volume paradox" has two devastating consequences for our data analysis:

* **Distance Concentration:** In high-dimensional space, the concept of a "nearest neighbor" becomes meaningless. As $D$ increases, the pairwise distances between any two randomly chosen points $\mathbf{x}_i$ and $\mathbf{x}_j$ all converge to the same value. The difference between the maximum and minimum distance in a dataset, relative to the average distance, shrinks to zero [8]. This means that **all points become "equally far" from each other**. Algorithms like K-nearest-neighbors, which rely on a meaningful concept of "closeness," fail.

* **Data Sparsity:** Our data cloud, no matter how large, is an infinitesimally sparse mist in $\mathbb{R}^D$. If we need 10 samples to get a reasonable density estimate on a 1D line, we would need $10^D$ samples to achieve the same resolution in $D$ dimensions. With $D=100$, this is more than the number of atoms in the universe. Any new point we wish to analyze will, with near-certainty, fall into a region of the space where we have *never* seen a single data point.

---

### **Implication: The Failure of Naïve Methods**

The curse of dimensionality dictates that **naïve density estimation (like histograms or kernels) is impossible in high dimensions.** Any attempt to "bin" the space will result in a near-infinite number of empty bins.

This is not just a technical challenge; it is a fundamental one. It tells us that we *cannot* hope to model the full distribution $p(\mathbf{x})$ in $\mathbb{R}^D$. The only reason we can make progress at all is the **Manifold Hypothesis** (Chapter 1): our data is not truly $D$-dimensional. It is confined to a much lower-dimensional manifold $\mathcal{M}$ embedded within the larger space.

This realization is the primary motivation for **Chapter 3: Dimensionality Reduction & Clustering**. Before we can perform statistical inference, we must first find a lower-dimensional representation of the data that "unrolls" this manifold and makes the data dense enough for statistical methods to work.

---

## **2.5 Sampling Strategies**

The "Curse of Dimensionality" (Section 2.4) establishes a formidable challenge: we cannot analytically integrate over $p(\mathbf{x})$ in $\mathbb{R}^D$, nor can we build a simple histogram to approximate it. Our only path forward is to compute expectation values $\langle f \rangle = \int f(\mathbf{x})p(\mathbf{x})d\mathbf{x}$ by **sampling**.

However, naïve sampling (e.g., from a uniform distribution) fails. If $p(\mathbf{x})$ is a low-temperature Boltzmann distribution, nearly all uniform samples will land in high-energy regions where $p(\mathbf{x}) \approx 0$, contributing nothing to the integral. The variance of such an estimator would be astronomically high. We must, therefore, use smarter strategies that *focus* our sampling on the regions where $p(\mathbf{x})$ is large—the low-energy basins.

---

### **Monte Carlo Sampling**

The general term **Monte Carlo sampling** refers to any method that estimates an integral by taking random samples. The "simple" Monte Carlo integral $\frac{1}{N}\sum f(\mathbf{x}_i)$ is the most basic form. The key to making this work in high dimensions is to *not* sample from a simple, uniform distribution, but to draw samples directly from the complex target distribution $p(\mathbf{x})$ itself. If we could magically generate $N$ independent samples $\{\mathbf{x}_i\}$ drawn from $p(\mathbf{x})$, our estimator would simply be the sample mean:

$$
\langle f \rangle_p \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}_i)
$$

This is the ideal scenario. The challenge is: how do we draw samples from a complex, high-dimensional $p(\mathbf{x})$ that we only know up to a (likely intractable) normalization constant $Z$?

---

### **Importance Sampling**

**Importance Sampling** is a strategy for sampling from a *different*, simpler **proposal distribution** $q(\mathbf{x})$ that we *can* draw from, and then correcting for the mismatch. We rewrite the expectation integral as:

$$
\langle f \rangle_p = \int f(\mathbf{x})p(\mathbf{x})d\mathbf{x} = \int f(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})}q(\mathbf{x})d\mathbf{x} = \left\langle f(\mathbf{x}) \frac{p(\mathbf{x})}{q(\mathbf{x})} \right\rangle_q
$$

This gives us a new estimator. We draw $N$ samples $\{\mathbf{x}_i\}$ from $q(\mathbf{x})$ and compute:

$$
\langle f \rangle_p \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}_i)w(\mathbf{x}_i) \quad \text{where} \quad w(\mathbf{x}_i) = \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}
$$

The terms $w(\mathbf{x}_i)$ are the **importance weights** that correct for the fact that we sampled from $q$ instead of $p$. This method works perfectly if $q(\mathbf{x})$ is non-zero wherever $p(\mathbf{x})$ is non-zero. The variance of this estimator is minimized when the proposal $q(\mathbf{x})$ is chosen to be "close" to $p(\mathbf{x})$ [1]. The primary difficulty, especially in high dimensions, is finding a simple $q(\mathbf{x})$ that is also a good match for the unknown, complex $p(\mathbf{x})$.

---

### **Markov Chain Monte Carlo (MCMC)**

**MCMC** is the most powerful and widely used solution to this problem. Instead of trying to guess a good proposal $q(\mathbf{x})$ for independent samples, MCMC constructs a **correlated** sequence of samples using a **Markov chain**—a "random walk" whose rules are cleverly designed.

The core idea is to define a "transition probability" $T(\mathbf{x}'|\mathbf{x})$ for stepping from state $\mathbf{x}$ to state $\mathbf{x}'$. This transition rule is built (e.g., via the **Metropolis-Hastings** algorithm) to ensure that the stationary distribution of this random walk is exactly our target distribution $p(\mathbf{x})$ [9].

We "run" the chain by:
1. Starting at a random $\mathbf{x}_0$.
2. Iteratively applying the transition rule: $\mathbf{x}_{t+1} \sim T(\mathbf{x}_{t+1}|\mathbf{x}_t)$.

After an initial "burn-in" period, the states $\{\mathbf{x}_t\}$ visited by the walker are, by construction, distributed according to $p(\mathbf{x})$. We can then use these states in our simple sample mean estimator. The trade-off is that the samples are no longer independent; they are correlated in "time." This requires careful analysis of autocorrelation to determine the true statistical error, but it allows us to sample from distributions in thousands or millions of dimensions without ever knowing the partition function $Z$.

---

### **Other Variance Reduction Techniques**

While MCMC is the workhorse in statistical physics, other methods are used in statistics and data science to improve sampling efficiency:
* **Stratified Sampling:** We partition the state space into disjoint regions (strata) and sample a fixed number of points from each. This ensures that low-probability regions are not missed by chance, which can significantly reduce the variance of the estimate.
* **Low-Discrepancy Sampling (Quasi-Monte Carlo):** This method replaces pseudo-random numbers with deterministic, "low-discrepancy" sequences (e.g., Sobol or Halton sequences). These sequences are designed to "fill" the high-dimensional space more uniformly than random points, often leading to a faster rate of convergence for the integral estimate [10].

These methods are powerful but are often difficult to apply to the irregular, high-dimensional energy landscapes of complex physical systems, making MCMC the dominant strategy.

---

## **2.6 Density Estimation**

The sampling methods in Section 2.5, particularly MCMC, allow us to compute expectation values $\langle f \rangle_p$ without ever writing down a functional form for $p(\mathbf{x})$. However, we often want the distribution itself. We want to turn our "spiky" empirical distribution (Section 2.1) into a smooth, analytical function $\hat{p}(\mathbf{x})$ that approximates the true $p(\mathbf{x})$. This process is called **density estimation**.

---

### **Parametric Approaches**

The most direct approach is to *assume* a functional form for $\hat{p}(\mathbf{x}|\mathbf{\theta})$ and then use the data to fit the parameters $\mathbf{\theta}$.

* **Single Gaussian:** The simplest assumption is that our data cloud is a single "hyper-ellipsoid." The model is $\hat{p}(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\mathbf{\mu}, \Sigma)$. In this case, the parameters $\mathbf{\theta} = (\mathbf{\mu}, \Sigma)$ are simply the sample mean and covariance we computed in Chapter 1.
* **Gaussian Mixture Model (GMM):** A far more flexible model, a GMM assumes the data is a weighted sum of $K$ different Gaussian "blobs."

$$
\hat{p}(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\mathbf{\mu}_k, \Sigma_k)
$$

This can capture multi-modal distributions, such as the distinct clusters corresponding to different physical phases (e.g., solid, liquid) in a simulation [1]. The parameters $(\pi_k, \mathbf{\mu}_k, \Sigma_k)$ are typically fit using the Expectation-Maximization (EM) algorithm.

The advantage of parametric models is that they are compact (defined by a small set of parameters). The disadvantage is their high **bias**; if the true distribution is not Gaussian-like, the model will be a poor fit regardless of how much data we have.

---

### **Nonparametric Approaches**

Nonparametric methods make minimal assumptions about the shape of $p(\mathbf{x})$. The model's complexity is allowed to grow as the number of data points $N$ increases.

* **Kernel Density Estimation (KDE):** This is the most intuitive "smoothing" of the empirical distribution $\hat{p}(\mathbf{x}) = \frac{1}{N}\sum \delta(\mathbf{x}-\mathbf{x}_i)$. Instead of placing an infinitely sharp "spike" (a delta function) on each data point, KDE places a smooth, local "bump" called a **kernel** $K$ (e.g., a small Gaussian) on each point. The final density estimate is the sum of all these bumps [11].

```
The model is:

```
$$
\hat{p}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N K_h(\mathbf{x} - \mathbf{x}_i)
$$

```
where $K_h$ is the kernel (e.g., a Gaussian) with a **bandwidth** $h$, which controls the "width" of the bumps. A small $h$ leads to a "spiky" (high-variance) estimate, while a large $h$ leads to an "oversmoothed" (high-bias) estimate.

```
* **K-Nearest Neighbors (KNN):** This method reverses the logic. Instead of fixing a bandwidth $h$ and counting the points inside, KNN fixes the number of points $k$ and measures the volume $V$ required to enclose them. The density at a point $\mathbf{x}$ is then estimated as:

$$
\hat{p}(\mathbf{x}) \approx \frac{k}{N \cdot V(\mathbf{x})}
$$

```
This is an *adaptive* method: in dense regions, $V$ will be small, giving a high density. In sparse regions, $V$ will be large, giving a low density.

```
---

### **The Curse, Revisited**

Nonparametric methods, which seem so powerful in 1D, are crippled by the **Curse of Dimensionality** (Section 2.4).
* In KDE, for a data point $\mathbf{x}_i$ to have any influence on $\hat{p}(\mathbf{x})$, the point $\mathbf{x}$ must lie within its kernel's radius (roughly $h$). But in high dimensions $D$, the volume of this kernel, which scales as $h^D$, is vanishingly small.
* To capture any data, $h$ must be made so large that the kernel covers a significant fraction of the entire data space. The estimate at $\mathbf{x}$ is no longer a "local" estimate but an average over a vast, non-local region. The bias becomes enormous.

This is a direct consequence of the **bias-variance trade-off**, which becomes exponentially more severe with increasing $D$.

---

### **Practical Viewpoint**

The futility of density estimation in high dimensions reinforces a central lesson: **we must not work in the full $\mathbb{R}^D$ space.**

The practical workflow is almost always to *first* apply dimensionality reduction (the topic of Chapter 3) to find the 1D, 2D, or 3D coordinates of the underlying manifold $\mathcal{M}$. We then perform density estimation (e.g., with KDE) in this low-dimensional, computationally-tractable latent space. This allows us to build a meaningful $\hat{p}(\mathbf{z})$ on the manifold, which is the "effective" distribution our physical system actually follows.

---

## **2.7 Entropy, Information, and Uncertainty**

In Section 2.2, we introduced entropy $S[p]$ as a measure of the "spread" of a distribution. Here, we formalize its connection to physical thermodynamics and introduce its relative, **mutual information**, to quantify the statistical dependence between variables. These information-theoretic tools provide a powerful lens for understanding and discovering physical order.

---

### **Shannon Entropy and Physical Entropy**

The **Shannon entropy** of a discrete distribution $p_i$ is a rigorous measure of the average "surprise" or "uncertainty" of an outcome [3]:

$$
S = -k_B \sum_i p_i \ln p_i
$$

This form is identical to the **Gibbs entropy** in statistical mechanics [2]. This is not a coincidence; it reflects a deep connection. The **Principle of Maximum Entropy** (MaxEnt) states that the most objective or "honest" distribution $p(\mathbf{x})$ that is consistent with a set of known constraints (e.g., a known average energy) is the one that maximizes $S[p]$ [4].

* **Example 1: MaxEnt and the Boltzmann Distribution.** The distribution that maximizes entropy subject to a *fixed average energy* $\langle E \rangle$ is precisely the Boltzmann distribution $p(\mathbf{x}) \propto e^{-\beta E(\mathbf{x})}$.
* **Example 2: MaxEnt and the Gaussian Distribution.** The distribution that maximizes entropy subject to a *fixed mean $\mathbf{\mu}$ and covariance $\Sigma$* is the multivariate Gaussian distribution $\mathcal{N}(\mathbf{\mu}, \Sigma)$.

This second example provides a profound justification for the importance of the Gaussian model: it is the "most random" or "least structured" distribution possible that matches the first and second moments (mean and covariance) we computed in Chapter 1.

---

### **Mutual Information: Measuring Dependence**

Entropy measures the uncertainty of a *single* variable. **Mutual Information (MI)** measures the "reduction in uncertainty" about one variable from knowing another. It quantifies the amount of information $X$ and $Y$ *share*.

Given two variables $X$ and $Y$ with a joint distribution $p(x,y)$ and marginals $p(x)$ and $p(y)$, their mutual information $I(X;Y)$ is:

$$
I(X;Y) = \sum_{x,y} p(x,y) \ln \left( \frac{p(x,y)}{p(x)p(y)} \right)
$$

* If $X$ and $Y$ are **independent**, then $p(x,y) = p(x)p(y)$, the log term is $\ln(1)=0$, and $I(X;Y) = 0$.
* If $X$ and $Y$ are **dependent**, $p(x,y) \neq p(x)p(y)$, and $I(X;Y) > 0$.

$I(X;Y)$ can also be expressed in terms of entropies, $I(X;Y) = S(X) - S(X|Y)$, which reads as "the uncertainty of $X$ minus the uncertainty of $X$ given $Y$." It is also equivalent to the KL divergence (Section 2.2) between the joint distribution and the product of its marginals: $I(X;Y) = D_{\mathrm{KL}}(p(x,y) || p(x)p(y))$.

??? question "When should you use mutual information vs. correlation?"
```
Correlation measures only *linear* relationships between variables and ranges from -1 to +1. Mutual information captures *any* statistical dependence (linear or nonlinear) and is always non-negative. For example, if $Y = X^2$ where $X$ is centered, the Pearson correlation is zero, but $I(X;Y)$ is large. Use MI when you suspect complex, nonlinear dependencies.

```
---

### **Application: Discovering Order Parameters**

Mutual information is not just a theoretical metric; it is a powerful computational tool for discovering hidden physical relationships. In a complex simulation (e.g., a phase transition), the raw state $\mathbf{x}$ (a $1000 \times 1000$ spin configuration) is too high-dimensional. We seek a low-dimensional **order parameter** $O(\mathbf{x})$ (like total magnetization, $M$) that effectively captures the state of the system.

But how do we find a good $O(\mathbf{x})$ if we don't already know it?

If we have labeled data (e.g., "Phase A" or "Phase B" from simulation temperature), we can find the optimal order parameter by *maximizing mutual information*. We search for a function $O(\mathbf{x})$ that maximizes $I(O; \text{Label})$. The function that best "predicts" the label is, by definition, the best order parameter. This approach has been used successfully in machine learning to automatically discover the order parameters of complex physical systems without prior human-supplied physical intuition [12].

---

## **2.8 Worked Example — Sampling a High-Dimensional Gaussian**

In this example, we bridge theory and practice. We will use a known high-dimensional probability distribution—the Multivariate Normal (Gaussian) distribution—as our "ground truth." By drawing samples from it, we can simulate a dataset and test how well our statistical tools (from Chapter 1) can recover the true underlying parameters. This will vividly illustrate the challenges of high-dimensional statistics, particularly the "curse of dimensionality."

---

### **The Ground Truth: A Correlated $D$-Dimensional Gaussian**

We define our true distribution $p(\mathbf{x})$ to be a $D$-dimensional Multivariate Gaussian $\mathcal{N}(\mathbf{\mu}_{\text{true}}, \Sigma_{\text{true}})$.
* **True Mean $\mathbf{\mu}_{\text{true}}$:** We'll set this to the zero vector $\mathbf{0}$ for simplicity.
* **True Covariance $\Sigma_{\text{true}}$:** This is the most important part. We will construct a $D \times D$ covariance matrix that is *not* diagonal. For example, we can define a correlated matrix where $\Sigma_{jj} = 1$ and the off-diagonals $\Sigma_{jk} = \rho^{|j-k|}$ (an autoregressive structure), creating a "true" physical correlation between features.

---

### **Sampling and Projection Effects**

We generate $N$ data samples $\{\mathbf{x}_i\}$ by drawing from $p(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \Sigma_{\text{true}})$. This $N \times D$ data matrix $X$ is our simulated "data cloud."

The first challenge is **visualization**. We cannot see the $D$-dimensional structure. Our only option is to look at 2D projections, or **pairwise marginals**—a scatter plot of feature $x_j$ versus feature $x_k$.

This reveals a critical pitfall:
* If we plot $x_1$ vs. $x_2$, we will see a clear, tilted ellipse, revealing the correlation $\rho$.
* However, if we plot $x_1$ vs. $x_5$, the correlation $\rho^4$ might be so small that the plot looks like an uncorrelated circular blob.

This demonstrates that **an absence of correlation in a 2D projection does not imply an absence of structure in $D$ dimensions.** A complex, high-dimensional manifold can look simple or unstructured from most random viewpoints.

---

### **Estimating the Covariance**

We now treat our $N$ samples as the *only* data we have, and we attempt to recover the "laws of physics" (i.e., $\Sigma_{\text{true}}$) that generated it. We compute the **empirical covariance matrix** from our data:

$$
\hat{\Sigma} = \frac{1}{N-1}\sum_{i=1}^N (\mathbf{x}_i - \mathbf{\mu}_{\text{emp}})(\mathbf{x}_i - \mathbf{\mu}_{\text{emp}})^{\top}
$$

where $\mathbf{\mu}_{\text{emp}}$ is the sample mean.

---

### **Observation: The Sample Size Requirement ($N$ vs. $D$)**

This is the key lesson. We compare $\hat{\Sigma}$ to $\Sigma_{\text{true}}$ and observe how the error $||\hat{\Sigma} - \Sigma_{\text{true}}||$ changes with $N$ and $D$.

* **Case 1: $N \gg D$ (e.g., $D=10, N=1,000,000$).** The number of samples is huge compared to the dimension. $\hat{\Sigma}$ will be a very accurate, "stable" estimate of $\Sigma_{\text{true}}$. The law of large numbers holds.

* **Case 2: $N \approx D$ or $N < D$ (e.g., $D=100, N=50$).** This is the **high-dimensional regime** common in physics (e.g., $D=3N_{\text{atoms}}$). The result is catastrophic:
    1.  **Singularity:** If $N < D$, the matrix $\hat{\Sigma}$ is mathematically **singular** (not invertible). It has at most $N-1$ non-zero eigenvalues, meaning it "thinks" the $D$-dimensional data lives in a much smaller $(N-1)$-dimensional subspace.
    2.  **High Variance:** The $D(D+1)/2$ unique elements of $\Sigma_{\text{true}}$ (e.g., 5,050 for $D=100$) are being estimated from only $N=50$ samples. The estimates are dominated by statistical noise. The empirical eigenvalues will not match the true ones at all (this is a central topic in Random Matrix Theory).

This example demonstrates that to get a stable, well-behaved estimate of our system's statistical geometry (its covariance), the number of samples $N$ must be significantly larger than the number of dimensions $D$. When $D$ is large, this is often impossible, reinforcing that we must first reduce the *effective* dimension before statistical inference can be trusted.

---

## **2.9 Code Demo — Monte Carlo Density Estimation**

This demonstration provides a practical implementation of the concepts from Section 2.6 (Density Estimation) and Section 2.8 (Sampling). We will:

1.  Define a "ground truth" 2D correlated Gaussian distribution, $p(\mathbf{x})$.
2.  Draw $N=5000$ samples from it, simulating our "data cloud."
3.  Use **Kernel Density Estimation (KDE)**, a powerful nonparametric method, to construct a smooth, continuous estimate $\hat{p}(\mathbf{x})$ from only the samples.
4.  Visualize the resulting learned density landscape.

This exercise is a low-dimensional analogue for inferring a free-energy landscape from a simulation trajectory.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde

## 1. Define the "ground truth" target distribution

## A 2D Gaussian with strong correlation (non-diagonal covariance)

mean = np.zeros(2)
cov = np.array([[1.0, 0.8], [0.8, 1.5]])
## This covariance matrix defines the "true" energy landscape.

## 2. Draw N samples (our "simulation data")

samples = np.random.multivariate_normal(mean, cov, 5000)

## 3. Perform Kernel Density Estimation (KDE)

## This is the practical implementation of the formula from Sec 2.6

## 'gaussian_kde' automatically determines a good bandwidth (h).

kde = gaussian_kde(samples.T)

## 4. Create a 2D grid to evaluate the learned density

x, y = np.mgrid[-3:3:100j, -3:3:100j]
grid_points = np.vstack([x.ravel(), y.ravel()])

## Evaluate the KDE on the grid to get the smooth density

z = kde(grid_points).reshape(100, 100)

## 5. Visualization

plt.figure(figsize=(9, 7))
## Plot the learned, smooth density as a filled contour map

plt.contourf(x, y, z, levels=30, cmap='viridis')
plt.colorbar(label='Estimated Density $\hat{p}(\mathbf{x})$')

## Overlay a few raw samples to show the "empirical" data

plt.scatter(samples[:200, 0], samples[:200, 1], s=5, color='white', alpha=0.5, label='Raw Samples')

plt.title('Monte Carlo + KDE Density Estimation')
plt.xlabel('Feature $x_1$')
plt.ylabel('Feature $x_2$')
plt.legend()
plt.show()
```

**Interpretation:**
The resulting plot shows the power of nonparametric estimation in low dimensions.

  * The **white dots** represent our raw, discrete "empirical distribution" $\hat{p}(\mathbf{x}) \approx \frac{1}{N}\sum \delta(\mathbf{x}-\mathbf{x}_i)$.
  * The **colored contour plot** represents the *learned, smooth* density $\hat{p}(\mathbf{x})$ from the KDE.

The KDE has successfully "filled in the gaps" between the samples to reconstruct a continuous function. Crucially, the shape of the contours—a tilted ellipse—has correctly inferred the underlying correlation structure (the `cov` matrix) of the true distribution, even though it was only given the list of samples. This visualization represents the ideal goal of density estimation: to turn a sparse set of points into a complete, continuous model of the underlying probability landscape.

**Bridge to Chapter 3:** This works beautifully in 2D. But as Section 2.6 warned, in high dimensions ($D \gg 2$), KDE fails catastrophically due to data sparsity. The next chapter addresses this: we will first reduce the dimension, *then* apply KDE in the low-dimensional latent space.

---

## **References**

[1] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

[2] Pathria, R. K., & Beale, P. D. (2011). *Statistical Mechanics* (3rd ed.). Academic Press.

[3] Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379–423.

[4] Jaynes, E. T. (1957). Information Theory and Statistical Mechanics. *Physical Review*, 106(4), 620–630.

[5] Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency. *Annals of Mathematical Statistics*, 22(1), 79–86.

[6] Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

[7] Bevington, P. R., & Robinson, D. K. (2003). *Data Reduction and Error Analysis for the Physical Sciences* (3rd ed.). McGraw-Hill.

[8] Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "Nearest Neighbor" Meaningful? *Proceedings of the 7th International Conference on Database Theory (ICDT)*, 217–235.

[9] Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. *The Journal of Chemical Physics*, 21(6), 1087–1092.

[10] Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.

[11] Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.

[12] Carrasquilla, J., & Melko, R. G. (2017). Machine Learning Phases of Matter. *Nature Physics*, 13(5), 431–434.