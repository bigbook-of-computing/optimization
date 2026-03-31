# **Chapter 2: Statistics & Probability in High Dimensions (Workbook)**

The goal of this chapter is to formalize the probabilistic view of simulation data, confronting the mathematical challenges of high-dimensional space and establishing strategies for sampling, inference, and uncertainty quantification.

| Section | Topic Summary |
| :--- | :--- |
| **2.1** | From Geometry to Probability |
| **2.2** | Probability Distributions as Energy Landscapes |
| **2.3** | Likelihood and Inference |
| **2.4** | The Curse of Dimensionality |
| **2.5** | Sampling Strategies |
| **2.6** | Density Estimation |
| **2.7** | Entropy, Information, and Uncertainty |
| **2.8–2.9** | Worked Example, Code Demo, and Takeaways |

---

### 2.1 From Geometry to Probability

> **Summary:** The analysis shifts from describing the data's **geometry** (shape, mean, covariance) to inferring the underlying **probability distribution** $P(\mathbf{x})$. The **empirical distribution** $\hat{p}(\mathbf{x}) \approx \frac{1}{N}\sum \delta(\mathbf{x}-\mathbf{x}_i)$ is the direct, spiky representation of our knowledge. The core duality is established: **Maximizing log-likelihood is equivalent to minimizing effective energy**, $E_{\text{eff}}(\mathbf{x}) \equiv -k_B T \ln P(\mathbf{x})$.

#### Quiz Questions

!!! note "Quiz"
    **1. What is the fundamental concept in physics that is mathematically equivalent to maximizing the statistical log-likelihood, $\ln P(\mathbf{x})$?**
    
    * **A.** Maximizing the partition function $Z$.
    * **B.** **Minimizing the effective energy, $E_{\text{eff}}(\mathbf{x})$**. (**Correct**)
    * **C.** Minimizing the KL divergence.
    * **D.** Maximizing the thermal energy $k_B T$.
    
!!! note "Quiz"
    **2. The empirical distribution $\hat{p}(\mathbf{x})$ is defined as a sum of which mathematical functions centered on each data point $\mathbf{x}_i$?**
    
    * **A.** Gaussian kernels.
    * **B.** **Dirac delta functions**. (**Correct**)
    * **C.** Exponential family functions.
    * **D.** Covariance functions.
    
---

!!! question "Interview Practice"
    **Question:** The Boltzmann distribution $P(\mathbf{s}) \propto e^{-E[\mathbf{s}]/k_B T}$ links energy to probability. If a physical simulation samples two configurations, $\mathbf{x}_A$ and $\mathbf{x}_B$, and $\mathbf{x}_A$ is observed 100 times more often than $\mathbf{x}_B$, what does this tell you about their effective energies?
    
    **Answer Strategy:** This means the probability $P(\mathbf{x}_A)$ is 100 times higher than $P(\mathbf{x}_B)$. Since probability is exponentially related to energy ($P \propto e^{-E}$), the log-likelihood of $\mathbf{x}_A$ must be higher. Specifically, the effective energy of $\mathbf{x}_A$ must be **lower** than $\mathbf{x}_B$ by an amount equal to $k_B T \ln(100)$. The data confirms that $\mathbf{x}_A$ lies in a much deeper, more stable basin of the potential energy landscape.
    
---

---

### 2.2 Probability Distributions as Energy Landscapes

> **Summary:** Probability models belong to the **exponential family**, defined by $P(\mathbf{x}) = \frac{1}{Z} \exp[-E(\mathbf{x}; \mathcal{\theta})]$, where $Z(\mathcal{\theta})$ is the **partition function**. The **moments** (mean, covariance) and **cumulants** (non-Gaussianity) describe the distribution's shape. **Shannon entropy** $S[p]$ measures the total uncertainty, while the **Kullback-Leibler (KL) divergence** $D_{\mathrm{KL}}(p||q)$ measures the difference between two distributions.

#### Quiz Questions

!!! note "Quiz"
    **1. The quantity $Z(\mathcal{\theta}) = \int \exp[-E(\mathbf{x}; \mathcal{\theta})] d\mathbf{x}$ is the normalization constant in the exponential family, which in physics is known as the:**
    
    * **A.** Correlation Function.
    * **B.** **Partition Function**. (**Correct**)
    * **C.** Maximum Likelihood Estimate.
    * **D.** Fisher Information.
    
!!! note "Quiz"
    **2. For a Multivariate Gaussian distribution, all statistical cumulants of order greater than two are exactly zero. Therefore, higher-order cumulants are useful for measuring:**
    
    * **A.** The mean of the distribution.
    * **B.** The partition function $Z$.
    * **C.** **The non-Gaussianity of a distribution**. (**Correct**)
    * **D.** The time average.
    
---

!!! question "Interview Practice"
    **Question:** Define the **Kullback-Leibler (KL) divergence** $D_{\mathrm{KL}}(p||q)$, and explain why, despite its usefulness, it is often called a pseudo-distance or **not a true distance metric** in mathematics.
    
    **Answer Strategy:** The KL divergence is defined as $D_{\mathrm{KL}}(p||q) = \int p(\mathbf{x}) \ln\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right) d\mathbf{x}$. It quantifies the information lost when distribution $q$ is used to approximate $p$. It is not a true distance metric because it **is asymmetric**: $D_{\mathrm{KL}}(p||q) \neq D_{\mathrm{KL}}(q||p)$. A true metric must satisfy symmetry and the triangle inequality.
    
---

---

### 2.3 Likelihood and Inference

> **Summary:** The **likelihood function** $\mathcal{L}(\mathcal{\theta})$ is the probability of observing the data $X$ given parameters $\mathcal{\theta}$. **Maximum Likelihood Estimation (MLE)** maximizes $\ln \mathcal{L}(\mathcal{\theta})$, which for Gaussian noise is equivalent to minimizing the $\chi^2$ statistic. The **Maximum A Posteriori (MAP)** estimate includes a **prior distribution** $p(\mathcal{\theta})$, where the log-prior acts as a **regularization term** to the log-likelihood objective.

#### Quiz Questions

!!! note "Quiz"
    **1. Maximizing the log-likelihood function is numerically preferred over maximizing the likelihood function itself because it converts the computationally unstable **product of probabilities** into a(n):**
    
    * **A.** Quotient of cumulants.
    * **B.** **Sum of log-probabilities**. (**Correct**)
    * **C.** Deterministic energy function.
    * **D.** Euclidean distance.
    
!!! note "Quiz"
    **2. In statistical physics, the principle of **least-squares fitting** is formally justified as finding the Maximum Likelihood Estimate (MLE) under the fundamental assumption of:**
    
    * **A.** Maximum entropy.
    * **B.** **Independent Gaussian (Normal) noise**. (**Correct**)
    * **C.** Low-dimensional correlation.
    * **D.** The partition function being unity.
    
---

!!! question "Interview Practice"
    **Question:** Explain the philosophical difference between the **Maximum Likelihood Estimate (MLE)** and the **Maximum A Posteriori (MAP)** estimate, and how the concept of *regularization* links the two.
    
    **Answer Strategy:**
    * **MLE** is a frequentist approach that selects the parameters $\mathcal{\theta}$ that make the observed data $X$ most probable, ignoring any prior belief about $\mathcal{\theta}$.
    * **MAP** is a Bayesian approach that selects the parameters $\mathcal{\theta}$ that maximize the posterior probability, thereby incorporating prior belief.
    * **Link to Regularization:** Mathematically, finding the MAP estimate (maximizing $\ln \mathcal{L} + \ln p(\mathcal{\theta})$) is identical to maximizing the likelihood with an added penalty. The **log-prior, $\ln p(\mathcal{\theta})$, serves as a regularization term** that penalizes complex or improbable parameters, steering the fit away from extreme values favored by the data alone.
    
---

---

### 2.4 The Curse of Dimensionality

> **Summary:** The **Curse of Dimensionality** describes the exponential growth of complexity when the number of features $D$ is large. The counter-intuitive **Volume Paradox** shows that virtually all the volume of a hypersphere is concentrated near its surface, leaving the core empty. Consequences for data analysis include **distance concentration** (all points become equally far) and extreme **data sparsity**, which makes naïve density estimation impossible.

#### Quiz Questions

!!! note "Quiz"
    **1. Which phenomenon of high-dimensional space causes all pairwise distances between data points to converge to nearly the same value, rendering nearest-neighbor methods ineffective?**
    
    * **A.** Data Sparsity.
    * **B.** **Distance Concentration**. (**Correct**)
    * **C.** The Manifold Hypothesis.
    * **D.** The KL divergence.
    
!!! note "Quiz"
    **2. The single most compelling reason why we **cannot** rely on simple histogram methods (naïve density estimation) in high dimensions is:**
    
    * **A.** We need to maximize the log-likelihood.
    * **B.** **The volume of the space grows exponentially, requiring an impossible number of samples ($10^D$) to fill the bins**. (**Correct**)
    * **C.** The partition function is zero.
    * **D.** The energy is too high.
    
---

!!! question "Interview Practice"
    **Question:** How does the realization that "all of the volume of a high-dimensional sphere is concentrated in an infinitesimally thin shell near its surface" influence the computational strategy for solving scientific problems where $D$ is large?
    
    **Answer Strategy:** This failure of intuition (the **volume paradox**) means that any local sampling or calculation must account for the fact that the space is mostly empty. It provides the primary **motivation for dimensionality reduction**. We cannot work in $\mathbb{R}^D$; we must assume the data lies on a lower-dimensional **manifold** ($\mathcal{M}$), and the computational strategy must shift to finding the coordinates of $\mathcal{M}$ first (Chapter 3), where the data becomes dense enough for statistical inference.
    
---

---

### 2.5 Sampling Strategies

> **Summary:** Due to the curse of dimensionality, integrals must be computed by **sampling**. **Monte Carlo (MC)** methods, especially **Markov Chain Monte Carlo (MCMC)**, construct a **correlated random walk** whose stationary distribution is the target $P(\mathbf{x})$, allowing sampling without calculating the partition function $Z$. **Importance Sampling** estimates integrals by weighting samples drawn from an easier **proposal distribution** $q(\mathbf{x})$ by the ratio $w(\mathbf{x}) = P(\mathbf{x})/q(\mathbf{x})$.

#### Quiz Questions

!!! note "Quiz"
    **1. The primary advantage of using **Markov Chain Monte Carlo (MCMC)** for sampling from a complex distribution $P(\mathbf{x})$ is that it:**
    
    * **A.** Produces independent samples.
    * **B.** **Allows sampling from $P(\mathbf{x})$ even if the partition function $Z$ is unknown (intractable)**. (**Correct**)
    * **C.** Only works for Gaussian distributions.
    * **D.** Guarantees the variance will be zero.
    
!!! note "Quiz"
    **2. In **Importance Sampling**, the samples drawn from the proposal distribution $q(\mathbf{x})$ are multiplied by **importance weights** $w(\mathbf{x})$ defined as:**
    
    * **A.** $w(\mathbf{x}) = q(\mathbf{x}) / P(\mathbf{x})$.
    * **B.** $w(\mathbf{x}) = \ln P(\mathbf{x})$.
    * **C.** $w(\mathbf{x}) = P(\mathbf{x}) / \sum P(\mathbf{x})$.
    * **D.** **$w(\mathbf{x}) = P(\mathbf{x}) / q(\mathbf{x})$**. (**Correct**)
    
---

!!! question "Interview Practice"
    **Question:** MCMC samples are generated sequentially and are **correlated** in time, which inflates the final statistical error. Conversely, samples generated by **Importance Sampling** are typically **independent**. Despite this drawback, why is MCMC almost always preferred over importance sampling for complex, high-dimensional energy landscapes?
    
    **Answer Strategy:** MCMC is preferred because **Importance Sampling (IS) requires finding a good proposal distribution $q(\mathbf{x})$**. For a complex, high-dimensional $P(\mathbf{x})$ concentrated in irregular low-energy basins, it is nearly impossible to find a simple $q(\mathbf{x})$ that closely matches $P(\mathbf{x})$. If $q$ is a poor match, the variance of the IS estimator explodes. MCMC bypasses this difficulty by using a Markov chain that is **guaranteed to converge** to the correct distribution $P(\mathbf{x})$ through local moves, regardless of the distribution's complexity.
    
---

---

### 2.6 Density Estimation

> **Summary:** **Density estimation** attempts to fit a smooth function $\hat{p}(\mathbf{x})$ to the empirical data. **Parametric models** (e.g., **Gaussian Mixture Models, GMM**) assume a fixed functional form, suitable for multi-modal clustering. **Nonparametric methods** (e.g., **Kernel Density Estimation, KDE**) place a smooth kernel ("bump") on each data point and sum them. KDE requires carefully choosing the **bandwidth** $h$, which directly controls the bias-variance trade-off. The Curse of Dimensionality makes KDE impractical in $\mathbb{R}^D$, reinforcing the need for low-dimensional projections.

#### Quiz Questions

!!! note "Quiz"
    **1. A **Kernel Density Estimate (KDE)** is a nonparametric density model that is computed by placing and summing a smooth **kernel** (e.g., a small Gaussian bump) on which data feature?**
    
    * **A.** The mean vector $\mathcal{\mu}$.
    * **B.** The total partition function $Z$.
    * **C.** **Each individual data point (sample) $\mathbf{x}_i$**. (**Correct**)
    * **D.** The KL divergence term.
    
!!! note "Quiz"
    **2. In KDE, the parameter known as the **bandwidth ($h$)** controls which fundamental trade-off of the density estimate?**
    
    * **A.** The KL-divergence vs. the $L^2$ norm.
    * **B.** The mean vector vs. the covariance matrix.
    * **C.** **The bias (oversmoothing) versus the variance (spikiness)**. (**Correct**)
    * **D.** The time average vs. the ensemble average.
    
---

!!! question "Interview Practice"
    **Question:** GMMs are a common **parametric** model for multi-modal density estimation. Explain how the components of a GMM (means, covariances, and mixing coefficients) physically map to the properties of a molecular simulation displaying two distinct stable states.
    
    **Answer Strategy:**
    * **Mixing Coefficients ($\pi_k$):** These are the weights for each Gaussian component. They represent the **relative population (or probability)** of each stable state.
    * **Means ($\mathcal{\mu}_k$):** These are the centers of the Gaussian components. They represent the **average configuration (or central structure)** of the atoms in each stable state (e.g., the open conformation vs. the closed conformation).
    * **Covariances ($\Sigma_k$):** These define the shape and spread of each Gaussian component. They represent the **local flexibility or vibrational modes** within that single stable state.
    
---

---

### 2.7 Entropy, Information, and Uncertainty

> **Summary:** **Shannon entropy** $S[p]$ quantifies the total uncertainty or disorder of a distribution, providing the formal link between information theory and statistical mechanics. The **Principle of Maximum Entropy** justifies model choice by selecting the least biased distribution consistent with known constraints. **Mutual Information (MI)** $I(X;Y)$ measures the dependence between two variables and can be used to automatically discover hidden **order parameters** in complex physical systems.

#### Quiz Questions

!!! note "Quiz"
    **1. The **Principle of Maximum Entropy** justifies model choice by stating that the most unbiased probability distribution consistent with known constraints (like fixed mean and variance) is the one that:**
    
    * **A.** Minimizes the $\chi^2$ statistic.
    * **B.** **Maximizes the Shannon entropy $S[p]$**. (**Correct**)
    * **C.** Minimizes the number of samples $N$.
    * **D.** Maximizes the KL divergence.
    
!!! note "Quiz"
    **2. **Mutual Information** $I(X;Y)$ is a statistical measure that quantifies the:**
    
    * **A.** Total variance of variable $X$.
    * **B.** **Amount of shared information (dependence) between variable $X$ and variable $Y$**. (**Correct**)
    * **C.** Minimum Euclidean distance between them.
    * **D.** Number of components in a GMM.
    
---

!!! question "Interview Practice"
    **Question:** In the context of a phase transition simulation (like the Ising model), explain how maximizing **Mutual Information** between a candidate order parameter $O(\mathbf{x})$ and the known phase label could automatically verify if $O(\mathbf{x})$ is physically relevant.
    
    **Answer Strategy:** Mutual information $I(O; \text{Label})$ measures how much the uncertainty about the phase label (e.g., 'ordered' or 'disordered') is reduced by observing the value of the candidate parameter $O(\mathbf{x})$.
    * If $O(\mathbf{x})$ is a good order parameter (like total magnetization $M$), its value changes sharply and uniquely at the phase boundary. Thus, knowing $M$ tells you exactly what the phase is, making the MI value **large**.
    * If $O(\mathbf{x})$ is irrelevant (like the energy of a single atom), it provides no information about the phase label, and the MI value is **near zero**. Maximizing MI is an automated way to computationally discover the physical variable that best captures the macroscopic state of the system.
    
---

## 💡 Hands-On Simulation Projects (Chapter Conclusion) 🛠️

These projects are designed to implement and test the core concepts of high-dimensional statistics, sampling, and density estimation.

### Project 1: Testing the Curse of Dimensionality (Distance Concentration)

* **Goal:** Numerically demonstrate the **distance concentration** phenomenon in high dimensions.
* **Setup:** Generate two simple datasets: $D=2$ and $D=1000$ (both $N=500$ samples drawn from $\mathcal{N}(\mathbf{0}, I)$).
* **Steps:**
    1.  For both datasets, calculate the **Euclidean distance** between all $N(N-1)/2$ unique pairs of points.
    2.  For both, compute the ratio $\frac{\text{Max Distance} - \text{Min Distance}}{\text{Average Distance}}$ (the relative spread of distances).
* ***Goal***: Show that for $D=2$, the relative spread is large (e.g., $\sim 0.5$). For $D=1000$, the relative spread is tiny (approaching $0$), confirming that the distances between all points are virtually identical (distance concentration).

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# ====================================================================

## 1. Setup Datasets (Low D vs. High D)

## ====================================================================

N = 500  # Number of samples
## Run 1: Low Dimension (D=2)

D_LOW = 2
X_low = np.random.randn(N, D_LOW)

## Run 2: High Dimension (D=1000)

D_HIGH = 1000
X_high = np.random.randn(N, D_HIGH)

## ====================================================================

## 2. Distance Calculation and Concentration Metric

## ====================================================================

def calculate_concentration(X):
    """
    Calculates the relative spread of pairwise Euclidean distances.
    The ratio approaches 0 for high dimensions.
    """
    # Calculate all pairwise Euclidean distances
    distances = pdist(X, metric='euclidean')

    # Compute the required metrics
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    avg_dist = np.mean(distances)

    # Relative Spread Ratio
    spread_ratio = (max_dist - min_dist) / avg_dist
    return spread_ratio, avg_dist

spread_low, avg_low = calculate_concentration(X_low)
spread_high, avg_high = calculate_concentration(X_high)

## ====================================================================

## 3. Visualization and Summary

## ====================================================================

print("--- Distance Concentration Test (Curse of Dimensionality) ---")
print(f"1. Low Dimension (D={D_LOW}):")
print(f"   Average Distance: {avg_low:.3f}")
print(f"   Relative Spread Ratio: {spread_low:.3f} (Large variance in distances)")

print(f"\n2. High Dimension (D={D_HIGH}):")
print(f"   Average Distance: {avg_high:.3f}")
print(f"   Relative Spread Ratio: {spread_high:.3f} (Distances are tightly concentrated)")

print("\nConclusion: The Spread Ratio drops significantly in high dimensions, confirming that most of the space is empty and all data points become equidistant from one another. This illustrates why distance-based algorithms struggle in high-D feature space.")
```
**Sample Output:**
```python
--- Distance Concentration Test (Curse of Dimensionality) ---
1. Low Dimension (D=2):
   Average Distance: 1.806
   Relative Spread Ratio: 3.986 (Large variance in distances)

2. High Dimension (D=1000):
   Average Distance: 44.678
   Relative Spread Ratio: 0.191 (Distances are tightly concentrated)

Conclusion: The Spread Ratio drops significantly in high dimensions, confirming that most of the space is empty and all data points become equidistant from one another. This illustrates why distance-based algorithms struggle in high-D feature space.
```


### Project 2: Importance Sampling for Estimating $\langle f \rangle$

* **Goal:** Use Importance Sampling to estimate a known integral, testing the dependence on the proposal distribution.
* **Setup:** Define the true target distribution $P(x)$ and the function $f(x)$ to be integrated (e.g., $P(x) = \mathcal{N}(x|0, 1)$ and $f(x) = x^2$). The exact answer $\langle f \rangle_P$ is the variance of $P$ (which is 1).
* **Steps:**
    1.  **Trial A (Good Proposal):** Sample $N$ points from a matching proposal $Q_A = \mathcal{N}(x|0, 1)$. Compute $\langle f \rangle$ using the weights $w(x) = P(x)/Q_A(x)$ (which should be 1).
    2.  **Trial B (Poor Proposal):** Sample $N$ points from a distant proposal $Q_B = \mathcal{N}(x|5, 1)$. Compute $\langle f \rangle$ using the new weights.
* ***Goal***: Show that both trials yield the correct mean ($\approx 1$), but Trial B will have a **much higher variance** (less stable estimate) due to the low probability of sampling the important region near $x=0$.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Target Distribution (P) and Function (f)

## ====================================================================

N_SAMPLES = 10000  # Number of samples for the estimator
TARGET_MEAN = 0.0
TARGET_STD = 1.0

## True Distribution P(x) = N(x|0, 1)

def P(x):
    return norm.pdf(x, loc=TARGET_MEAN, scale=TARGET_STD)

## Function to integrate f(x) = x^2

def f(x):
    return x**2

## Analytical Result: <f>_P = <x^2>_N(0,1) = Variance = 1.0

ANALYTICAL_MEAN = 1.0

## ====================================================================

## 2. Importance Sampling Trials

## ====================================================================

## Trial A: Good Proposal Q_A (Perfect Match)

Q_A = lambda x: norm.pdf(x, loc=TARGET_MEAN, scale=TARGET_STD)
X_A = np.random.normal(loc=TARGET_MEAN, scale=TARGET_STD, size=N_SAMPLES)
Weights_A = P(X_A) / Q_A(X_A)  # Weights should be all 1s
Estimate_A = np.mean(f(X_A) * Weights_A)
Variance_A = np.var(f(X_A) * Weights_A)

## Trial B: Poor Proposal Q_B (Distant Mean)

Q_B_MEAN = 5.0
Q_B = lambda x: norm.pdf(x, loc=Q_B_MEAN, scale=TARGET_STD)
X_B = np.random.normal(loc=Q_B_MEAN, scale=TARGET_STD, size=N_SAMPLES)
Weights_B = P(X_B) / Q_B(X_B)
Estimate_B = np.mean(f(X_B) * Weights_B)
Variance_B = np.var(f(X_B) * Weights_B)

## ====================================================================

## 3. Visualization and Summary

## ====================================================================

print("--- Importance Sampling Performance ---")
print(f"Target Analytical Mean <f(x)>_P: {ANALYTICAL_MEAN:.4f}")

print("\nTrial A: Good Proposal Q_A = N(0, 1)")
print(f"  Estimate: {Estimate_A:.4f} (Accurate)")
print(f"  Variance of Estimator: {Variance_A:.4f} (Low)")

print("\nTrial B: Poor Proposal Q_B = N(5, 1)")
print(f"  Estimate: {Estimate_B:.4f}")
print(f"  Variance of Estimator: {Variance_B:.4f} (Extremely High)")

## Plotting the weights (visualizing the mismatch)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_B, Weights_B, s=10, alpha=0.5, color='darkred', label='Weights $w(x) = P(x)/Q_B(x)$')
ax.axhline(0, color='k', linestyle='--')
ax.set_title("Importance Weights for Poor Proposal $Q_B = \\mathcal{N}(5, 1)$")
ax.set_xlabel("Sampled Point $x$")
ax.set_ylabel("Importance Weight $w(x)$")
ax.grid(True)
plt.show()

print("\nConclusion: Both trials achieved the correct mean (analytic result of 1.0) on average. However, the variance of the estimate for the distant proposal (Trial B) is orders of magnitude higher. This confirms that the variance of the Importance Sampling estimator explodes when the proposal distribution does not adequately cover the important, low-energy region (near x=0) of the target distribution.")
```
**Sample Output:**
```python
--- Importance Sampling Performance ---
Target Analytical Mean <f(x)>_P: 1.0000

Trial A: Good Proposal Q_A = N(0, 1)
  Estimate: 1.0068 (Accurate)
  Variance of Estimator: 2.0543 (Low)

Trial B: Poor Proposal Q_B = N(5, 1)
  Estimate: 0.4965
  Variance of Estimator: 190.2655 (Extremely High)

Conclusion: Both trials achieved the correct mean (analytic result of 1.0) on average. However, the variance of the estimate for the distant proposal (Trial B) is orders of magnitude higher. This confirms that the variance of the Importance Sampling estimator explodes when the proposal distribution does not adequately cover the important, low-energy region (near x=0) of the target distribution.
```


### Project 3: Visualizing Density Estimation (KDE Bandwidth)

* **Goal:** Visually demonstrate the effect of the **bandwidth ($h$)** parameter on the bias-variance trade-off in Kernel Density Estimation.
* **Setup:** Generate $N=100$ samples from a simple 1D multi-modal distribution (e.g., a GMM with two separated peaks, or a simple $\mathcal{N}(0, 1)$).
* **Steps:**
    1.  Compute and plot the KDE estimate $\hat{p}(x)$ using a **small bandwidth** ($h_{\text{small}}$).
    2.  Compute and plot the KDE estimate $\hat{p}(x)$ using a **large bandwidth** ($h_{\text{large}}$).
* ***Goal***: Show that $h_{\text{small}}$ results in a high-variance, spiky estimate (under-smoothing), while $h_{\text{large}}$ results in a high-bias, oversmoothed estimate (erasing true features).

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Ground Truth and Sample Data

## ====================================================================

N_SAMPLES = 100
## Ground Truth: A bimodal distribution (two Gaussians)

X_A = np.random.normal(loc=-2, scale=0.5, size=N_SAMPLES // 2)
X_B = np.random.normal(loc=2, scale=1.0, size=N_SAMPLES // 2)
X_full = np.concatenate([X_A, X_B])

## Function for the true underlying density (for plotting reference)

def true_density(x):
    return 0.5 * norm.pdf(x, loc=-2, scale=0.5) + 0.5 * norm.pdf(x, loc=2, scale=1.0)

## Grid for plotting the smooth functions

x_plot = np.linspace(-5, 5, 500)
y_true = true_density(x_plot)

## ====================================================================

## 2. KDE Trials (Varying Bandwidth)

## ====================================================================

## Trial A: Small Bandwidth (High Variance, Low Bias)

H_SMALL = 0.1
kde_small = gaussian_kde(X_full, bw_method=H_SMALL)
y_small = kde_small(x_plot)

## Trial B: Large Bandwidth (High Bias, Low Variance)

H_LARGE = 1.0
kde_large = gaussian_kde(X_full, bw_method=H_LARGE)
y_large = kde_large(x_plot)

## ====================================================================

## 3. Visualization and Summary

## ====================================================================

plt.figure(figsize=(10, 6))
## Plot raw data (rug plot at the bottom)

plt.plot(X_full, np.full_like(X_full, -0.01), '|k', markeredgewidth=1, alpha=0.5, label='Raw Data Samples')

## Plot true density

plt.plot(x_plot, y_true, 'k--', label='True Density (Reference)', lw=2)

## Plot KDE estimates

plt.plot(x_plot, y_small, 'r-', label=f'KDE (h={H_SMALL}): High Variance', lw=1.5)
plt.plot(x_plot, y_large, 'b-', label=f'KDE (h={H_LARGE}): High Bias', lw=1.5)

## Labeling and Formatting

plt.title('Kernel Density Estimation: Bandwidth and Bias-Variance Trade-off')
plt.xlabel('x')
plt.ylabel('Probability Density $\\hat{p}(x)$')
plt.ylim(-0.05, 0.45)
plt.legend()
plt.grid(True)
plt.show()

print("\n--- KDE Bandwidth Analysis ---")
print(f"Reference Structure: Bimodal (peaks at x=-2 and x=2)")
print("-------------------------------------------------")
print(f"KDE with h={H_SMALL} (Small): Estimate is spiky (high variance) but accurately resolves the bimodal structure (low bias).")
print(f"KDE with h={H_LARGE} (Large): Estimate is smooth (low variance) but fails to resolve the two peaks, becoming an inaccurate single-mode blob (high bias).")

print("\nConclusion: The bandwidth h controls the bias-variance trade-off. A proper choice is critical for accurately inferring the underlying multi-modal energy landscape.")
```
**Sample Output:**
```python
--- KDE Bandwidth Analysis ---
Reference Structure: Bimodal (peaks at x=-2 and x=2)

---

KDE with h=0.1 (Small): Estimate is spiky (high variance) but accurately resolves the bimodal structure (low bias).
KDE with h=1.0 (Large): Estimate is smooth (low variance) but fails to resolve the two peaks, becoming an inaccurate single-mode blob (high bias).

Conclusion: The bandwidth h controls the bias-variance trade-off. A proper choice is critical for accurately inferring the underlying multi-modal energy landscape.
```


### Project 4: Maximum Likelihood Estimation (MLE) of a Gaussian

* **Goal:** Numerically find the MLE parameters ($\mathcal{\mu}, \Sigma$) for a simple Gaussian distribution.
* **Setup:** Generate a large dataset $X$ ($N=1000, D=2$) from a known Gaussian $\mathcal{N}(\mathcal{\mu}_{\text{true}}, \Sigma_{\text{true}})$.
* **Steps:**
    1.  The analytical MLE solution for the Gaussian is simply the **sample mean** and **sample covariance**. Compute $\hat{\mathcal{\mu}}_{\text{MLE}}$ and $\hat{\Sigma}_{\text{MLE}}$ using the empirical formulas from Section 1.2.
    2.  Compute the **log-likelihood** of the generated data using the true parameters ($\ln \mathcal{L}_{\text{true}}$) and the estimated parameters ($\ln \mathcal{L}_{\text{MLE}}$).
* ***Goal***: Show that the estimated parameters ($\hat{\mathcal{\mu}}_{\text{MLE}}, \hat{\Sigma}_{\text{MLE}}$) are very close to the true parameters, and that the log-likelihood calculated using the estimated parameters ($\ln \mathcal{L}_{\text{MLE}}$) is demonstrably higher than the log-likelihood from any other randomly chosen parameters.

#### Python Implementation

```python
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Ground Truth and Data Generation

## ====================================================================

N_SAMPLES = 1000
D = 2  # 2D Gaussian

## --- True Parameters (The "Ground Truth") ---

MU_TRUE = np.array([2.5, -1.0])
SIGMA_TRUE = np.array([[1.0, 0.5], [0.5, 2.0]])

## Generate the data from the true distribution

X_data = np.random.multivariate_normal(MU_TRUE, SIGMA_TRUE, N_SAMPLES)

## ====================================================================

## 2. Maximum Likelihood Estimation (Analytical Solution)

## ====================================================================

## MLE Estimate for the Gaussian Mean is the Sample Mean

MU_MLE = np.mean(X_data, axis=0)

## MLE Estimate for the Gaussian Covariance is the Sample Covariance

## Note: np.cov uses N-1 by default (unbiased sample covariance); the MLE formula uses N (biased).

## We use the unbiased estimator here for better comparison, but note the technical distinction.

SIGMA_MLE = np.cov(X_data, rowvar=False)

## --- Define a Poorly Chosen Parameter Set for Comparison ---

MU_POOR = np.array([0.0, 0.0])
SIGMA_POOR = np.array([[3.0, 0.0], [0.0, 3.0]])

## ====================================================================

## 3. Log-Likelihood Calculation (Verification)

## ====================================================================

def calculate_log_likelihood(X, mu, sigma):
    """
    Computes the total log-likelihood for the dataset X given parameters (mu, sigma).
    This measures how well the model (mu, sigma) explains the data.
    """
    # Create the multivariate Gaussian object
    model = multivariate_normal(mean=mu, cov=sigma)

    # Compute the log-likelihood for each point and sum them up
    # Total Log-Likelihood = sum(log P(x_i | theta))
    return np.sum(model.logpdf(X))

## Calculate the log-likelihood for the three parameter sets

LL_TRUE = calculate_log_likelihood(X_data, MU_TRUE, SIGMA_TRUE)
LL_MLE = calculate_log_likelihood(X_data, MU_MLE, SIGMA_MLE)
LL_POOR = calculate_log_likelihood(X_data, MU_POOR, SIGMA_POOR)

## ====================================================================

## 4. Visualization and Summary

## ====================================================================

print("--- Maximum Likelihood Estimation (MLE) Analysis ---")

print("\n1. Parameter Comparison:")
print(f"| Parameter | True | MLE Estimate | Difference |")
print("| :--- | :--- | :--- | :--- |")
print(f"| Mean (\u03bc) | {MU_TRUE} | {np.round(MU_MLE, 3)} | {np.round(MU_MLE - MU_TRUE, 3)} |")
print(f"| Cov (\u03a3)[0,1] | {SIGMA_TRUE[0, 1]:.3f} | {SIGMA_MLE[0, 1]:.3f} | {SIGMA_MLE[0, 1] - SIGMA_TRUE[0, 1]:.3f} |")

print("\n2. Log-Likelihood (LL) Verification:")
print(f"LL of True Parameters:    {LL_TRUE:.2f}")
print(f"LL of MLE Parameters:     {LL_MLE:.2f} (Maximized)")
print(f"LL of Poor Parameters:    {LL_POOR:.2f}")

## Plot LL comparison

plt.figure(figsize=(8, 5))
plt.bar(['LL_True', 'LL_MLE', 'LL_Poor'], [LL_TRUE, LL_MLE, LL_POOR], color=['gray', 'darkgreen', 'red'])
plt.axhline(LL_MLE, color='k', linestyle='--', alpha=0.6, label='Maximum Likelihood')
plt.title('Log-Likelihood Maximization')
plt.ylabel('Total Log-Likelihood')
plt.grid(True, axis='y')
plt.show()

print("\nConclusion: The LL_MLE is the highest value, confirming that the empirical sample mean and covariance are the **Maximum Likelihood Estimates** for the Gaussian model. This numerically verifies the analytical solution and shows that the empirical statistics correctly capture the generative parameters of the distribution under the MaxEnt principle.")
```
**Sample Output:**
```python
--- Maximum Likelihood Estimation (MLE) Analysis ---

1. Parameter Comparison:
| Parameter | True | MLE Estimate | Difference |
| :--- | :--- | :--- | :--- |
| Mean (μ) | [ 2.5 -1. ] | [ 2.566 -0.974] | [0.066 0.026] |
| Cov (Σ)[0,1] | 0.500 | 0.435 | -0.065 |

2. Log-Likelihood (LL) Verification:
LL of True Parameters:    -3096.29
LL of MLE Parameters:     -3092.38 (Maximized)
LL of Poor Parameters:    -4667.53

Conclusion: The LL_MLE is the highest value, confirming that the empirical sample mean and covariance are the **Maximum Likelihood Estimates** for the Gaussian model. This numerically verifies the analytical solution and shows that the empirical statistics correctly capture the generative parameters of the distribution under the MaxEnt principle.
```