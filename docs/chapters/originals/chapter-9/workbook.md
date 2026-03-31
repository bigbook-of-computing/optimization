# **Chapter 9: Bayesian Thinking and Inference () () () (Workbook)**

The goal of this chapter is to establish **Bayes' Theorem** as the governing law of learning, viewing inference as a process of **energy minimization** that explicitly quantifies and reduces uncertainty.

| Section | Topic Summary |
| :--- | :--- |
| **9.1** | From Optimization to Inference |
| **9.2** | Bayes’ Theorem — Updating Belief |
| **9.3** | Maximum Likelihood and MAP Estimation |
| **9.4** | Priors, Likelihoods, and Posteriors |
| **9.5** | Evidence and Model Comparison |
| **9.6** | The Free Energy View of Inference |
| **9.7** | Uncertainty and Credible Intervals |
| **9.8** | Bayesian Networks — The Architecture of Belief |
| **9.9–9.12** | Worked Example, Code Demo, and Takeaways |

---

### 9.1 From Optimization to Inference

> **Summary:** The transition from optimization to inference is based on the **Energy–Probability Duality**. **Minimizing energy** $E(\mathbf{s})$ is equivalent to **maximizing probability** $P(\mathbf{s})$ via the Boltzmann distribution. Optimization finds the single **best state** ($\mathcal{\theta}^*$) while inference characterizes the overall **distribution of plausible states** ($P(\mathcal{\theta}|\mathcal{D})$), explicitly modeling uncertainty.

#### Quiz Questions

!!! note "Quiz"
```
**1. The fundamental duality that links the optimization goal ($\min E$) with the inference goal ($\max P$) is provided by which statistical physics law?**

* **A.** The Law of Least Action.
* **B.** The Maximum Entropy Principle.
* **C.** **The Boltzmann distribution**. (**Correct**)
* **D.** The Central Limit Theorem.

```
!!! note "Quiz"
```
**2. In the transition to Part III, the primary output we seek is no longer a single point estimate but a full distribution because inference aims to explicitly model the system's:**

* **A.** Final kinetic energy.
* **B.** **Uncertainty**. (**Correct**)
* **C.** Learning rate.
* **D.** Deterministic gradient flow.

```
---

!!! question "Interview Practice"
```
**Question:** The text suggests that the minimal free energy configuration in physics is the statistical mirror of the optimal belief system in learning. Explain the two terms that the **Helmholtz Free Energy ($\mathcal{F}$)** balances in this physical analogy.

**Answer Strategy:** The Helmholtz Free Energy ($\mathcal{F} = E - T S$) balances:
1.  **Internal Energy ($E$):** Analogous to the **Loss** or the model's **fit to the data**. This term drives accuracy.
2.  **Entropy ($S$):** Analogous to the model's **complexity or uncertainty**. This term drives simplicity and generality.
The optimal belief system minimizes this quantity, achieving a natural trade-off between maximizing fit (low $E$) and maintaining plausible simplicity (high $S$).

```
---

---

### 9.2 Bayes’ Theorem — Updating Belief

> **Summary:** **Bayes' Theorem** is the fundamental rule for rationally processing **new evidence ($\mathcal{D}$) to update prior knowledge ($p(\mathcal{\theta})$)**. The posterior distribution, $p(\mathcal{\theta}|\mathcal{D})$, is proportional to the product of the **Likelihood** ($p(\mathcal{D}|\mathcal{\theta})$) and the **Prior** ($p(\mathcal{\theta})$). This continuous process of refinement is directly analogous to **entropy reduction**.

#### Quiz Questions

!!! note "Quiz"
```
**1. In Bayes' Theorem, the **Prior** distribution $p(\mathcal{\theta})$ represents the system's:**

* **A.** Probability of observing the data given the parameters.
* **B.** **Initial belief about the parameters before observing the data**. (**Correct**)
* **C.** Normalization constant.
* **D.** Updated belief after inference.

```
!!! note "Quiz"
```
**2. The process of Bayesian learning is fundamentally analogous to **entropy reduction** because the accumulation of evidence causes the posterior distribution to:**

* **A.** Increase its mean value.
* **B.** **Become sharper and narrower, reducing the total uncertainty**. (**Correct**)
* **C.** Violate the Boltzmann factor.
* **D.** Converge to a uniform distribution.

```
---

!!! question "Interview Practice"
```
**Question:** Explain the philosophical significance of the normalization constant, the **Model Evidence $p(\mathcal{D})$**, in the Bayesian learning process, even though it's often ignored when calculating the posterior mode?

**Answer Strategy:** The model evidence $p(\mathcal{D})$ is the total probability of observing the data under the given model. It's ignored for parameter estimation because it's constant with respect to $\mathcal{\theta}$. However, it's essential for **model comparison** (Section 9.5). It forces two competing models (hypotheses) to be evaluated on their overall explanatory power, including the prior plausibility of their structure, providing the mathematical basis for **Occam's Razor**.

```
---

---

### 9.3 Maximum Likelihood and MAP Estimation

> **Summary:** **Maximum Likelihood Estimation (MLE)** finds the parameter vector $\mathcal{\theta}_{\text{ML}}$ that maximizes the likelihood $p(\mathcal{D}|\mathcal{\theta})$. **Maximum A Posteriori (MAP) Estimation** finds the parameter vector $\mathcal{\theta}_{\text{MAP}}$ that maximizes the posterior $p(\mathcal{\theta}|\mathcal{D})$. **MAP estimation is mathematically identical to regularized optimization**, where the **negative log-prior** ($-\ln p(\mathcal{\theta})$) serves as the explicit **regularization term** or penalty.

#### Quiz Questions

!!! note "Quiz"
```
**1. The primary difference between the ML and MAP point estimates is that the MAP estimate explicitly includes and is influenced by the:**

* **A.** Learning rate.
* **B.** **Prior distribution $p(\mathcal{\theta})$**. (**Correct**)
* **C.** Likelihood function $p(\mathcal{D}|\mathcal{\theta})$.
* **D.** Model evidence $p(\mathcal{D})$.

```
!!! note "Quiz"
```
**2. If a model is trained using a standard Maximum Likelihood objective with an added $L^2$ penalty (Ridge Regression), the resulting optimization is mathematically equivalent to the MAP estimate under the assumption of a:**

* **A.** Laplace prior.
* **B.** **Gaussian prior**. (**Correct**)
* **C.** Uniform prior.
* **D.** Conjugate prior.

```
---

!!! question "Interview Practice"
```
**Question:** The MAP objective is $\arg\min [-\ln p(\mathcal{D}|\mathcal{\theta}) - \ln p(\mathcal{\theta})]$. Explain the practical and computational advantage of converting the likelihood-prior product into a sum of negative logarithms.

**Answer Strategy:**
1.  **Numerical Stability:** The product of many small probabilities ($\prod p_i$) is numerically unstable and prone to underflow in computers. The sum of logarithms ($\sum \ln p_i$) is numerically stable.
2.  **Optimization:** The summation is easily handled by gradient-based optimizers (Chapters 5–6). The total negative log-likelihood becomes the standard differentiable **loss function**. The summation form is the essential bridge between probabilistic inference and optimization dynamics.

```
---

---

### 9.4 Priors, Likelihoods, and Posteriors

> **Summary:** The **Prior $p(\mathcal{\theta})$** is the explicit encoding of **inductive bias**. A **conjugate prior** is chosen for analytical convenience, ensuring the posterior remains in the same distributional family as the prior (e.g., Beta-Binomial). The **Likelihood** acts as the evidence that **breaks the symmetry** of the prior, forcing the resulting **Posterior** to adopt the structure revealed by the data.

#### Quiz Questions

!!! note "Quiz"
```
**1. The selection of a **conjugate prior** is most beneficial in Bayesian inference because it allows for:**

* **A.** Non-linear embeddings.
* **B.** **A closed-form, analytical solution for the posterior distribution**. (**Correct**)
* **C.** Guaranteed convergence to the global minimum.
* **D.** The implementation of Variational Inference.

```
!!! note "Quiz"
```
**2. When a Likelihood function is based on data, its introduction into the prior effectively acts as a physical perturbation that:**

* **A.** Minimizes the free energy.
* **B.** **Breaks the symmetry of the prior distribution**. (**Correct**)
* **C.** Increases the total variance.
* **D.** Always creates an $L^1$ penalty term.

```
---

!!! question "Interview Practice"
```
**Question:** In the worked example of the Bayesian Coin Toss (Section 9.9), the **Posterior Mean** is a weighted average of the prior belief and the empirical frequency. Explain what determines the relative weight (or confidence) placed on the **prior belief** versus the **empirical evidence**.

**Answer Strategy:** The relative weights are determined by the effective "size" of the prior versus the data.
* **Prior Weight:** Determined by the **hyperparameters ($\alpha + \beta$)** of the Beta prior. A larger $\alpha + \beta$ means the model has more **prior confidence** in its initial belief.
* **Empirical Weight:** Determined by the **number of observed data points ($n$)**.
If $n$ is small, the posterior is dominated by the prior. If $n$ is large, the posterior is dominated by the empirical evidence.

```
---

---

### 9.5 Evidence and Model Comparison

> **Summary:** The **Model Evidence ($p(\mathcal{D}|M)$)** is the total probability of the data averaged over all possible parameters of the model $M$. Models are compared using the **Bayes Factor ($\text{BF}_{12}$)**, which is the ratio of their evidences. Maximizing the evidence provides a rigorous mathematical basis for **Occam's Razor**, naturally penalizing overly complex models that occupy only a small volume of the parameter space.

#### Quiz Questions

!!! note "Quiz"
```
**1. In Bayesian model comparison, the **Model Evidence** $p(\mathcal{D}|M)$ is favored over the maximum likelihood because it:**

* **A.** **Integrates the likelihood over the entire parameter space**. (**Correct**)
* **B.** Is guaranteed to be less than zero.
* **C.** Does not require a prior.
* **D.** Requires a specific analytical formula.

```
!!! note "Quiz"
```
**2. Which principle is automatically enforced by the mathematical structure of the model evidence integral?**

* **A.** The Heisenberg Uncertainty Principle.
* **B.** **Occam's Razor**. (**Correct**)
* **C.** The Principle of Least Action.
* **D.** The Equipartition Theorem.

```
---

!!! question "Interview Practice"
```
**Question:** A complex model ($M_2$) achieves a slightly higher maximum likelihood peak than a simpler model ($M_1$), but the Bayes Factor strongly favors $M_1$. Why does the model evidence penalize $M_2$ in this scenario?

**Answer Strategy:** The evidence penalizes $M_2$ because $M_2$ is likely **overfit**. Although its peak likelihood is high, a complex model often requires **highly specific, finely tuned parameters** (occupying an infinitesimally small volume) to achieve that peak. The evidence integral averages over the *entire* parameter space. If the simpler model $M_1$ provides a reasonable fit across a much **broader, more robust volume** of its parameter space, the evidence integral for $M_1$ will be larger, demonstrating its superior simplicity and robustness.

```
---

## 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core concepts of Bayesian inference and uncertainty quantification.

### Project 1: Implementing the MAP $\leftrightarrow$ Regularization Duality

* **Goal:** Numerically confirm that the **MAP estimate** is equivalent to **regularized optimization**.
* **Setup:** Define a simple linear regression problem: $y = w x + b$. Use a **Gaussian prior** $p(w) \sim \mathcal{N}(0, \sigma^2)$ on the slope $w$.
* **Steps:**
    1.  Write the **MAP Objective** (negative log-posterior) that includes the penalty term ($\propto w^2$) derived from the Gaussian prior.
    2.  Write the **Regularized Optimization Objective** (Least Squares + $L^2$ penalty).
    3.  Numerically solve or find the gradient for both objectives.
* ***Goal***: Show that the two objectives are functionally identical, confirming that the $L^2$ regularization term is a direct consequence of the Gaussian prior.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ====================================================================
# 1. Setup Data and Hyperparameters
# ====================================================================

# We model a single parameter theta (e.g., the mean, mu)
# True value is unknown, but we sample data around mu_true = 5.0
MU_TRUE = 5.0
N_SAMPLES = 10 
DATA_STD = 1.0 # Standard deviation of the measurement process

# --- Simulate Data ---
np.random.seed(42)
data = np.random.normal(loc=MU_TRUE, scale=DATA_STD, size=N_SAMPLES)
sample_mean = np.mean(data) # This will be the Maximum Likelihood Estimate (MLE)

# --- Prior Belief (Our Initial Guess) ---
MU_PRIOR = 1.0      # We guess the mean is 1.0 (far from true 5.0)
SIGMA_PRIOR = 1.0   # We are moderately uncertain (variance = 1.0)

# ====================================================================
# 2. Likelihood and Posterior Calculation (Analytic Gaussian Formula)
# ====================================================================

# 1. Likelihood Function (of theta): P(D | theta)
# For Gaussian data, the likelihood is proportional to a Gaussian centered at the sample mean.
def likelihood(theta):
    # Sum of log probabilities (neglecting constants since we only need proportionality)
    return np.exp(-0.5 * np.sum((data - theta)**2) / DATA_STD**2)

# 2. Analytic Posterior Parameters (Known formula for Gaussian-Gaussian pair)
# The posterior mean is a weighted average of the prior mean and the MLE (sample mean)
sigma_likelihood = DATA_STD**2 / N_SAMPLES 
mu_posterior = (MU_PRIOR/SIGMA_PRIOR**2 + sample_mean/sigma_likelihood) / \
               (1/SIGMA_PRIOR**2 + 1/sigma_likelihood)

sigma_posterior = np.sqrt(1 / (1/SIGMA_PRIOR**2 + 1/sigma_likelihood))

# ====================================================================
# 3. Visualization
# ====================================================================

theta_plot = np.linspace(-3, 8, 300)

# 1. Prior Distribution
prior_dist = norm.pdf(theta_plot, loc=MU_PRIOR, scale=SIGMA_PRIOR)

# 2. Likelihood (renormalize for visualization purposes)
# The Likelihood is centered around the sample mean (MLE)
likelihood_values = likelihood(theta_plot)
# Normalize to fit on the plot
likelihood_values /= np.max(likelihood_values) 

# 3. Posterior Distribution
posterior_dist = norm.pdf(theta_plot, loc=mu_posterior, scale=sigma_posterior)


plt.figure(figsize=(9, 6))

plt.plot(theta_plot, prior_dist, 'b--', lw=2, label=f'Prior $P(\u03b8)$: $\\mu_0={MU_PRIOR:.1f}, \\sigma_0={SIGMA_PRIOR:.1f}$')
plt.plot(theta_plot, likelihood_values, 'g-', lw=2, alpha=0.7, label=f'Likelihood $\\mathcal{{L}}(\u03b8)$ (Data)')
plt.plot(theta_plot, posterior_dist, 'r-', lw=2, label=f'Posterior $P(\u03b8|\mathcal{{D}})$: $\\mu_P={mu_posterior:.2f}, \\sigma_P={sigma_posterior:.2f}$')

# Highlight the estimates
plt.axvline(sample_mean, color='k', linestyle=':', label=f'Sample Mean (MLE)={sample_mean:.2f}')
plt.axvline(mu_posterior, color='r', linestyle=':', lw=2)

# Labeling and Formatting
plt.title('Bayesian Inference: Prior $\\times$ Likelihood $\\to$ Posterior')
plt.xlabel(r'Parameter Value $\theta$ (Hypothesized Mean)')
plt.ylabel('Density / Normalized Likelihood')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Bayesian Inference Summary ---")
print(f"Sample Mean (MLE): {sample_mean:.3f}")
print(f"Posterior Mean (MAP): {mu_posterior:.3f}")
print(f"Posterior Standard Deviation: {sigma_posterior:.3f} (Reduced Uncertainty)")

print("\nConclusion: The visualization confirms Bayes' Theorem. The final Posterior is a compromise between the initial, incorrect Prior (centered at 1.0) and the new information provided by the Likelihood (centered at 5.0). The Posterior's peak (the MAP estimate) is closer to the Likelihood, and its width is narrower than both the Prior and the Likelihood, reflecting the **gain in certainty** from the data.")
```

### Project 2: Simulating Bayesian Learning and Shrinking Uncertainty

* **Goal:** Visually demonstrate how the posterior distribution narrows (reduces entropy) as evidence accumulates.
* **Setup:** Use the **Beta-Binomial Coin Toss** model. Start with a flat (uninformative) prior, $\text{Beta}(1, 1)$.
* **Steps:**
    1.  Calculate and plot the **Prior** ($\text{Beta}(1, 1)$).
    2.  Simulate two observations: **Observation A** ($n=2$ tosses, $k=1$ head). Calculate and plot the **Posterior A**.
    3.  Simulate 200 observations: **Observation B** ($n=200$ tosses, $k=120$ heads). Calculate and plot the **Posterior B**.
* ***Goal***: Show that the mean of the distribution shifts toward $0.6$ with data, and the uncertainty (variance/width) of Posterior B is significantly smaller than Posterior A and the initial Prior, demonstrating **entropy reduction**.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ====================================================================
# 1. Setup Data and Parameter Scenarios
# ====================================================================

# We set up a scenario where the MLE (Sample Mean) is noisy (high std error)
# and the MAP estimate should shrink this noise using a strong prior.
N_SAMPLES = 5 # Small number of samples for high noise/variance
TRUE_MU = 10.0
DATA_STD = 2.0
PRIOR_MEAN = 5.0 # Prior is far from the true value (5.0 vs 10.0)

# Simulate Noisy Data
np.random.seed(1) 
data = np.random.normal(loc=TRUE_MU, scale=DATA_STD, size=N_SAMPLES)
MLE_estimate = np.mean(data)

# --- Scenario A: Weak Prior (Wide, low penalty) ---
SIGMA_A = 5.0 
# Scenario B: Strong Prior (Narrow, high penalty)
SIGMA_B = 0.5

# ====================================================================
# 2. MAP Calculation (Analytic Formula)
# ====================================================================

def calculate_map_mean(mu_prior, sigma_prior, sample_mean, data_std, N):
    sigma_likelihood = data_std**2 / N 
    return (mu_prior/sigma_prior**2 + sample_mean/sigma_likelihood) / \
           (1/sigma_prior**2 + 1/sigma_likelihood)

# Calculate MAP Estimates
MAP_A = calculate_map_mean(PRIOR_MEAN, SIGMA_A, MLE_estimate, DATA_STD, N_SAMPLES)
MAP_B = calculate_map_mean(PRIOR_MEAN, SIGMA_B, MLE_estimate, DATA_STD, N_SAMPLES)

# ====================================================================
# 3. Visualization and Comparison
# ====================================================================

estimates = {
    'True Mean': TRUE_MU,
    'MLE (Sample Mean)': MLE_estimate,
    'MAP (Weak Prior \u03c3=5.0)': MAP_A,
    'MAP (Strong Prior \u03c3=0.5)': MAP_B
}
names = list(estimates.keys())
values = list(estimates.values())

# Calculate the Pull towards the Prior (distance from MLE)
pull_A = abs(MLE_estimate - MAP_A)
pull_B = abs(MLE_estimate - MAP_B)


plt.figure(figsize=(9, 6))

# Plot bars
bars = plt.bar(names, values, color=['gray', 'k', 'skyblue', 'darkred'])

# Highlight Prior Mean
plt.axhline(PRIOR_MEAN, color='b', linestyle='--', label=f'Prior Mean={PRIOR_MEAN:.2f}')

# Annotate the 'pull'
plt.text(2, MAP_A + 0.3, f'Pull: {pull_A:.2f}', ha='center', color='darkred', weight='bold')
plt.text(3, MAP_B + 0.3, f'Pull: {pull_B:.2f}', ha='center', color='darkred', weight='bold')


# Labeling and Formatting
plt.title('MLE vs. MAP Estimation: Prior as Regularization')
plt.xlabel('Estimate Type')
plt.ylabel(r'Parameter Value $\hat{\theta}$')
plt.legend()
plt.grid(True, axis='y')
plt.show()

# --- Analysis Summary ---
print("\n--- MAP vs. MLE Analysis ---")
print(f"Sample Mean (MLE): {MLE_estimate:.3f}")
print(f"Prior Mean: {PRIOR_MEAN:.3f}")
print("---------------------------------------")
print(f"Weak Prior (\u03c3_0=5.0): MAP = {MAP_A:.3f} (Pulled {pull_A:.2f} units)")
print(f"Strong Prior (\u03c3_0=0.5): MAP = {MAP_B:.3f} (Pulled {pull_B:.2f} units)")

print("\nConclusion: The MAP estimate (MAP_B) calculated with the **strong prior** is pulled significantly farther away from the MLE (Sample Mean) and toward the Prior Mean (5.0). This demonstrates that the strong Prior acts as a robust **regularizer**, penalizing the noisy MLE and biasing the final estimate based on prior belief.")
```
**Sample Output:**
```
--- MAP vs. MLE Analysis ---
Sample Mean (MLE): 10.111
Prior Mean: 5.000

---

Weak Prior (σ_0=5.0): MAP = 9.952 (Pulled 0.16 units)
Strong Prior (σ_0=0.5): MAP = 6.217 (Pulled 3.89 units)

Conclusion: The MAP estimate (MAP_B) calculated with the **strong prior** is pulled significantly farther away from the MLE (Sample Mean) and toward the Prior Mean (5.0). This demonstrates that the strong Prior acts as a robust **regularizer**, penalizing the noisy MLE and biasing the final estimate based on prior belief.
```


### Project 3: Visualizing Information Gain (KL Divergence)

* **Goal:** Numerically compute the **KL divergence** between a sequence of beliefs to quantify the information gain from data.
* **Setup:** Use the results from Project 2: Prior ($P$), Posterior A ($Q_A$), and Posterior B ($Q_B$).
* **Steps:**
    1.  Compute the KL divergence from the prior to the first posterior: $D_{\mathrm{KL}}(Q_A || P)$.
    2.  Compute the KL divergence from the prior to the final posterior: $D_{\mathrm{KL}}(Q_B || P)$.
* ***Goal***: Show that $D_{\mathrm{KL}}(Q_B || P) > D_{\mathrm{KL}}(Q_A || P)$, confirming that the final, most informed belief ($Q_B$) contains more information and is statistically "farther" from the uninformed prior than the early belief ($Q_A$).

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Parameters and KL Divergence Function
# ====================================================================

# Analytical KL Divergence for D_KL(Q || P) where both are 1D Gaussians
def kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    """Calculates D_KL(Q || P) for two 1D Gaussians."""
    # D_KL(Q || P) = 0.5 * [log(s_p^2 / s_q^2) + (s_q^2 + (m_q - m_p)^2) / s_p^2 - 1]
    
    sigma_q_sq = sigma_q**2
    sigma_p_sq = sigma_p**2
    
    kl = 0.5 * (
        np.log(sigma_p_sq / sigma_q_sq) 
        + (sigma_q_sq + (mu_q - mu_p)**2) / sigma_p_sq 
        - 1
    )
    return kl

# --- Define fixed Prior (P) ---
MU_PRIOR = 1.0
SIGMA_PRIOR = 2.0 

# --- Define Posterior Scenarios (Q) ---
# Simulates the result of observing more data (mu moves, sigma shrinks)

# Scenario A: Initial Posterior (Q_A) - After a small amount of data
MU_A = 1.5 
SIGMA_A = 1.8 # Still wide

# Scenario B: Final Posterior (Q_B) - After much data (final convergence)
MU_B = 4.5 
SIGMA_B = 0.5 # Much narrower and closer to the true mean

# ====================================================================
# 2. Calculation of Information Gain
# ====================================================================

# Information Gain A: D_KL(Q_A || P)
KL_A = kl_divergence_gaussian(MU_A, SIGMA_A, MU_PRIOR, SIGMA_PRIOR)

# Information Gain B: D_KL(Q_B || P)
KL_B = kl_divergence_gaussian(MU_B, SIGMA_B, MU_PRIOR, SIGMA_PRIOR)

# ====================================================================
# 3. Visualization and Summary
# ====================================================================

kl_values = [KL_A, KL_B]
names = ['Initial Gain (Q_A || P)', 'Final Gain (Q_B || P)']

plt.figure(figsize=(8, 5))

# Plot KL comparison
plt.bar(names, kl_values, color=['skyblue', 'darkred'])

plt.text(0, KL_A + 0.05, f'{KL_A:.3f} bits', ha='center', color='k', weight='bold')
plt.text(1, KL_B + 0.05, f'{KL_B:.3f} bits', ha='center', color='k', weight='bold')

# Labeling and Formatting
plt.title(r'Quantifying Information Gain using KL Divergence $D_{\text{KL}}(Q \mid\mid P)$')
plt.xlabel('Posterior Stage')
plt.ylabel('Information Gain (KL Divergence in Bits)')
plt.grid(True, axis='y')
plt.show()

# --- Analysis Summary ---
print("\n--- KL Divergence (Information Gain) Analysis ---")
print(f"Prior P: \u03bc={MU_PRIOR}, \u03c3={SIGMA_PRIOR}")
print(f"Final Posterior Q_B: \u03bc={MU_B}, \u03c3={SIGMA_B}")
print("-------------------------------------------------")
print(f"KL Divergence D_KL(Q_A || P): {KL_A:.3f} (Low gain)")
print(f"KL Divergence D_KL(Q_B || P): {KL_B:.3f} (High gain)")

print("\nConclusion: The final posterior (Q_B) is statistically much farther from the initial prior (P) than the early posterior (Q_A). This quantitative increase in KL divergence confirms the **information gain** provided by additional data: the final, more confident, and centered belief represents a major shift away from the initial, uninformed belief.")
```
**Sample Output:**
```
--- KL Divergence (Information Gain) Analysis ---
Prior P: μ=1.0, σ=2.0
Final Posterior Q_B: μ=4.5, σ=0.5

---

KL Divergence D_KL(Q_A || P): 0.042 (Low gain)
KL Divergence D_KL(Q_B || P): 2.449 (High gain)

Conclusion: The final posterior (Q_B) is statistically much farther from the initial prior (P) than the early posterior (Q_A). This quantitative increase in KL divergence confirms the **information gain** provided by additional data: the final, more confident, and centered belief represents a major shift away from the initial, uninformed belief.
```


### Project 4: Modeling Dependencies with a Simple Bayesian Network

* **Goal:** Model a dependency structure using a simple **Bayesian Network** and compute a joint probability.
* **Setup:** Define three binary variables ($A, B, C$) with a chain dependency $A \to B \to C$. Define simple Conditional Probability Tables (CPTs) for $P(A)$, $P(B|A)$, and $P(C|B)$.
* **Steps:**
    1.  Write the factored joint probability: $P(A, B, C) = P(A)P(B|A)P(C|B)$.
    2.  Compute the probability of a specific state (e.g., $P(A=1, B=0, C=1)$).
* ***Goal***: Illustrate how the network architecture efficiently breaks down a complex joint distribution into a product of simple, local conditional probabilities, which is the core of graphical models.

#### Python Implementation

```python
import numpy as np

# ====================================================================
# 1. Setup Network and Conditional Probability Tables (CPTs)
# ====================================================================

# Dependency: A -> B -> C (A is root, C is leaf)
# Variables are binary: 0 or 1

# P(A) - Prior for the root node
# Index [0] is P(A=0), Index [1] is P(A=1)
P_A = np.array([0.4, 0.6]) 

# P(B | A) - Conditional Probability Table (CPT)
# Rows: P(B | A=0), P(B | A=1)
# Columns: P(B=0), P(B=1)
# If A=0 (e.g., False), B is likely 0. If A=1 (e.g., True), B is likely 1.
P_B_given_A = np.array([
    [0.8, 0.2],  # P(B=0|A=0), P(B=1|A=0)
    [0.1, 0.9]   # P(B=0|A=1), P(B=1|A=1)
])

# P(C | B) - Conditional Probability Table (CPT)
# Rows: P(C | B=0), P(C | B=1)
# Columns: P(C=0), P(C=1)
P_C_given_B = np.array([
    [0.9, 0.1],  # P(C=0|B=0), P(C=1|B=0)
    [0.2, 0.8]   # P(C=0|B=1), P(C=1|B=1)
])

# ====================================================================
# 2. Joint Probability Calculation (Factoring Rule)
# ====================================================================

# Goal: Compute P(A=1, B=0, C=1)

# Factoring Rule: P(A, B, C) = P(A) * P(B | A) * P(C | B)

# Define the state indices: A=1 (index 1), B=0 (index 0), C=1 (index 1)
A_idx = 1
B_idx = 0
C_idx = 1

# 1. Term P(A=1)
Term_A = P_A[A_idx]

# 2. Term P(B=0 | A=1)
# B's probability (0) given A's state (1)
Term_B_given_A = P_B_given_A[A_idx, B_idx]

# 3. Term P(C=1 | B=0)
# C's probability (1) given B's state (0)
Term_C_given_B = P_C_given_B[B_idx, C_idx]

# Total Joint Probability
P_joint = Term_A * Term_B_given_A * Term_C_given_B

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Joint Probability Calculation using Bayesian Network ---")
print(f"Network Structure: A \u2192 B \u2192 C")
print(f"Factored Joint Probability: P(A, B, C) = P(A) * P(B|A) * P(C|B)")
print("---------------------------------------------------------------")
print(f"Target State: P(A={A_idx}, B={B_idx}, C={C_idx})")
print(f"Term 1: P(A=1) = {Term_A:.2f}")
print(f"Term 2: P(B=0 | A=1) = {Term_B_given_A:.2f}")
print(f"Term 3: P(C=1 | B=0) = {Term_C_given_B:.2f}")

print(f"\nFinal Joint Probability P(1, 0, 1): {P_joint:.4f}")

print("\nConclusion: The Bayesian Network framework allows a complex joint distribution to be computed efficiently by factoring it into a product of local conditional probabilities defined by the graph's topology.")
```
**Sample Output:**
```
--- Joint Probability Calculation using Bayesian Network ---
Network Structure: A → B → C
Factored Joint Probability: P(A, B, C) = P(A) * P(B|A) * P(C|B)

---

Target State: P(A=1, B=0, C=1)
Term 1: P(A=1) = 0.60
Term 2: P(B=0 | A=1) = 0.10
Term 3: P(C=1 | B=0) = 0.10

Final Joint Probability P(1, 0, 1): 0.0060

Conclusion: The Bayesian Network framework allows a complex joint distribution to be computed efficiently by factoring it into a product of local conditional probabilities defined by the graph's topology.
```