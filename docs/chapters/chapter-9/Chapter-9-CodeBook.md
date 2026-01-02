# Chapter 9: Bayesian Thinking and Inference

## Project 1: Visualizing Prior, Likelihood, and Posterior

-----

### Definition: Visualizing Prior, Likelihood, and Posterior

The goal is to implement the core components of **Bayes' Theorem**—the **Prior ($P(\boldsymbol{\theta})$)**, the **Likelihood ($\mathcal{L}(\boldsymbol{\theta})$)**, and the resulting **Posterior ($P(\boldsymbol{\theta} | \mathcal{D})$)**—and visualize how the data (Likelihood) updates the initial belief (Prior).

### Theory: Bayes' Theorem

Bayes' Theorem is the fundamental law of learning, describing how prior uncertainty is transformed into posterior knowledge after observing data ($\mathcal{D}$):

$$P(\boldsymbol{\theta} | \mathcal{D}) = \frac{P(\mathcal{D} | \boldsymbol{\theta}) P(\boldsymbol{\theta})}{P(\mathcal{D})} \quad \text{or, more simply:} \quad P(\boldsymbol{\theta} | \mathcal{D}) \propto \mathcal{L}(\boldsymbol{\theta}) P(\boldsymbol{\theta})$$

  * **Prior ($P(\boldsymbol{\theta})$):** Represents our belief about the parameters $\boldsymbol{\theta}$ *before* seeing the data. We typically assume a **Gaussian Prior**.
  * **Likelihood ($\mathcal{L}(\boldsymbol{\theta})$):** Represents the probability of observing the data $\mathcal{D}$ for a given set of parameters $\boldsymbol{\theta}$. This is the information provided by the simulation data.
  * **Posterior ($P(\boldsymbol{\theta} | \mathcal{D})$):** The final, updated belief. In this simple Gaussian-Gaussian case, the posterior is also Gaussian (**conjugate prior** property). Its mean and variance are tighter than the prior, reflecting reduced uncertainty.

-----

### Extensive Python Code and Visualization

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

-----

## Project 2: Maximum A Posteriori (MAP) Estimation vs. Maximum Likelihood Estimation (MLE)

-----

### Definition: Maximum A Posteriori (MAP) Estimation vs. Maximum Likelihood Estimation (MLE)

The goal is to compute the **Maximum Likelihood Estimate (MLE)** and the **Maximum A Posteriori (MAP) Estimate** for a parameter and demonstrate that the **Prior acts as a regularization term** that pulls the MAP estimate away from the raw data average (MLE).

### Theory: Prior as Regularizer

1.  **Maximum Likelihood (MLE):** The point estimate that maximizes the likelihood of the data $\mathcal{L}(\boldsymbol{\theta})$. For the Gaussian mean, $\hat{\boldsymbol{\theta}}_{\text{MLE}} = \text{Sample Mean}$.
2.  **Maximum A Posteriori (MAP):** The point estimate that maximizes the posterior $P(\boldsymbol{\theta} | \mathcal{D})$:
    $$\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} [\ln \mathcal{L}(\boldsymbol{\theta}) + \ln P(\boldsymbol{\theta})]$$

The term $\ln P(\boldsymbol{\theta})$ acts as a **regularization penalty**. A strong (narrow) prior effectively pulls the MAP estimate toward the prior mean, while a weak (wide) prior allows the MAP estimate to remain close to the MLE (the sample mean).

-----

### Extensive Python Code and Visualization

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

-----

## Project 3: Quantifying Information Gain (KL Divergence)

-----

### Definition: Quantifying Information Gain (KL Divergence)

The goal is to quantify the **information gain** over the course of an inference process using the **Kullback-Leibler (KL) Divergence**. The process involves comparing the statistical distance between an uninformed **Prior** and two progressively more informed **Posterior** distributions.

### Theory: KL Divergence and Statistical Distance

The KL divergence, $D_{\mathrm{KL}}(P||Q)$, measures the statistical difference between two distributions, $P$ and $Q$. In Bayesian terms, we measure the distance between the **Posterior ($Q$)** and the **Prior ($P$):**

$$D_{\mathrm{KL}}(Q || P) = \int Q(\boldsymbol{\theta}) \ln \frac{Q(\boldsymbol{\theta})}{P(\boldsymbol{\theta})} d\boldsymbol{\theta}$$

If the inference process successfully reduces uncertainty and moves the belief closer to the true value, the **Final Posterior ($Q_B$)** should be statistically *farther* from the uninformed **Prior ($P$)** than an early estimate ($Q_A$).

For two Gaussians, $P=\mathcal{N}(\mu_p, \sigma_p^2)$ and $Q=\mathcal{N}(\mu_q, \sigma_q^2)$, the KL divergence simplifies to the following analytical formula:

$$D_{\mathrm{KL}}(Q || P) = \frac{1}{2} \left[ \ln\left(\frac{\sigma_p^2}{\sigma_q^2}\right) + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{\sigma_p^2} - 1 \right]$$

-----

### Extensive Python Code and Visualization

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

-----

## Project 4: Modeling Dependencies with a Simple Bayesian Network

-----

### Definition: Modeling Dependencies with a Simple Bayesian Network

The goal is to model a dependency structure using a simple **Bayesian Network (BN)** and calculate the joint probability of a specific state using the **factoring rule**.

### Theory: Bayesian Networks and Factoring

A Bayesian Network uses a **Directed Acyclic Graph (DAG)** to represent the conditional dependencies between a set of variables. The structure allows the complex **joint probability** to be factored into a product of simpler **conditional probabilities**:

$$P(X_1, X_2, \dots, X_N) = \prod_{i=1}^N P(X_i \mid \text{Parents}(X_i))$$

For the simple chain dependency $A \to B \to C$:

$$P(A, B, C) = P(A) P(B \mid A) P(C \mid B)$$

The problem is solved by defining the **Conditional Probability Tables (CPTs)** and performing a direct multiplication.

-----

### Extensive Python Code

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


