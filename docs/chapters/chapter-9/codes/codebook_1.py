# Source: Optimization/chapter-9/codebook.md -- Block 1

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
