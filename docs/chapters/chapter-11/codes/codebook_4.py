# Source: Optimization/chapter-11/codebook.md -- Block 4

import numpy as np

# ====================================================================
# 1. Setup Conceptual Functions
# ====================================================================

# We model the ELBO components conceptually to show the maximization logic.
# Assume the true model P is a known function of a single parameter \theta.

# True Model Parameters
TRUE_THETA = 5.0
DATA = 100.0 # Hypothetical summary statistic of the data

# 1. Energy Term (ln P(D, \theta))
# Conceptual Joint Likelihood: Penalizes deviation from the data (DATA)
def log_joint_likelihood(theta, data_summary):
    # Penalizes distance from data center (e.g., L2 loss)
    return -0.5 * (theta - data_summary)**2

# 2. Entropy Term (-ln Q(\theta))
# Conceptual Entropy for a simple Gaussian Q ~ N(\mu_Q, \sigma_Q)
# The Gaussian entropy is H(Q) = 0.5 * log(2\pi e \sigma_Q^2)
def entropy(sigma_q):
    # We use -H(Q) for -E_Q[ln Q] in the ELBO formula
    return 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_q**2)

# ====================================================================
# 2. ELBO Calculation and Optimization Logic
# ====================================================================

def calculate_elbo(mu_q, sigma_q, data_summary=DATA):
    """
    Conceptual ELBO for a Gaussian Q: ELBO = E_Q[ln P(D,\theta)] - E_Q[ln Q(\theta)]
    """
    # 1. Energy Term: E_Q [ln P(D,\theta)] - We approximate this with the likelihood at mu_Q
    # In a full VI, this is calculated with Monte Carlo sampling over Q.
    energy_term = log_joint_likelihood(mu_q, data_summary) 
    
    # 2. Entropy Term: E_Q [ln Q(\theta)] = -H(Q)
    # The term -E_Q[ln Q] is the negative entropy
    neg_entropy_term = -entropy(sigma_q)
    
    return energy_term - neg_entropy_term

# --- Optimization Scenario ---
# We track ELBO evolution as Q is optimized toward the true Posterior.

MU_Q_INIT = 0.0 # Initial guess for Q's mean
SIGMA_Q_INIT = 4.0 # Initial guess for Q's standard deviation (wide)

# We conceptualize the optimization:
# Step 1: Initial (Poor) Q
ELBO_INIT = calculate_elbo(MU_Q_INIT, SIGMA_Q_INIT)

# Step 2: Optimized (Better) Q
# The mean moves toward the data center (100) and the variance shrinks.
MU_Q_OPT = 90.0 
SIGMA_Q_OPT = 1.0 
ELBO_OPT = calculate_elbo(MU_Q_OPT, SIGMA_Q_OPT)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

elbo_values = [ELBO_INIT, ELBO_OPT]
names = ['Initial Q (Low ELBO)', 'Optimized Q (High ELBO)']

print("--- Variational Inference (VI) and ELBO Maximization ---")

# Plot ELBO evolution
plt.figure(figsize=(8, 5))
plt.bar(names, elbo_values, color=['skyblue', 'darkgreen'])
plt.title(r'ELBO Maximization: Inference as Optimization')
plt.ylabel('Evidence Lower Bound (ELBO)')
plt.grid(True, axis='y')
plt.show()

print("\nConclusion: The ELBO increases from the initial, uninformed distribution (Q_INIT) to the optimized distribution (Q_OPT). This demonstrates that **Variational Inference** solves the inference problem by framing it as a deterministic **maximization of the ELBO**, which is computationally equivalent to minimizing the statistical distance (KL divergence) between the approximation Q and the true Posterior P.")
