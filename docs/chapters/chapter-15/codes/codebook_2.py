# Source: Optimization/chapter-15/codebook.md -- Block 2

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Policy and Environment (Conceptual)
# ====================================================================

# We model a single state (s) and two actions (a1, a2)
# Policy is parameterized by a single logit \theta_1 (weights are not explicit here)
N_ACTIONS = 2
N_STEPS = 100

# Conceptual Rewards (representing a sampled episode)
# Return G is assumed to be known for each action in the episode
RETURNS = np.array([10.0, 5.0, 15.0, 2.0, 8.0, 1.0])
ACTIONS = np.array([0, 1, 0, 1, 0, 1]) # 0=a1, 1=a2 (taken actions)

# Conceptual Policy (Softmax over logits \theta)
# The logits are conceptual parameters defining the policy preference
# Let's assume the current logits are [0.5, 0.0]
LOGITS = np.array([0.5, 0.0])

def softmax(logits):
    """Calculates action probabilities \pi(a|s)."""
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def get_policy_gradient_log(logits, action_index):
    """
    Calculates the gradient of the log-policy: \nabla_{\theta} \ln \pi(a|s).
    For a simple Softmax policy, this is: e_a - \pi(a|s)
    where e_a is a one-hot vector for action a.
    """
    pi_a = softmax(logits)
    e_a = np.zeros_like(logits)
    e_a[action_index] = 1.0
    return e_a - pi_a

# ====================================================================
# 2. Gradient Estimation Trials
# ====================================================================

# 1. Conceptual Baseline (Value function V(s))
# We approximate V(s) as the average return (baseline)
BASELINE = np.mean(RETURNS)

# Estimate the Gradient for the entire sampled episode (sum over time steps t)
GRADIENT_NO_BASELINE = np.zeros(N_ACTIONS)
GRADIENT_WITH_BASELINE = np.zeros(N_ACTIONS)
Variance_Check = []

for G_t, a_t in zip(RETURNS, ACTIONS):
    # Get the policy gradient term for the action taken (a_t)
    policy_grad_log = get_policy_gradient_log(LOGITS, a_t)
    
    # --- Trial A: No Baseline (High Variance) ---
    # Gradient = \nabla \ln \pi * G_t
    GRADIENT_NO_BASELINE += policy_grad_log * G_t
    Variance_Check.append(policy_grad_log * G_t)
    
    # --- Trial B: With Baseline (Low Variance) ---
    # Advantage A_t = G_t - b(s)
    Advantage_t = G_t - BASELINE
    
    # Gradient = \nabla \ln \pi * A_t
    GRADIENT_WITH_BASELINE += policy_grad_log * Advantage_t
    
# Calculate Variance (Crucial step for demonstration)
# We look at the variance of the individual contributions to the gradient estimate (Policy * Return/Advantage)
Variance_Check_array = np.array(Variance_Check)
# We specifically check the variance of the components that make up the final sum
VAR_CONTRIB_NO_BASELINE = np.var(Variance_Check_array[:, 0]) 

# Recalculate Variance_Check for baseline case
Variance_Check_baseline = []
for G_t, a_t in zip(RETURNS, ACTIONS):
    policy_grad_log = get_policy_gradient_log(LOGITS, a_t)
    Advantage_t = G_t - BASELINE
    Variance_Check_baseline.append(policy_grad_log * Advantage_t)
VAR_CONTRIB_WITH_BASELINE = np.var(np.array(Variance_Check_baseline)[:, 0])

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Policy Gradient Estimation (REINFORCE) ---")
print(f"Average Return (Baseline): {BASELINE:.2f}")
print("-------------------------------------------------------")

print("\nTrial A: Gradient Estimate (NO BASELINE):")
print(f"  Gradient Components: {np.round(GRADIENT_NO_BASELINE, 4)}")
print(f"  Variance of Contributions (\u03b8_1): {VAR_CONTRIB_NO_BASELINE:.4f} (High)")

print("\nTrial B: Gradient Estimate (WITH BASELINE):")
print(f"  Gradient Components: {np.round(GRADIENT_WITH_BASELINE, 4)}")
print(f"  Variance of Contributions (\u03b8_1): {VAR_CONTRIB_WITH_BASELINE:.4f} (Reduced)")

print("\nConclusion: The variance of the individual contributions to the gradient estimate (Trial B) is significantly lower when the average return (baseline) is subtracted from the actual return. This confirms the mathematical principle of variance reduction: the baseline removes common mode noise, making the learning process more stable and enabling faster convergence.")
