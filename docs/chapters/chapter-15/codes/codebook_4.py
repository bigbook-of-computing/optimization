# Source: Optimization/chapter-15/codebook.md -- Block 4

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Boltzmann Policy Function
# ====================================================================

def boltzmann_policy(Q_values, T):
    """
    Calculates the action probabilities \pi(a|s) using the Boltzmann (Softmax) policy.
    \pi(a|s) = exp(Q/T) / sum(exp(Q/T))
    """
    if T <= 1e-6:
        # Near T=0 limit: pure greedy exploitation
        max_q = np.max(Q_values)
        pi = (Q_values == max_q).astype(float)
        return pi / np.sum(pi)
    
    # Numerical stability: subtract max(Q) from Q/T before exponentiating
    scaled_Q = Q_values / T
    exp_scaled_Q = np.exp(scaled_Q - np.max(scaled_Q))
    
    return exp_scaled_Q / np.sum(exp_scaled_Q)

# ====================================================================
# 2. Scenario Testing
# ====================================================================

# Fixed Q-Values for a single state (s) and three actions (a1, a2, a3)
# Best Action is a2 (Q=10)
Q_VALUES = np.array([5.0, 10.0, 0.0]) 
ACTION_NAMES = ['A1 (Q=5)', 'A2 (Q=10, Best)', 'A3 (Q=0)']

# --- Scenario A: High Temperature (High Exploration) ---
T_HIGH = 10.0
PI_HIGH = boltzmann_policy(Q_VALUES, T_HIGH)

# --- Scenario B: Low Temperature (High Exploitation) ---
T_LOW = 0.1
PI_LOW = boltzmann_policy(Q_VALUES, T_LOW)

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

x = np.arange(len(Q_VALUES))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

# Plot 1: High Temperature (Uniform distribution)
rects1 = ax.bar(x - width/2, PI_HIGH, width, label=f'$T={T_HIGH}$ (High Exploration)', color='skyblue')
# Plot 2: Low Temperature (Greedy distribution)
rects2 = ax.bar(x + width/2, PI_LOW, width, label=f'$T={T_LOW}$ (High Exploitation)', color='darkred')

# Labeling and Formatting
ax.set_title('Boltzmann Policy: Temperature Control of Exploration-Exploitation')
ax.set_xlabel('Action')
ax.set_ylabel('Selection Probability $\\pi(a \mid s)$')
ax.set_xticks(x)
ax.set_xticklabels(ACTION_NAMES)
ax.set_ylim(0, 1.0)
ax.legend()
plt.grid(True, axis='y')
plt.show()

print("\n--- Boltzmann Exploration Analysis ---")
print(f"Q-Values: {Q_VALUES}")

print("\n1. High Temperature (T=10.0):")
print(f"  Probabilities: {np.round(PI_HIGH, 3)} (Near Uniform)")
print("  Behavior: High entropy, actions selected almost equally (EXPLORATION).")

print("\n2. Low Temperature (T=0.1):")
print(f"  Probabilities: {np.round(PI_LOW, 3)} (Near Greedy)")
print("  Behavior: Low entropy, action A2 (Q=10) is selected with near-certainty (EXPLOITATION).")

print("\nConclusion: The temperature parameter T successfully controls the exploration-exploitation balance. At high T, the policy flattens, increasing randomness (entropy). At low T, the policy concentrates the probability mass on the best action (Q=10), minimizing effective energy and maximizing deterministic exploitation.")
