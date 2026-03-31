# Source: Optimization/chapter-15/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Conceptual Objective Function
# ====================================================================

# Conceptual components of the Objective Function J for a policy \pi
# These are the expected values under the current policy \pi.
# We track how the total objective changes with the \alpha parameter.

# 1. Expected Reward (Energy Term)
# This is fixed by the environment and is generally high when the policy is stable.
REWARD_TERM_BASE = 50.0

# 2. Entropy Term (Exploration Term)
# This is maximized when the policy is random (uniform probability)
# Assume H(\pi) ranges from 0 (deterministic) to 2.0 (fully random)
ENTROPY_TERM_HIGH = 1.5
ENTROPY_TERM_LOW = 0.5

# ====================================================================
# 2. Free-Energy/MaxEnt Scenarios
# ====================================================================

# --- Scenario A: Low Alpha (Low Exploration, High Exploitation) ---
ALPHA_A = 0.1 
# The system favors reward. Assumes a policy \pi_A that is high reward/low entropy.
OBJECTIVE_A = REWARD_TERM_BASE + ALPHA_A * ENTROPY_TERM_LOW

# --- Scenario B: High Alpha (High Exploration, Low Exploitation) ---
ALPHA_B = 5.0
# The system favors entropy. Assumes a policy \pi_B that is slightly lower reward/higher entropy.
REWARD_TERM_B = 45.0 # Lower reward achieved
OBJECTIVE_B = REWARD_TERM_B + ALPHA_B * ENTROPY_TERM_HIGH

# --- Scenario C: Zero Alpha (Standard RL / Pure Energy Minimization) ---
ALPHA_C = 0.0
OBJECTIVE_C = REWARD_TERM_BASE + ALPHA_C * ENTROPY_TERM_LOW

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

alphas = [ALPHA_C, ALPHA_A, ALPHA_B]
objectives = [OBJECTIVE_C, OBJECTIVE_A, OBJECTIVE_B]
names = [r'Standard RL ($\alpha=0$)', r'Low Entropy Cost ($\alpha=0.1$)', r'High Entropy Cost ($\alpha=5.0$)']

plt.figure(figsize=(9, 6))

# Plot Objectives (analogous to negative Free Energy)
bars = plt.bar(names, objectives, color=['gray', 'skyblue', 'darkred'])

# Annotate the components
plt.text(0, objectives[0] + 1, f'Reward:{REWARD_TERM_BASE:.1f}', ha='center', color='k')
plt.text(2, OBJECTIVE_B - 4, f'Reward:{REWARD_TERM_B:.1f}', ha='center', color='w', fontweight='bold')
plt.text(2, OBJECTIVE_B - 8, f'Entropy:{ALPHA_B * ENTROPY_TERM_HIGH:.1f}', ha='center', color='w', fontweight='bold')

# Labeling and Formatting
plt.title(r'MaxEnt RL Objective Function $J_{\alpha}$: Reward-Entropy Trade-Off')
plt.ylabel('Expected Return / Objective Value (J)')
plt.grid(True, axis='y')
plt.show()

print("\n--- MaxEnt RL (Free-Energy) Analysis ---")
print(f"Objective A (\u03b1=0.1): J = {OBJECTIVE_A:.2f} (High Reward, Low Entropy Contribution)")
print(f"Objective B (\u03b1=5.0): J = {OBJECTIVE_B:.2f} (Lower Reward, High Entropy Contribution)")

print("\nConclusion: The Objective B, despite having a lower raw reward (45.0), achieves a higher total objective value (52.5) because the high alpha factor drastically increases the contribution of the entropy term. This confirms that MaxEnt RL formalizes the exploration-exploitation balance as a thermodynamic trade-off, where the policy minimizes an effective Free Energy by balancing the system's energy (negative reward) and entropy (randomness).")
