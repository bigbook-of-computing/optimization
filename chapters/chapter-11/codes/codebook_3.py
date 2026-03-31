# Source: Optimization/chapter-11/codebook.md -- Block 3

import numpy as np

# ====================================================================
# 1. Setup HMM Parameters (Two Hidden States: Cold=0, Hot=1)
# ====================================================================

# Hidden States: Cold (0), Hot (1)
# Observations: Low Energy (0), High Energy (1)

# 1. Transition Matrix (P(z_t | z_{t-1}))
# Rows: z_{t-1} (Start), Columns: z_t (End)
# Favors staying in the same state (P_Cold_to_Cold = 0.9)
A = np.array([
    [0.9, 0.1],  # Cold -> Cold (0.9), Cold -> Hot (0.1)
    [0.2, 0.8]   # Hot -> Cold (0.2), Hot -> Hot (0.8)
])

# 2. Observation Matrix (P(x_t | z_t))
# Rows: z_t (Hidden State), Columns: x_t (Observation)
# Cold state strongly predicts Low Energy, Hot state strongly predicts High Energy
B = np.array([
    [0.9, 0.1],  # Cold predicts Low E (0.9), High E (0.1)
    [0.3, 0.7]   # Hot predicts Low E (0.3), High E (0.7)
])

# Initial Probability (Prior belief at t=0)
PI = np.array([0.7, 0.3]) # Start with a strong belief in the Cold state

# ====================================================================
# 2. Forward Algorithm Implementation (State Estimation)
# ====================================================================

# Sequence of observations: Low E (0) -> High E (1)
# We track the belief \alpha_t at each step
Observations = [0, 1] 
Belief_History = [PI.copy()]

# The forward algorithm loop
belief = PI.copy() # Current belief P(z_t | x_1:t)

for t, x_t in enumerate(Observations):
    # --- 1. Prediction Step (Predict next state based on transition dynamics) ---
    # Prediction: P(z_t | x_1:t-1) = sum_{z_{t-1}} P(z_t | z_{t-1}) * P(z_{t-1} | x_1:t-1)
    # Predicted_belief = belief_t-1 @ A (matrix multiplication)
    predicted_belief = belief @ A
    
    # --- 2. Observation Update (Correct prediction with noisy data) ---
    # Update: P(z_t | x_1:t) \propto P(x_t | z_t) * Predicted_belief
    
    # Likelihood of observing x_t for each state z_t
    likelihood_x_t = B[:, x_t] 
    
    # Updated belief (unnormalized)
    unnorm_belief = predicted_belief * likelihood_x_t
    
    # Normalization (Crucial step for proper probability)
    belief = unnorm_belief / np.sum(unnorm_belief)
    
    Belief_History.append(belief)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

df_belief = pd.DataFrame(Belief_History, columns=['P(Cold)', 'P(Hot)'])
df_belief.index.name = 'Time Step'

print("--- HMM Forward Algorithm: State Estimation ---")
print(f"Observation Sequence: {Observations}")
print(df_belief.to_markdown(floatfmt=".3f"))

# Plot the evolution of belief
df_belief.plot(kind='line', style=['-o', '--s'], figsize=(8, 5))
plt.title('HMM Belief Evolution: P($z_t$ | $x_{1:t}$)')
plt.xlabel('Time Step (t)')
plt.ylabel('Belief Probability')
plt.xticks(np.arange(len(Belief_History)), labels=['t=0 (Prior)', 't=1 (Obs=0)', 't=2 (Obs=1)'])
plt.ylim(0, 1.0)
plt.grid(True)
plt.show()

print("\nConclusion: The belief system starts strongly Cold (0.7). After the second observation (x=1, High Energy), the belief in the Hot state increases sharply (from 0.3 to \u22480.64), demonstrating the core HMM task of updating the hidden state probability based on a sequence of noisy, external observations.")
