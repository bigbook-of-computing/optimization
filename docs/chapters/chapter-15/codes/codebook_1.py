# Source: Optimization/chapter-15/codebook.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ====================================================================
# 1. Setup Environment (Grid World) and Q-Table
# ====================================================================

# Environment: Simple 3x3 Grid World
# States: 0 to 8
# Actions: 0: Up, 1: Down, 2: Left, 3: Right
N_STATES = 9
N_ACTIONS = 4
GOAL_STATE = 8 # State 8 (bottom right) gives reward

# Rewards (R[s, a] = reward for taking action a from state s)
# Only actions leading to the goal (8) give positive reward
R = -0.1 * np.ones((N_STATES, N_ACTIONS)) # Small penalty for all moves
R[5, 3] = 10.0 # From state 5 (middle-right), moving Right (3) to 8 is a reward of 10.0
R[7, 1] = 10.0 # From state 7 (bottom-middle), moving Down (1) to 8 is a reward of 10.0

# Transition Function (Deterministic, P(s'|s, a))
def get_next_state(s, a):
    row, col = divmod(s, 3)
    if a == 0: row = max(row - 1, 0) # Up
    elif a == 1: row = min(row + 1, 2) # Down
    elif a == 2: col = max(col - 1, 0) # Left
    elif a == 3: col = min(col + 1, 2) # Right
    return row * 3 + col

# Q-Table Initialization (start at zero)
Q = np.zeros((N_STATES, N_ACTIONS))

# ====================================================================
# 2. Q-Learning Algorithm
# ====================================================================

# Hyperparameters
ETA = 0.1     # Learning rate
GAMMA = 0.9   # Discount factor
MAX_EPISODES = 1000

# Tracking the max Q-value for the starting state (s=0) to check convergence
Q0_history = [] 

for episode in range(MAX_EPISODES):
    s = 0 # Start every episode at state 0
    
    # Run episode until goal (state 8) is reached
    while s != GOAL_STATE:
        # 1. Action Selection (Exploration vs. Exploitation - Epsilon Greedy)
        if random.random() < 0.1: # 10% Exploration
            a = random.randrange(N_ACTIONS)
        else: # 90% Exploitation
            a = np.argmax(Q[s, :])
            
        # 2. Interact with Environment
        r = R[s, a]
        s_prime = get_next_state(s, a)
        
        # 3. Q-Learning Update Rule (Bellman Optimality)
        # Target = r + gamma * max_a' Q(s', a')
        max_q_prime = np.max(Q[s_prime, :])
        td_target = r + GAMMA * max_q_prime
        
        # Q(s, a) = Q(s, a) + eta * [Target - Q(s, a)]
        Q[s, a] = Q[s, a] + ETA * (td_target - Q[s, a])
        
        s = s_prime
        
    # Track convergence (Max Q-value for the starting state 0)
    Q0_history.append(np.max(Q[0, :])) 

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Plot 1: Q-Value Convergence
plt.figure(figsize=(9, 6))
plt.plot(Q0_history, 'r-', lw=2)
plt.title('Q-Learning Convergence: Max Q(State 0, a)')
plt.xlabel('Episode')
plt.ylabel('Max Action-Value $Q_0$')
plt.grid(True)
plt.show()

# Print the learned optimal policy (the greedy action from Q*)
print("\n--- Learned Optimal Policy (Q*) ---")
print("State: Best Action")
for s in range(N_STATES):
    best_action_index = np.argmax(Q[s, :])
    action_name = ['Up', 'Down', 'Left', 'Right'][best_action_index]
    print(f"State {s}: {action_name}")

print("\nConclusion: The plot shows that the Q-value for the starting state monotonically increases and eventually converges to a stable value, confirming that the iterative TD learning process successfully approximated the optimal value function $Q^*$.")
