# Source: Optimization/chapter-15/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

n_states = 25
Q = np.zeros((n_states, 4))  # 25 states, 4 actions (up, right, down, left)
gamma, alpha, eps = 0.9, 0.1, 0.1

def step(s, a):
    # Converts state index 's' (0-24) to grid coordinates (row, col)
    row, col = divmod(s,5)
    
    # 1. Apply action 'a'
    if a==0: row = max(0,row-1)    # Up
    elif a==1: col = min(4,col+1)  # Right
    elif a==2: row = min(4,row+1)  # Down
    else: col = max(0,col-1)      # Left
    
    # 2. Determine new state s2 and reward
    s2 = row*5+col
    # Reward is -1 per step, 0 upon reaching the goal (state 24)
    reward = -1 if s2!=24 else 0
    return s2, reward

for ep in range(500):
    s=0 # Start at state 0
    while s!=24: # Loop until the goal state is reached
        
        # 1. Action Selection (Exploration/Exploitation - Epsilon-Greedy)
        # Selects a random action (exploration) with probability epsilon (eps)
        # Otherwise, selects the action with the maximum current Q-value (exploitation)
        a = np.random.choice(4) if np.random.rand()<eps else np.argmax(Q[s])
        
        s2,r = step(s,a) # Take action, observe next state (s2) and reward (r)
        
        # 2. Q-Value Update (Temporal-Difference Learning)
        # Q(s,a) <-- Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
        # This is the core Q-Learning update rule (Section 15.5)
        Q[s,a] += alpha*(r + gamma*np.max(Q[s2]) - Q[s,a])
        
        s=s2 # Move to the new state
        
# Visualization of the final learned state values
# max(Q, axis=1) gives the maximum Q-value for each state V_max(s)
plt.imshow(Q.max(1).reshape(5,5), cmap='plasma')
plt.title("Learned Value Landscape $V_{\\max}(s)$")
plt.colorbar(label="Max Expected Return ($Q^*$)")
plt.show()
