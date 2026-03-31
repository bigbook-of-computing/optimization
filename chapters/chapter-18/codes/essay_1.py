# Source: Optimization/chapter-18/essay.md -- Block 1

import torch

# --- 1. Graph Structure (Adjacency Matrix) ---
# Defines a simple 3-node graph connected in a triangle (Nodes 1-2-3 all linked)
A = torch.tensor([[0,1,1],[1,0,1],[1,1,0]],dtype=torch.float32) 
# A is 3x3: A[i,j] = 1 if an edge exists (A[1,2]=1, A[2,1]=1, etc.)

# --- 2. Initial Node Features (State) ---
# H is 3x4: 3 nodes, each with a 4-dimensional feature vector (h_i)
H = torch.randn(3,4)  
# H is the initial state of the system

# --- 3. Trainable Weights (Filter) ---
# W is 4x4: The learned linear transformation matrix (weight matrix)
W = torch.randn(4,4) 

# --- 4. Message Passing and Update Loop ---
for _ in range(3):
    # Step 1: Aggregation/Message Passing (A @ H)
    # A @ H: Each node's new features are a sum of its neighbors' current features.
    # Step 2: Feature Transformation ((A @ H) @ W)
    # The aggregated features are transformed by the learned weights W.
    # Step 3: Non-linearity (torch.relu)
    H = torch.relu(A @ H @ W)   # General form of a GNN layer

print(H)
