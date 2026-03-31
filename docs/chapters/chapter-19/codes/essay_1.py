# Source: Optimization/chapter-19/essay.md -- Block 1

import torch, torch.nn.functional as F

# --- 1. Setup Input and Weights ---
X = torch.randn(5, 8)  # Input: 5 tokens (elements), each with 8-dimensional features
d_k = 8                # Dimension of the Key/Query space (d_k)

# Initialize weight matrices for the Q, K, and V projections (learned parameters)
W_Q = torch.randn(8, d_k)
W_K = torch.randn(8, d_k)
W_V = torch.randn(8, d_k)

# --- 2. Projection to Q, K, V ---
# These linear transformations occur in parallel
Q = X @ W_Q  # Queries (5x8 matrix)
K = X @ W_K  # Keys (5x8 matrix)
V = X @ W_V  # Values (5x8 matrix)

# --- 3. Compute Attention Weights (Interaction Kernel) ---
# Raw Interaction Potential: Q @ K.T (5x5 matrix of dot products)
Raw_Scores = Q @ K.T

# Scaling: Scores are scaled by 1/sqrt(d_k) for stability
Scaled_Scores = Raw_Scores / (d_k**0.5)

# Softmax: Converts scores (negative energy) into a probability distribution (Attention Matrix)
A = F.softmax(Scaled_Scores, dim=-1)

# --- 4. Aggregate Content ---
# Final Output Y is the weighted sum of the Value vectors
Y = A @ V

# --- Output the Attention Weights ---
print("Input X shape:", X.shape)
print("Attention Matrix A shape:", A.shape)
print("Attention weights (A = Softmax(Q K^T / sqrt(d_k))):\n", A)
