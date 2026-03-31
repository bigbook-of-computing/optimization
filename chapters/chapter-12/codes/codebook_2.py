# Source: Optimization/chapter-12/codebook.md -- Block 2

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Functions and Network Structure
# ====================================================================

# Activation Function: Sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivative of Sigmoid (Local Gradient Term)
def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

# --- Network Structure and Data ---
N_INPUT = 3  # Features + Bias
N_HIDDEN = 4
N_OUTPUT = 1

# Weights (W_jk: from layer j to layer k)
# W_H_O: Weights from Hidden Layer (H) to Output Layer (O)
W_H_O = np.random.randn(N_HIDDEN, N_OUTPUT) * 0.1 

# W_I_H: Weights from Input Layer (I) to Hidden Layer (H)
W_I_H = np.random.randn(N_INPUT, N_HIDDEN) * 0.1

# Input and Target
X_input = np.array([0.5, 0.2, 1.0]) # [x1, x2, bias]
y_target = np.array([0.8])

# ====================================================================
# 2. Forward Pass (Compute Activations z and a)
# ====================================================================

# Hidden Layer Forward Pass
z_H = X_input @ W_I_H
a_H = sigmoid(z_H) # Activations of the hidden layer

# Output Layer Forward Pass
z_O = a_H @ W_H_O
a_O = sigmoid(z_O) # Final output prediction (y_hat)

# Loss: Mean Squared Error (for demonstration)
L_loss = 0.5 * np.sum((a_O - y_target)**2)

# ====================================================================
# 3. Backward Pass - Core Delta Calculation
# ====================================================================

# --- Step 1: Calculate Delta for the Output Layer (\delta_O) ---
# \delta_O = (a_O - y_target) * \phi'(z_O)
dL_da_O = a_O - y_target
phi_prime_O = sigmoid_prime(z_O)
delta_O = dL_da_O * phi_prime_O

# --- Step 2: Backpropagate Error to the Hidden Layer (\delta_H) ---
# \delta_j = \phi'(z_j) * \sum_{k} w_{jk} \delta_k

# 1. Backpropagated Error Term (Sum over k): \sum_{k} w_{jk} \delta_k
# W_H_O.T @ delta_O is the required matrix multiplication
backprop_error_H = W_H_O.T @ delta_O

# 2. Local Gradient Term: \phi'(z_H)
phi_prime_H = sigmoid_prime(z_H)

# 3. Final Hidden Delta: \delta_H
delta_H = phi_prime_H * backprop_error_H

# ====================================================================
# 4. Analysis and Verification
# ====================================================================

print("--- Backpropagation Core (\u03b4) Calculation ---")
print(f"Output Prediction (a_O): {a_O[0]:.4f}")
print(f"Loss (L): {L_loss:.4f}")
print("-------------------------------------------------------")
print(f"Output Layer Delta (\u03b4_O): {delta_O[0]:.4f}")
print(f"Hidden Layer Delta (\u03b4_H): {np.round(delta_H, 4)}")
print("-------------------------------------------------------")

print("\nConclusion: The calculation successfully determined the error signal (\u03b4_H) for the hidden layer. This process, driven by the chain rule, provides the necessary local gradient information (d L / d w_ij) required for the subsequent weight updates.")
