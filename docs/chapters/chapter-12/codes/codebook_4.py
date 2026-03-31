# Source: Optimization/chapter-12/codebook.md -- Block 4

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 1. Setup Activation Functions
# ====================================================================

# Sigmoid Activation Function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivative of Sigmoid (The Backpropagation Factor)
def sigmoid_prime(z):
    # \sigma'(z) = \sigma(z) * (1 - \sigma(z))
    s = sigmoid(z)
    return s * (1.0 - s)

# Comparison Activation: ReLU (Rectified Linear Unit)
def relu_prime(z):
    # ReLU derivative is 1 for z > 0, 0 for z < 0
    return np.where(z > 0, 1.0, 0.0)


# ====================================================================
# 2. Data Generation and Gradient Calculation
# ====================================================================

# Range of input z (logit/pre-activation value)
Z_range = np.linspace(-10, 10, 200)

# Calculate the local gradient factor for both functions
Sigmoid_grad = sigmoid_prime(Z_range)
ReLU_grad = relu_prime(Z_range)

# ====================================================================
# 3. Visualization
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot 1: Sigmoid Gradient
plt.plot(Z_range, Sigmoid_grad, 'r-', lw=2, label=r'$\sigma\'(z)$ (Sigmoid Derivative)')

# Plot 2: ReLU Gradient (for comparison)
plt.plot(Z_range, ReLU_grad, 'b--', lw=2, alpha=0.6, label=r'ReLU Derivative ($\phi\'(z)$)')

# Highlight the saturation regions where vanishing occurs
plt.axhline(0, color='k', linestyle='-', lw=0.8)
plt.axvline(-4, color='gray', linestyle=':', label='Saturation Region')
plt.axvline(4, color='gray', linestyle=':')

# Labeling and Formatting
plt.title('Activation Function Derivatives and Vanishing Gradient')
plt.xlabel('Input Logit $z$')
plt.ylabel('Local Gradient Factor $\\phi\'(z)$')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Vanishing Gradient Analysis ---")
print("Observation (Sigmoid): The derivative \u03c3'(z) peaks at 0.25 (when z=0) and rapidly approaches zero as |z| increases beyond \u22484.0.")
print("Interpretation: In deep networks, if early-layer neurons produce large inputs (|z|>4), the local gradient factor becomes minuscule (e.g., <0.01). Multiplying this small factor across many layers causes the error signal to 'vanish,' stalling learning in the crucial early layers.")
print("Contrast (ReLU): The ReLU derivative is constantly 1.0 for positive inputs, which prevents the gradient from vanishing, making it the preferred modern activation.")
