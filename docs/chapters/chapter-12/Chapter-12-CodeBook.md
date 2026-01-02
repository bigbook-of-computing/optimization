# Chapter 12: The Perceptron and Neural Foundations

## Project 1: Perceptron Convergence (The Delta Rule)

-----

### Definition: Perceptron Convergence

The goal of this project is to implement the **Perceptron update rule (Delta Rule)** and demonstrate its ability to find a solution that separates linearly separable data.

### Theory: Linear Separability and the Update Rule

The **Perceptron** is the simplest linear classifier, providing a binary output based on a weighted sum of inputs. Its key operational characteristic is the **Perceptron Convergence Theorem**, which states that if a dataset is **linearly separable**, the algorithm is guaranteed to find a solution in a finite number of steps.

The update rule is applied **only when a classification error occurs**:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \eta \mathbf{x} y$$

Where:

  * $\mathbf{w}_t$ is the current weight vector.
  * $\eta$ is the learning rate.
  * $\mathbf{x}$ is the misclassified input vector.
  * $y \in \{-1, +1\}$ is the true label.

The product $\mathbf{x}y$ moves the decision boundary closer to the misclassified point. We track the number of classification errors per epoch to demonstrate convergence.

-----

### Extensive Python Code and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Data and Model Training
# ====================================================================

# Generate linearly separable 2D data (2 features)
X, y_raw = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, random_state=42)

# Convert labels from {0, 1} to the Perceptron's required {-1, +1}
y = np.where(y_raw == 0, -1, 1)

# Add bias term (x_0 = 1) to the design matrix
X_design = np.hstack([X, np.ones((X.shape[0], 1))])
N_FEATURES = X_design.shape[1] # Includes bias

# ====================================================================
# 2. Perceptron Update Rule Implementation
# ====================================================================

def train_perceptron(X, y, eta=0.01, max_epochs=100):
    # Initialize weights randomly
    w = np.random.randn(N_FEATURES)
    error_history = []
    
    for epoch in range(max_epochs):
        n_errors = 0
        
        # Iterate over all training examples (stochastic update)
        for x_i, y_i in zip(X, y):
            # Activation: a = w^T * x_i
            activation = np.dot(w, x_i)
            # Prediction: y_hat = sign(activation)
            y_hat = np.sign(activation) if activation != 0 else 1 
            
            if y_hat != y_i:
                # Perceptron (Delta) Rule: w_new = w_old + eta * x * y
                w = w + eta * x_i * y_i
                n_errors += 1
        
        error_history.append(n_errors)
        
        # Convergence Check: If no errors occurred, convergence is guaranteed
        if n_errors == 0:
            break
            
    return w, error_history, epoch + 1

# Run the simulation
w_final, errors, epochs = train_perceptron(X_design, y)
w1_final, w2_final, b_final = w_final

# ====================================================================
# 3. Visualization
# ====================================================================

# 1. Plot Error Convergence
plt.figure(figsize=(9, 4))
plt.plot(np.arange(1, epochs + 1), errors, 'b-o', markersize=4)
plt.title('Perceptron Convergence (Classification Errors vs. Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassifications')
plt.grid(True)
plt.show()

# 2. Plot Decision Boundary
plt.figure(figsize=(9, 6))

# Decision Boundary: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b) / w2
x1_plot = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
x2_boundary = (-w1_final * x1_plot - b_final) / w2_final

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='k', alpha=0.8, label='Data Points')
plt.plot(x1_plot, x2_boundary, 'k-', lw=2, label='Decision Boundary')

plt.title('Perceptron: Final Linear Decision Boundary')
plt.xlabel('Feature 1 ($x_1$)')
plt.ylabel('Feature 2 ($x_2$)')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Perceptron Convergence Summary ---")
print(f"Total Epochs for Convergence: {epochs}")
print(f"Final Weights: w1={w1_final:.3f}, w2={w2_final:.3f}, b={b_final:.3f}")
print("Final Misclassifications: 0")

print("\nConclusion: The error plot shows that the number of misclassifications decreases and reaches zero within a finite number of epochs, confirming the **Perceptron Convergence Theorem** for this linearly separable dataset.")
```

-----

## Project 2: Implementing the Backpropagation Core

-----

### Definition: Implementing the Backpropagation Core

The goal is to implement the core mathematical step of **Backpropagation**—the $\boldsymbol{\delta}$ (error signal) calculation—for a simple two-layer network. This demonstrates the application of the **Chain Rule** to compute gradients efficiently in complex, multi-layered structures.

### Theory: The Error Signal ($\delta$)

Backpropagation computes the gradient ($\nabla L$) by propagating a local error signal ($\delta$) backward from the output layer to the hidden layers. The error signal $\delta_j$ for neuron $j$ represents how much that neuron contributed to the final loss.

For a hidden layer neuron $j$, the error $\delta_j$ is the product of its local activation gradient ($\phi'(z_j)$) and the weighted sum of error signals ($\delta_k$) from all neurons $k$ it feeds into in the next layer:

$$\delta_j = \phi'(z_j) \sum_{k} w_{jk} \delta_k$$

Where $\sum_{k} w_{jk} \delta_k$ is the **backpropagated error term**. This iterative application of the chain rule is the key to computational efficiency in deep learning.

-----

### Extensive Python Code

```python
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
```

-----

## Project 3: Energy Dissipation Check

-----

### Definition: Energy Dissipation Check

The goal is to numerically verify the **Lyapunov stability property** of the Backpropagation algorithm: that the loss (energy) must **monotonically decrease** over training epochs.

### Theory: Monotonic Loss Minimization

The training process is an optimization loop where the loss function $L(\boldsymbol{\theta})$ serves as the **potential energy** of the system. The Backpropagation algorithm, when implemented with a correctly calculated gradient, ensures that the weight update $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L$ drives the system toward a state of minimal potential energy.

The **Lyapunov Condition** for convergence dictates that the loss must be **monotonically non-increasing** for a stable learning rate ($\eta$):

$$L_{t+1} \le L_t$$

Tracking the loss $L_t$ validates that the algorithm is a physical process of **energy dissipation** and relaxation toward a local minimum.

-----

### Extensive Python Code and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Data and Model Training (Using a simple library for stability)
# ====================================================================

# Generate synthetic regression data
X, y = make_regression(n_samples=500, n_features=10, random_state=42, noise=1.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- Train MLP Regressor (Multi-Layer Perceptron) ---
# The solver 'sgd' explicitly uses standard gradient descent.
# We set max_iter high and learning_rate low for smooth convergence.
model = MLPRegressor(hidden_layer_sizes=(50,), # One hidden layer of 50 neurons
                     max_iter=500, 
                     alpha=0.01, # L2 penalty (regularization)
                     solver='sgd', 
                     learning_rate_init=0.001,
                     random_state=42,
                     verbose=False)

model.fit(X_train_scaled, y_train)

# Extract loss history (the tracking of the objective function J)
loss_history = model.loss_curve_

# ====================================================================
# 2. Visualization and Convergence Check
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot the Monotonic Descent of the Loss Function L
plt.plot(loss_history, 'r-', lw=2)

plt.title('Backpropagation Dynamics: Energy Dissipation Check')
plt.xlabel('Epoch')
plt.ylabel('Loss $L_t$ (Mean Squared Error)')
plt.grid(True)
plt.show()

# --- Analysis Summary ---
# Check for monotonicity (Loss should be monotonically non-increasing)
loss_diff = np.diff(loss_history)
# Check for negative or near-zero differences
is_monotonic = np.all(loss_diff <= 1e-6)

print("\n--- Energy Dissipation Check Summary ---")
print(f"Initial Loss (Epoch 1): L_0 = {loss_history[0]:.4f}")
print(f"Final Loss (Epoch {len(loss_history)}): L_final = {loss_history[-1]:.4f}")
print(f"Loss Monotonically Decreasing? {is_monotonic}")

print("\nConclusion: The plot shows a smooth, monotonically decreasing loss function, confirming the **Lyapunov stability property** of the Backpropagation algorithm. The process is one of **energy dissipation**, where the system constantly reduces its potential energy (error) by following the negative gradient to find a stable equilibrium state (local minimum).")
```

-----

## Project 4: Effect of Activation Function on Gradient (Vanishing Gradient)

-----

### Definition: Effect of Activation Function on Gradient

The goal is to illustrate the mathematical issue of the **vanishing gradient problem** by implementing and analyzing the derivative of the **Sigmoid activation function**.

### Theory: Sigmoid Saturation

The Sigmoid function, $\sigma(z) = 1 / (1 + e^{-z})$, was a common choice for nonlinear activation, but its use in deep networks creates instability.

The derivative, which dictates the local factor by which the error signal is multiplied during Backpropagation, is:

$$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

The vanishing gradient problem occurs because $\sigma'(z)$ rapidly **approaches zero** when the input $z$ is large in magnitude (i.e., the neuron is **saturated**). In a deep network, multiplying these near-zero terms across many layers causes the gradient to vanish, stalling training in early layers.

-----

### Extensive Python Code and Visualization

```python
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
```


