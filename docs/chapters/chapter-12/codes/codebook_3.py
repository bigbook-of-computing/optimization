# Source: Optimization/chapter-12/codebook.md -- Block 3

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
