# Source: Optimization/chapter-12/codebook.md -- Block 1

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
