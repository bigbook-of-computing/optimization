# Source: Optimization/chapter-10/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Data and Model Training
# ====================================================================

# Generate linearly separable 2D data (2 features)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=42)

# Train the Logistic Regression Model
model = LogisticRegression(solver='lbfgs', random_state=42)
model.fit(X, y)

# Extract learned weights and bias (w1, w2, b)
w1, w2 = model.coef_[0]
b = model.intercept_[0]

# ====================================================================
# 2. Decision Boundary Calculation
# ====================================================================

# The decision boundary is defined by the line: w1*x1 + w2*x2 + b = 0
# We solve for x2 (the y-axis feature) to plot the line:
# x2 = (-w1*x1 - b) / w2

# Create a range for the x1 feature
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x1_plot = np.linspace(x1_min, x1_max, 100)

# Calculate the corresponding x2 values for the boundary
x2_boundary = (-w1 * x1_plot - b) / w2

# ====================================================================
# 3. Visualization
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot the raw data points, colored by their true class (y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='k', alpha=0.8, label='Data Points')

# Plot the linear decision boundary
plt.plot(x1_plot, x2_boundary, 'k--', lw=2, 
         label=f'Decision Boundary ($\mathbf{{w}}^T\mathbf{{x}} + b = 0$)')

# Labeling and Formatting
plt.title('Logistic Regression: Visualizing the Linear Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Decision Boundary Summary ---")
print(f"Learned Weights: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

print("\nConclusion: The visualization confirms that Logistic Regression, despite outputting a non-linear probability via the sigmoid function, is fundamentally a **linear classifier**. The decision boundary (where P=0.5) is a straight line, representing the optimal linear separation found through Maximum Likelihood estimation.")
