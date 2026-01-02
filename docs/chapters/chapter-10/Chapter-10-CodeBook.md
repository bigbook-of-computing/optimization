# 📈 Chapter 10: Regression & Classification: The Linear Family

## Project 1: Linear Regression as Maximum Likelihood (Closed-Form Solution)

-----

### Definition: Linear Regression as Maximum Likelihood

The goal is to implement the **analytical closed-form solution** for **Linear Regression** (Ordinary Least Squares, or OLS). This demonstrates that the minimum of the **Sum of Squared Errors (SSE)** can be found directly through matrix algebra.

### Theory: MLE, SSE, and the Closed Form

Linear Regression assumes a linear relationship between input features $\mathbf{x}$ and output $y$ with additive **Gaussian noise** ($\epsilon \sim \mathcal{N}(0, \sigma^2)$):

$$y = \mathbf{w}^T \mathbf{x} + b + \epsilon$$

Minimizing the **Sum of Squared Errors (SSE)** is mathematically equivalent to finding the **Maximum Likelihood Estimate (MLE)** for the weights $\mathbf{w}$. The analytical solution, which minimizes the quadratic loss surface in one step, is given by the **Normal Equation**:

$$\mathbf{w}^{*} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

Where $\mathbf{X}$ is the design matrix (including a column of ones for the bias/intercept $b$), $\mathbf{y}$ is the target vector, and $\mathbf{w}^*$ is the vector of optimal weights.

### Extensive Python Code and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Data Generation
# ====================================================================

N = 100  # Number of samples
TRUE_W = np.array([3.0, 5.0])  # True weights (w1, bias b)
NOISE_STD = 2.0

# Generate synthetic linear data: y = w1*x + b + noise
X_input = np.linspace(0, 10, N)
X_bias = np.ones(N)

# Design Matrix X: [X_input, X_bias]
X_design = np.vstack([X_input, X_bias]).T 

# Target Vector y: y = 3.0*x + 5.0 + noise
y = X_design @ TRUE_W + np.random.normal(0, NOISE_STD, N)

# ====================================================================
# 2. Closed-Form Solution (Normal Equation)
# ====================================================================

# Formula: w* = (X^T X)^{-1} X^T y
XTX = X_design.T @ X_design
XTy = X_design.T @ y

# Compute inverse and solve for w* (using np.linalg.solve for stability)
w_mle = np.linalg.solve(XTX, XTy)

# Separate weight and bias for interpretation
w1_mle, b_mle = w_mle

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Prediction line
y_pred = X_design @ w_mle
y_true_line = X_design @ TRUE_W

plt.figure(figsize=(9, 6))

# Plot raw data
plt.scatter(X_input, y, label='Simulated Data', alpha=0.6)

# Plot the true line
plt.plot(X_input, y_true_line, 'k--', label=f'True Model ($y={TRUE_W[0]:.2f}x + {TRUE_W[1]:.2f}$)')

# Plot the fitted line (MLE solution)
plt.plot(X_input, y_pred, 'r-', lw=2, label=f'MLE Fit ($y={w1_mle:.2f}x + {b_mle:.2f}$)')

# Labeling and Formatting
plt.title('Linear Regression: Closed-Form Maximum Likelihood Estimate (MLE)')
plt.xlabel('Input Feature $x$')
plt.ylabel('Target $y$')
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Linear Regression (OLS/MLE) Summary ---")
print(f"True Weights (w, b): {TRUE_W}")
print(f"MLE Estimates (w*, b*): {np.round(w_mle, 3)}")

print("\nConclusion: The closed-form solution successfully calculated the optimal weights (w*, b*) in a single matrix operation. The fitted line closely matches the true underlying function, demonstrating the equivalence between minimizing the Sum of Squared Errors (SSE) and finding the Maximum Likelihood Estimate under Gaussian noise assumptions.")
```

-----

## Project 2: Implementing Ridge Regression ($\boldsymbol{L_2}$ Regularization)

-----

### Definition: Implementing Ridge Regression

The goal is to implement the analytical closed-form solution for **Ridge Regression**. We will compare the weights learned with and without the $\boldsymbol{L_2}$ penalty to demonstrate how the **regularization term** shrinks model complexity.

### Theory: Regularization and the Gaussian Prior

Ridge Regression modifies the least-squares objective by adding a penalty proportional to the squared magnitude of the weights $\|\mathbf{w}\|_2^2$:

$$L_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2}\|\mathbf{X} \mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$$

The penalty term $\lambda \|\mathbf{w}\|_2^2$ serves to prevent **overfitting** by keeping the weights small.

The analytical solution for the optimal Ridge weights is:

$$\mathbf{w}^{*} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

In Bayesian inference, this is equivalent to finding the **Maximum A Posteriori (MAP) estimate** using a **Gaussian Prior** $P(\mathbf{w})$ centered at zero ($\mu=0$). The regularization strength $\lambda$ controls the tightness of this prior.

-----

### Extensive Python Code and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup High-Dimensional Data (Prone to Overfitting)
# ====================================================================

N = 50   # Samples
D = 10   # High dimensionality (features)
TRUE_W_SMALL = np.array([3.0] + [0.0]*(D-1)) # Only one feature is truly important
NOISE_STD = 1.0

# Generate random features
X_input = np.random.randn(N, D)
# Add bias column
X_design = np.hstack([X_input, np.ones((N, 1))])
W_TRUE = np.concatenate([TRUE_W_SMALL, [5.0]]) # [w1, w2...w10, b]

# Target Vector y: y = X @ W_TRUE + noise
y = X_design @ W_TRUE + np.random.normal(0, NOISE_STD, N)

# Define the identity matrix for the regularization term (D+1 dimensions)
I_reg = np.eye(D + 1)
I_reg[-1, -1] = 0 # Do not regularize the bias term (b)

# ====================================================================
# 2. Closed-Form Solutions (OLS vs. Ridge)
# ====================================================================

def solve_ridge(X_design, y, lambda_reg, I_reg):
    """
    Solves the Ridge/OLS problem analytically.
    w* = (X^T X + lambda*I)^{-1} X^T y
    """
    XTX = X_design.T @ X_design
    XTy = X_design.T @ y
    
    # Add regularization term to XTX
    reg_term = lambda_reg * I_reg
    w_star = np.linalg.solve(XTX + reg_term, XTy)
    return w_star

# --- Scenario A: OLS (No Regularization, lambda = 0) ---
W_OLS = solve_ridge(X_design, y, lambda_reg=0.0, I_reg=I_reg)

# --- Scenario B: Ridge (Strong Regularization, lambda = 10) ---
W_RIDGE = solve_ridge(X_design, y, lambda_reg=10.0, I_reg=I_reg)

# ====================================================================
# 3. Visualization and Analysis
# ====================================================================

# Prepare data for plotting weights (excluding bias term at index D)
w_names = [f'$w_{i+1}$' for i in range(D)]
w_true_val = W_TRUE[:D]
w_ols_val = W_OLS[:D]
w_ridge_val = W_RIDGE[:D]

df_weights = pd.DataFrame({
    'True': w_true_val,
    'OLS (\u03bb=0)': w_ols_val,
    'Ridge (\u03bb=10)': w_ridge_val
}, index=w_names)

print("--- Weight Comparison: OLS vs. Ridge Regression ---")
print(df_weights.to_markdown(floatfmt=".3f"))

# Plot 1: Weight Magnitudes
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(D)
width = 0.25

ax.bar(x - width, w_true_val, width, label='True Weight', color='gray')
ax.bar(x, w_ols_val, width, label='OLS (Unregularized)', color='skyblue')
ax.bar(x + width, w_ridge_val, width, label='Ridge (Regularized, $\\lambda=10$)', color='darkred')

# Labeling and Formatting
ax.set_title('Regularization Shrinks Weight Magnitude')
ax.set_xlabel('Feature Weight')
ax.set_ylabel('Weight Magnitude')
ax.set_xticks(x)
ax.set_xticklabels(w_names)
ax.axhline(0, color='k', lw=0.8)
ax.legend()
plt.tight_layout()
plt.show()

print("\nConclusion: The OLS weights for irrelevant features (w2-w10) are large and noisy. Ridge Regression successfully applies the L2 penalty, shrinking the magnitude of these irrelevant weights toward zero. This demonstrates that regularization is computationally equivalent to introducing a strong **Gaussian Prior** that favors simpler models.")
```

-----

## Project 3: Logistic Regression: Visualizing the Decision Boundary

-----

### Definition: Logistic Regression: Visualizing the Decision Boundary

The goal is to implement and visualize the **linear decision boundary** of a **Logistic Regression** classifier. This visually confirms that although the output is a probability (nonlinear), the underlying boundary separating the two classes is strictly linear.

### Theory: Decision Boundary and Sigmoid Function

Logistic Regression uses a linear combination of features, $\mathbf{z} = \mathbf{w}^T \mathbf{x} + b$, and transforms it into a probability $P(y=1|\mathbf{x})$ using the **Sigmoid function** ($\sigma(\mathbf{z})$):

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{z}) = \frac{1}{1 + e^{-\mathbf{z}}}$$

The **decision boundary** is the contour where the probability of belonging to either class is equal: $P(y=1|\mathbf{x}) = 0.5$. Since $\sigma(\mathbf{z})=0.5$ only when $\mathbf{z}=0$, the boundary is defined by the linear condition:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

This is a straight line (or hyperplane in higher dimensions), illustrating that Logistic Regression is fundamentally a **linear classifier**.

-----

### Extensive Python Code and Visualization

```python
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
```

-----

## Project 4: Multiclass Softmax as an Energy Model

-----

### Definition: Multiclass Softmax as an Energy Model

The goal is to implement the **Multiclass Softmax function** and demonstrate its relationship to the **Boltzmann distribution** in statistical physics.

### Theory: Boltzmann Distribution and Logits

The Softmax function transforms arbitrary real-valued scores (called **logits**, $\mathbf{z}$) into a discrete probability distribution ($\mathbf{p}$) over $K$ classes:

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

The core analogy is that the Softmax function is the continuous, differentiable equivalent of the **Boltzmann distribution**:

$$P(s) \propto e^{-E(s)/k_B T}$$

In this view, the **logit $z_k$** corresponds to the **negative energy** of state $k$ ($\mathbf{z}_k \propto -E_k$), and the summation in the denominator is analogous to the **partition function** ($\mathbf{Z}$). Softmax ensures that the highest-scoring state receives the largest probability, and the sum of all probabilities is always 1, reflecting the fundamental constraint of a physical probability distribution.

-----

### Extensive Python Code

```python
import numpy as np

# ====================================================================
# 1. Softmax Implementation
# ====================================================================

def softmax(z):
    """
    Implements the numerically stable Multiclass Softmax function.
    p_k = exp(z_k) / sum(exp(z_j))
    """
    # Numerical stability trick: subtract max(z) from all logits before exponentiating
    z_max = np.max(z)
    e_z = np.exp(z - z_max)
    
    # Partition function Z is the sum of exponentiated logits
    Z = np.sum(e_z) 
    
    # Probability p_k
    p = e_z / Z
    return p, Z

# ====================================================================
# 2. Scenario Testing
# ====================================================================

# Define three classes (K=3)
CLASSES = ['Class 1', 'Class 2', 'Class 3']

# --- Scenario A: Strong bias toward Class 3 ---
Z_A = np.array([1.0, 2.0, 3.0])
P_A, Z_A_part = softmax(Z_A)

# --- Scenario B: Strong bias toward Class 1 ---
Z_B = np.array([3.0, 2.0, 1.0])
P_B, Z_B_part = softmax(Z_B)

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- Multiclass Softmax as Boltzmann Distribution ---")
print(f"Logits are analogous to Negative Energy: z_k \u223d -E_k")

print("\n1. Scenario A: Logits Z_A = [1.0, 2.0, 3.0] (Bias to Class 3)")
print(f"   Partition Function (Z): {Z_A_part:.3f}")
print(f"   Probabilities (P): {np.round(P_A, 3)}")
print(f"   Sum(P): {np.sum(P_A):.0f} (Verifies constraint)")

print("\n2. Scenario B: Logits Z_B = [3.0, 2.0, 1.0] (Bias to Class 1)")
print(f"   Partition Function (Z): {Z_B_part:.3f}")
print(f"   Probabilities (P): {np.round(P_B, 3)}")
print(f"   Sum(P): {np.sum(P_B):.0f} (Verifies constraint)")

print("\nConclusion: The Softmax function successfully transforms linear scores (negative energies) into a valid probability distribution. The largest logit (highest negative energy) receives the highest probability. The denominator acts as the **Partition Function (Z)**, normalizing the distribution and completing the direct computational analogy to the physical Boltzmann distribution.")
```


