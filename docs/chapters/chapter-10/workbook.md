# **Chapter 10: Regression & Classification: The Linear Family (Workbook)**

The goal of this chapter is to apply the principles of **Maximum Likelihood (ML) inference** to the fundamental predictive models, establishing the direct equivalence between **energy minimization** (least squares) and **probabilistic prediction**.

| Section | Topic Summary |
| :--- | :--- |
| **10.1** | From Inference to Prediction |
| **10.2** | Linear Regression as Maximum Likelihood |
| **10.3** | Geometry of Least Squares |
| **10.4** | Regularization and the Bias–Variance Tradeoff |
| **10.5** | Bayesian Linear Regression |
| **10.6** | Logistic Regression — From Lines to Boundaries |
| **10.7** | Linear Discriminant Analysis (LDA) |
| **10.8** | Multiclass and Multivariate Extensions |
| **10.9–10.12** | Worked Example, Code Demo, and Takeaways |

---

### 10.1 From Inference to Prediction

> **Summary:** The goal shifts from **inference** (characterizing $P(\mathcal{\theta})$) to **prediction** (calculating $\hat{y}$). The simplest model class is the **linear model** ($\hat{y} = \mathbf{w}^T \mathbf{x} + b$). **Regression** predicts continuous values, and **Classification** predicts discrete labels. Both are unified by the principle that minimizing loss is equivalent to performing **Maximum Likelihood (ML) inference** under specific noise assumptions.

#### Quiz Questions

!!! note "Quiz"
    **1. The primary goal of this chapter's modeling is to transition from the inference task of characterizing $P(\mathcal{\theta})$ to the practical task of:**
    
    * **A.** Minimizing the partition function $Z$.
    * **B.** **Predicting a future outcome $\hat{y}$ given a new input $\mathbf{x}$**. (**Correct**)
    * **C.** Maximizing the model evidence $p(\mathcal{D})$.
    * **D.** Calculating the total energy $E$.
    
!!! note "Quiz"
    **2. The single principle that conceptually unifies Linear Regression and Logistic Regression is that both methods seek to find the optimal weights $\mathbf{w}^*$ by minimizing a loss function that is equivalent to:**
    
    * **A.** Maximizing the prior distribution.
    * **B.** **Maximizing the likelihood of the observed data (ML Inference)**. (**Correct**)
    * **C.** Minimizing the total number of features $D$.
    * **D.** Maximizing the irreducible error $\sigma^2$.
    
---

!!! question "Interview Practice"
    **Question:** Linear models are described as the "harmonic oscillators" of machine learning. Explain the physical analogy connecting the **Least Squares Objective** to the **harmonic oscillator potential energy**.
    
    **Answer Strategy:** The least-squares objective $L(\mathbf{w}) = \sum (y_i - \hat{y}_i)^2$ minimizes the total squared error. This is mathematically identical to minimizing the potential energy in a system of linear springs. The error $(y_i - \hat{y}_i)$ represents the **displacement** of a spring, and the squared error represents the **elastic potential energy** ($E \propto x^2$) stored in that spring. Finding the best-fit line is simply finding the configuration of weights $\mathbf{w}^*$ that minimizes the total elastic potential energy stored across all data points.
    
---

---

### 10.2 Linear Regression as Maximum Likelihood

> **Summary:** Linear Regression is the **Maximum Likelihood Estimate (MLE)** under the assumption that the noise corrupting the data ($\epsilon_i$) is **independent and identically distributed (i.i.d.) Gaussian**. Maximizing the Gaussian log-likelihood is mathematically equivalent to minimizing the **Least-Squares Objective ($L(\mathbf{w})$)**. This is one of the few optimization problems with a **closed-form, analytic solution**.

#### Quiz Questions

!!! note "Quiz"
    **1. The assumption of **i.i.d. Gaussian noise** is necessary in linear regression to establish the probabilistic equivalence between maximizing the log-likelihood and minimizing the:**
    
    * **A.** Total entropy.
    * **B.** **Sum of squared errors (Least-Squares objective)**. (**Correct**)
    * **C.** $L^1$ penalty term.
    * **D.** Prior distribution.
    
!!! note "Quiz"
    **2. The term $\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$ represents:**
    
    * **A.** The parameter vector for Logistic Regression.
    * **B.** The covariance matrix.
    * **C.** **The closed-form, analytic solution for the optimal linear regression weights**. (**Correct**)
    * **D.** The gradient of the loss function.
    
---

!!! question "Interview Practice"
    **Question:** The log-likelihood function for linear regression is $\ln P \propto \sum [-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(y_i - \hat{y}_i)^2}{2\sigma^2}]$. Explain why, when maximizing this function, the physicist only needs to consider the term proportional to the squared error, and can ignore the $\ln(2\pi\sigma^2)$ term.
    
    **Answer Strategy:** The goal is to find $\arg\max \ln P$ with respect to the weights $\mathbf{w}$. The term $-\frac{1}{2} \ln(2\pi\sigma^2)$ is the **normalization constant** for the Gaussian noise; it depends on the variance $\sigma^2$ but **not** on the weights $\mathbf{w}$. Since it doesn't depend on the optimization variable, it is a constant offset and can be ignored. Maximization then reduces to maximizing the negative squared error term, which is equivalent to **minimizing the sum of squared errors**.
    
---

---

### 10.3 Geometry of Least Squares

> **Summary:** The least-squares solution has a direct **geometric interpretation** in vector space. The prediction vector $\hat{\mathbf{y}}$ must lie in the **column space of the design matrix $X$**. The optimal solution $\mathbf{w}^*$ is the one that makes $\hat{\mathbf{y}}$ the **orthogonal projection** of the target vector $\mathbf{y}$ onto this subspace. At equilibrium, the **residual error vector ($\mathbf{y} - \hat{\mathbf{y}}$)** is **orthogonal** to the feature vectors, meaning $X^T(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$.

#### Quiz Questions

!!! note "Quiz"
    **1. The least-squares solution $\mathbf{w}^*$ is found by projecting the target vector $\mathbf{y}$ onto the subspace spanned by the input features. This projection is fundamentally:**
    
    * **A.** A geometric warping.
    * **B.** **An orthogonal projection**. (**Correct**)
    * **C.** A non-convex mapping.
    * **D.** A least action trajectory.
    
!!! note "Quiz"
    **2. The condition $X^T(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$ confirms that the least-squares solution is at equilibrium because it proves that the residual error vector is:**
    
    * **A.** Zero.
    * **B.** Parallel to the prediction vector $\hat{\mathbf{y}}$.
    * **C.** **Orthogonal (perpendicular) to the subspace spanned by the feature vectors**. (**Correct**)
    * **D.** Proportional to the learning rate $\eta$.
    
---

!!! question "Interview Practice"
    **Question:** In the geometric analogy (Section 10.3.4), the least-squares solution is analogous to finding a point of **equilibrium under orthogonal constraint forces**. Describe the relationship between the **residual error vector** and the **constraint subspace** at this equilibrium point.
    
    **Answer Strategy:** The **constraint subspace** is the column space of $X$ (the space of all possible predictions $\hat{\mathbf{y}}$). At equilibrium, the **residual error vector** ($\mathbf{y} - \hat{\mathbf{y}}$) is the measure of the force exerted by the true target $\mathbf{y}$. Equilibrium is achieved when this error vector is **perfectly orthogonal** (perpendicular) to the constraint subspace. This means no further reduction in error is possible within the model's capabilities, ensuring that the gradient $\nabla L$ is zero and no residual **work** can be done by adjusting $\mathbf{w}$.
    
---

---

### 10.4 Regularization and the Bias–Variance Tradeoff

> **Summary:** **Regularization** is used to constrain model complexity and improve **generalization**. This is achieved by adding a penalty term to the least-squares objective, effectively converting the solution from an **MLE to a MAP estimate**. **Ridge Regression ($L^2$ penalty)** uses a Gaussian prior to apply an **elastic restraint** on weights, reducing **variance**. **LASSO ($L^1$ penalty)** uses a Laplace prior to promote **sparsity**. The total prediction error is decomposed into **Bias** (underfitting) and **Variance** (overfitting).

#### Quiz Questions

!!! note "Quiz"
    **1. The primary purpose of applying a regularization penalty (like $L^1$ or $L^2$) to the linear regression objective is to:**
    
    * **A.** Find the closed-form solution.
    * **B.** **Improve the model's ability to generalize to unseen data**. (**Correct**)
    * **C.** Maximize the total energy.
    * **D.** Force the model to use the Mahalanobis distance.
    
!!! note "Quiz"
    **2. In the Bias–Variance Tradeoff, a model with **high bias** is characterized by:**
    
    * **A.** **Underfitting (the model is too rigid)**. (**Correct**)
    * **B.** High flexibility (the model is too complex).
    * **C.** Zero residual error.
    * **D.** A single global minimum.
    
---

!!! question "Interview Practice"
    **Question:** Compare and contrast the **$L^2$ penalty (Ridge Regression)** with the **$L^1$ penalty (LASSO)** in terms of their effect on the final weight vector $\mathbf{w}^*$.
    
    **Answer Strategy:**
    * **$L^2$ (Ridge):** Adds an energetic penalty proportional to $\sum w_i^2$. It applies an **elastic restraint** that discourages large weight magnitudes, shrinking all weights towards zero evenly. It reduces variance but keeps all features in the model.
    * **$L^1$ (LASSO):** Adds a penalty proportional to $\sum |w_i|$. It encourages some weights to become **exactly zero**, thus promoting **sparsity** and performing intrinsic feature selection. This behavior is due to its geometric connection to the Laplace prior.
    
---

### 💡 Hands-On Project Ideas 🛠️

These projects require computational techniques to implement and analyze the core concepts of linear model inference and geometry.

### Project 1: Implementing and Testing Ridge Regression ($L^2$)

* **Goal:** Implement the Ridge Regression objective and observe the **shrinkage** of weights as the regularization strength ($\lambda$) increases.
* **Setup:** Generate synthetic data $y = 3x_1 + 0.1x_2 + 5 + \epsilon$ where $x_2$ is a high-variance, low-importance noise feature.
* **Steps:**
    1.  Fit a standard Linear Regression model ($\lambda=0$) and record the weights $\mathbf{w}_0$.
    2.  Fit a Ridge Regression model ($\lambda=10$) and record the weights $\mathbf{w}_{10}$.
* ***Goal***: Show that $w_{10}$ is smaller in magnitude than $w_0$, and that the penalty disproportionately shrinks the weights of noisy features (like $x_2$) toward zero, illustrating the regularization effect.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Data Generation

## ====================================================================

N = 100  # Number of samples
TRUE_W = np.array([3.0, 5.0])  # True weights (w1, bias b)
NOISE_STD = 2.0

## Generate synthetic linear data: y = w1*x + b + noise

X_input = np.linspace(0, 10, N)
X_bias = np.ones(N)

## Design Matrix X: [X_input, X_bias]

X_design = np.vstack([X_input, X_bias]).T

## Target Vector y: y = 3.0*x + 5.0 + noise

y = X_design @ TRUE_W + np.random.normal(0, NOISE_STD, N)

## ====================================================================

## 2. Closed-Form Solution (Normal Equation)

## ====================================================================

## Formula: w* = (X^T X)^{-1} X^T y

XTX = X_design.T @ X_design
XTy = X_design.T @ y

## Compute inverse and solve for w* (using np.linalg.solve for stability)

w_mle = np.linalg.solve(XTX, XTy)

## Separate weight and bias for interpretation

w1_mle, b_mle = w_mle

## ====================================================================

## 3. Visualization and Analysis

## ====================================================================

## Prediction line

y_pred = X_design @ w_mle
y_true_line = X_design @ TRUE_W

plt.figure(figsize=(9, 6))

## Plot raw data

plt.scatter(X_input, y, label='Simulated Data', alpha=0.6)

## Plot the true line

plt.plot(X_input, y_true_line, 'k--', label=f'True Model ($y={TRUE_W[0]:.2f}x + {TRUE_W[1]:.2f}$)')

## Plot the fitted line (MLE solution)

plt.plot(X_input, y_pred, 'r-', lw=2, label=f'MLE Fit ($y={w1_mle:.2f}x + {b_mle:.2f}$)')

## Labeling and Formatting

plt.title('Linear Regression: Closed-Form Maximum Likelihood Estimate (MLE)')
plt.xlabel('Input Feature $x$')
plt.ylabel('Target $y$')
plt.legend()
plt.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- Linear Regression (OLS/MLE) Summary ---")
print(f"True Weights (w, b): {TRUE_W}")
print(f"MLE Estimates (w*, b*): {np.round(w_mle, 3)}")

print("\nConclusion: The closed-form solution successfully calculated the optimal weights (w*, b*) in a single matrix operation. The fitted line closely matches the true underlying function, demonstrating the equivalence between minimizing the Sum of Squared Errors (SSE) and finding the Maximum Likelihood Estimate under Gaussian noise assumptions.")
```
**Sample Output:**
```python
--- Linear Regression (OLS/MLE) Summary ---
True Weights (w, b): [3. 5.]
MLE Estimates (w*, b*): [3.028 4.654]

Conclusion: The closed-form solution successfully calculated the optimal weights (w*, b*) in a single matrix operation. The fitted line closely matches the true underlying function, demonstrating the equivalence between minimizing the Sum of Squared Errors (SSE) and finding the Maximum Likelihood Estimate under Gaussian noise assumptions.
```


### Project 2: Geometric Check of Least Squares Orthogonality

* **Goal:** Numerically verify the fundamental geometric property of least squares: that the residual error is orthogonal to the feature space.
* **Setup:** Use a simple dataset $X$ (2 features) and target $\mathbf{y}$. Fit the optimal weights $\mathbf{w}^*$ using the analytic solution.
* **Steps:**
    1.  Calculate the prediction vector $\hat{\mathbf{y}} = X\mathbf{w}^*$.
    2.  Calculate the residual error vector $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$.
    3.  Numerically compute the dot product between the feature matrix and the residual: $X^T \mathbf{r}$.
* ***Goal***: Show that the resulting vector $X^T \mathbf{r}$ is numerically near zero (e.g., $10^{-8}$), confirming the geometric condition for equilibrium.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup High-Dimensional Data (Prone to Overfitting)

## ====================================================================

N = 50   # Samples
D = 10   # High dimensionality (features)
TRUE_W_SMALL = np.array([3.0] + [0.0]*(D-1)) # Only one feature is truly important
NOISE_STD = 1.0

## Generate random features

X_input = np.random.randn(N, D)
## Add bias column

X_design = np.hstack([X_input, np.ones((N, 1))])
W_TRUE = np.concatenate([TRUE_W_SMALL, [5.0]]) # [w1, w2...w10, b]

## Target Vector y: y = X @ W_TRUE + noise

y = X_design @ W_TRUE + np.random.normal(0, NOISE_STD, N)

## Define the identity matrix for the regularization term (D+1 dimensions)

I_reg = np.eye(D + 1)
I_reg[-1, -1] = 0 # Do not regularize the bias term (b)

## ====================================================================

## 2. Closed-Form Solutions (OLS vs. Ridge)

## ====================================================================

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

## --- Scenario A: OLS (No Regularization, lambda = 0) ---

W_OLS = solve_ridge(X_design, y, lambda_reg=0.0, I_reg=I_reg)

## --- Scenario B: Ridge (Strong Regularization, lambda = 10) ---

W_RIDGE = solve_ridge(X_design, y, lambda_reg=10.0, I_reg=I_reg)

## ====================================================================

## 3. Visualization and Analysis

## ====================================================================

## Prepare data for plotting weights (excluding bias term at index D)

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

## Plot 1: Weight Magnitudes

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(D)
width = 0.25

ax.bar(x - width, w_true_val, width, label='True Weight', color='gray')
ax.bar(x, w_ols_val, width, label='OLS (Unregularized)', color='skyblue')
ax.bar(x + width, w_ridge_val, width, label='Ridge (Regularized, $\\lambda=10$)', color='darkred')

## Labeling and Formatting

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
**Sample Output:**
```python
--- Weight Comparison: OLS vs. Ridge Regression ---
|        |   True |   OLS (λ=0) |   Ridge (λ=10) |
|:-------|-------:|------------:|---------------:|
| $w_1$  |  3.000 |       2.820 |          2.274 |
| $w_2$  |  0.000 |      -0.301 |         -0.211 |
| $w_3$  |  0.000 |       0.162 |          0.157 |
| $w_4$  |  0.000 |       0.092 |         -0.002 |
| $w_5$  |  0.000 |       0.121 |          0.071 |
| $w_6$  |  0.000 |      -0.149 |         -0.057 |
| $w_7$  |  0.000 |      -0.099 |         -0.055 |
| $w_8$  |  0.000 |       0.245 |          0.226 |
| $w_9$  |  0.000 |      -0.293 |         -0.217 |
| $w_10$ |  0.000 |      -0.255 |         -0.255 |

Conclusion: The OLS weights for irrelevant features (w2-w10) are large and noisy. Ridge Regression successfully applies the L2 penalty, shrinking the magnitude of these irrelevant weights toward zero. This demonstrates that regularization is computationally equivalent to introducing a strong **Gaussian Prior** that favors simpler models.
```


### Project 3: Visualizing the Logistic Decision Boundary

* **Goal:** Implement and visualize the linear decision boundary found by Logistic Regression on a 2D classification problem.
* **Setup:** Generate synthetic 2D data that is linearly separable.
* **Steps:**
    1.  Fit the Logistic Regression classifier.
    2.  Calculate the probability $P(y=1|\mathbf{x})$ across a 2D grid using the sigmoid function output.
* ***Goal***: Plot the decision boundary (the contour where $P=0.5$). The resulting line must be a straight line that optimally separates the two classes, illustrating the linear nature of the solution.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Data and Model Training

## ====================================================================

## Generate linearly separable 2D data (2 features)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=42)

## Train the Logistic Regression Model

model = LogisticRegression(solver='lbfgs', random_state=42)
model.fit(X, y)

## Extract learned weights and bias (w1, w2, b)

w1, w2 = model.coef_[0]
b = model.intercept_[0]

## ====================================================================

## 2. Decision Boundary Calculation

## ====================================================================

## The decision boundary is defined by the line: w1*x1 + w2*x2 + b = 0

## We solve for x2 (the y-axis feature) to plot the line:

## x2 = (-w1*x1 - b) / w2

## Create a range for the x1 feature

x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x1_plot = np.linspace(x1_min, x1_max, 100)

## Calculate the corresponding x2 values for the boundary

x2_boundary = (-w1 * x1_plot - b) / w2

## ====================================================================

## 3. Visualization

## ====================================================================

plt.figure(figsize=(9, 6))

## Plot the raw data points, colored by their true class (y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='k', alpha=0.8, label='Data Points')

## Plot the linear decision boundary

plt.plot(x1_plot, x2_boundary, 'k--', lw=2,
         label=f'Decision Boundary ($\mathbf{{w}}^T\mathbf{{x}} + b = 0$)')

## Labeling and Formatting

plt.title('Logistic Regression: Visualizing the Linear Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- Decision Boundary Summary ---")
print(f"Learned Weights: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

print("\nConclusion: The visualization confirms that Logistic Regression, despite outputting a non-linear probability via the sigmoid function, is fundamentally a **linear classifier**. The decision boundary (where P=0.5) is a straight line, representing the optimal linear separation found through Maximum Likelihood estimation.")
```
**Sample Output:**
```python
--- Decision Boundary Summary ---
Learned Weights: w1=-1.89, w2=3.13, b=2.05

Conclusion: The visualization confirms that Logistic Regression, despite outputting a non-linear probability via the sigmoid function, is fundamentally a **linear classifier**. The decision boundary (where P=0.5) is a straight line, representing the optimal linear separation found through Maximum Likelihood estimation.
```


### Project 4: Multiclass Softmax as an Energy Model

* **Goal:** Implement the Multiclass Softmax function and demonstrate its relation to the Boltzmann energy distribution.
* **Setup:** Define three classes ($K=3$). Define a set of arbitrary linear scores (logits): $z = [z_1, z_2, z_3]$.
* **Steps:**
    1.  Implement the Softmax function: $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$.
    2.  Calculate the probability vector $\mathbf{p} = [p_1, p_2, p_3]$ for two different score vectors: $\mathbf{z}_A = [1.0, 2.0, 3.0]$ and $\mathbf{z}_B = [3.0, 2.0, 1.0]$.
* ***Goal***: Show that for $\mathbf{z}_A$, the probability $p_3$ is highest, and for $\mathbf{z}_B$, $p_1$ is highest. Confirm that the sum of the probabilities is always 1, demonstrating that the function acts as a probabilistic **partition function** over the classes.

#### Python Implementation

```python
import numpy as np

## ====================================================================

## 1. Softmax Implementation

## ====================================================================

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

## ====================================================================

## 2. Scenario Testing

## ====================================================================

## Define three classes (K=3)

CLASSES = ['Class 1', 'Class 2', 'Class 3']

## --- Scenario A: Strong bias toward Class 3 ---

Z_A = np.array([1.0, 2.0, 3.0])
P_A, Z_A_part = softmax(Z_A)

## --- Scenario B: Strong bias toward Class 1 ---

Z_B = np.array([3.0, 2.0, 1.0])
P_B, Z_B_part = softmax(Z_B)

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

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
**Sample Output:**
```python
--- Multiclass Softmax as Boltzmann Distribution ---
Logits are analogous to Negative Energy: z_k ∽ -E_k

1. Scenario A: Logits Z_A = [1.0, 2.0, 3.0] (Bias to Class 3)
   Partition Function (Z): 1.503
   Probabilities (P): [0.09  0.245 0.665]
   Sum(P): 1 (Verifies constraint)

2. Scenario B: Logits Z_B = [3.0, 2.0, 1.0] (Bias to Class 1)
   Partition Function (Z): 1.503
   Probabilities (P): [0.665 0.245 0.09 ]
   Sum(P): 1 (Verifies constraint)

Conclusion: The Softmax function successfully transforms linear scores (negative energies) into a valid probability distribution. The largest logit (highest negative energy) receives the highest probability. The denominator acts as the **Partition Function (Z)**, normalizing the distribution and completing the direct computational analogy to the physical Boltzmann distribution.
```