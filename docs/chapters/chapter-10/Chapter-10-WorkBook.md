## Chapter 10: Regression & Classification: The Linear Family (Workbook)

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

> **Summary:** The goal shifts from **inference** (characterizing $P(\boldsymbol{\theta})$) to **prediction** (calculating $\hat{y}$). The simplest model class is the **linear model** ($\hat{y} = \mathbf{w}^T \mathbf{x} + b$). **Regression** predicts continuous values, and **Classification** predicts discrete labels. Both are unified by the principle that minimizing loss is equivalent to performing **Maximum Likelihood (ML) inference** under specific noise assumptions.

#### Quiz Questions

**1. The primary goal of this chapter's modeling is to transition from the inference task of characterizing $P(\boldsymbol{\theta})$ to the practical task of:**

* **A.** Minimizing the partition function $Z$.
* **B.** **Predicting a future outcome $\hat{y}$ given a new input $\mathbf{x}$**. (**Correct**)
* **C.** Maximizing the model evidence $p(\mathcal{D})$.
* **D.** Calculating the total energy $E$.

**2. The single principle that conceptually unifies Linear Regression and Logistic Regression is that both methods seek to find the optimal weights $\mathbf{w}^*$ by minimizing a loss function that is equivalent to:**

* **A.** Maximizing the prior distribution.
* **B.** **Maximizing the likelihood of the observed data (ML Inference)**. (**Correct**)
* **C.** Minimizing the total number of features $D$.
* **D.** Maximizing the irreducible error $\sigma^2$.

---

#### Interview-Style Question

**Question:** Linear models are described as the "harmonic oscillators" of machine learning. Explain the physical analogy connecting the **Least Squares Objective** to the **harmonic oscillator potential energy**.

**Answer Strategy:** The least-squares objective $L(\mathbf{w}) = \sum (y_i - \hat{y}_i)^2$ minimizes the total squared error. This is mathematically identical to minimizing the potential energy in a system of linear springs. The error $(y_i - \hat{y}_i)$ represents the **displacement** of a spring, and the squared error represents the **elastic potential energy** ($E \propto x^2$) stored in that spring. Finding the best-fit line is simply finding the configuration of weights $\mathbf{w}^*$ that minimizes the total elastic potential energy stored across all data points.

---
***

### 10.2 Linear Regression as Maximum Likelihood

> **Summary:** Linear Regression is the **Maximum Likelihood Estimate (MLE)** under the assumption that the noise corrupting the data ($\epsilon_i$) is **independent and identically distributed (i.i.d.) Gaussian**. Maximizing the Gaussian log-likelihood is mathematically equivalent to minimizing the **Least-Squares Objective ($L(\mathbf{w})$)**. This is one of the few optimization problems with a **closed-form, analytic solution**.

#### Quiz Questions

**1. The assumption of **i.i.d. Gaussian noise** is necessary in linear regression to establish the probabilistic equivalence between maximizing the log-likelihood and minimizing the:**

* **A.** Total entropy.
* **B.** **Sum of squared errors (Least-Squares objective)**. (**Correct**)
* **C.** $L^1$ penalty term.
* **D.** Prior distribution.

**2. The term $\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$ represents:**

* **A.** The parameter vector for Logistic Regression.
* **B.** The covariance matrix.
* **C.** **The closed-form, analytic solution for the optimal linear regression weights**. (**Correct**)
* **D.** The gradient of the loss function.

---

#### Interview-Style Question

**Question:** The log-likelihood function for linear regression is $\ln P \propto \sum [-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(y_i - \hat{y}_i)^2}{2\sigma^2}]$. Explain why, when maximizing this function, the physicist only needs to consider the term proportional to the squared error, and can ignore the $\ln(2\pi\sigma^2)$ term.

**Answer Strategy:** The goal is to find $\arg\max \ln P$ with respect to the weights $\mathbf{w}$. The term $-\frac{1}{2} \ln(2\pi\sigma^2)$ is the **normalization constant** for the Gaussian noise; it depends on the variance $\sigma^2$ but **not** on the weights $\mathbf{w}$. Since it doesn't depend on the optimization variable, it is a constant offset and can be ignored. Maximization then reduces to maximizing the negative squared error term, which is equivalent to **minimizing the sum of squared errors**.

---
***

### 10.3 Geometry of Least Squares

> **Summary:** The least-squares solution has a direct **geometric interpretation** in vector space. The prediction vector $\hat{\mathbf{y}}$ must lie in the **column space of the design matrix $X$**. The optimal solution $\mathbf{w}^*$ is the one that makes $\hat{\mathbf{y}}$ the **orthogonal projection** of the target vector $\mathbf{y}$ onto this subspace. At equilibrium, the **residual error vector ($\mathbf{y} - \hat{\mathbf{y}}$)** is **orthogonal** to the feature vectors, meaning $X^T(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$.

#### Quiz Questions

**1. The least-squares solution $\mathbf{w}^*$ is found by projecting the target vector $\mathbf{y}$ onto the subspace spanned by the input features. This projection is fundamentally:**

* **A.** A geometric warping.
* **B.** **An orthogonal projection**. (**Correct**)
* **C.** A non-convex mapping.
* **D.** A least action trajectory.

**2. The condition $X^T(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$ confirms that the least-squares solution is at equilibrium because it proves that the residual error vector is:**

* **A.** Zero.
* **B.** Parallel to the prediction vector $\hat{\mathbf{y}}$.
* **C.** **Orthogonal (perpendicular) to the subspace spanned by the feature vectors**. (**Correct**)
* **D.** Proportional to the learning rate $\eta$.

---

#### Interview-Style Question

**Question:** In the geometric analogy (Section 10.3.4), the least-squares solution is analogous to finding a point of **equilibrium under orthogonal constraint forces**. Describe the relationship between the **residual error vector** and the **constraint subspace** at this equilibrium point.

**Answer Strategy:** The **constraint subspace** is the column space of $X$ (the space of all possible predictions $\hat{\mathbf{y}}$). At equilibrium, the **residual error vector** ($\mathbf{y} - \hat{\mathbf{y}}$) is the measure of the force exerted by the true target $\mathbf{y}$. Equilibrium is achieved when this error vector is **perfectly orthogonal** (perpendicular) to the constraint subspace. This means no further reduction in error is possible within the model's capabilities, ensuring that the gradient $\nabla L$ is zero and no residual **work** can be done by adjusting $\mathbf{w}$.

---
***

### 10.4 Regularization and the Bias–Variance Tradeoff

> **Summary:** **Regularization** is used to constrain model complexity and improve **generalization**. This is achieved by adding a penalty term to the least-squares objective, effectively converting the solution from an **MLE to a MAP estimate**. **Ridge Regression ($L^2$ penalty)** uses a Gaussian prior to apply an **elastic restraint** on weights, reducing **variance**. **LASSO ($L^1$ penalty)** uses a Laplace prior to promote **sparsity**. The total prediction error is decomposed into **Bias** (underfitting) and **Variance** (overfitting).

#### Quiz Questions

**1. The primary purpose of applying a regularization penalty (like $L^1$ or $L^2$) to the linear regression objective is to:**

* **A.** Find the closed-form solution.
* **B.** **Improve the model's ability to generalize to unseen data**. (**Correct**)
* **C.** Maximize the total energy.
* **D.** Force the model to use the Mahalanobis distance.

**2. In the Bias–Variance Tradeoff, a model with **high bias** is characterized by:**

* **A.** **Underfitting (the model is too rigid)**. (**Correct**)
* **B.** High flexibility (the model is too complex).
* **C.** Zero residual error.
* **D.** A single global minimum.

---

#### Interview-Style Question

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

### Project 2: Geometric Check of Least Squares Orthogonality

* **Goal:** Numerically verify the fundamental geometric property of least squares: that the residual error is orthogonal to the feature space.
* **Setup:** Use a simple dataset $X$ (2 features) and target $\mathbf{y}$. Fit the optimal weights $\mathbf{w}^*$ using the analytic solution.
* **Steps:**
    1.  Calculate the prediction vector $\hat{\mathbf{y}} = X\mathbf{w}^*$.
    2.  Calculate the residual error vector $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$.
    3.  Numerically compute the dot product between the feature matrix and the residual: $X^T \mathbf{r}$.
* ***Goal***: Show that the resulting vector $X^T \mathbf{r}$ is numerically near zero (e.g., $10^{-8}$), confirming the geometric condition for equilibrium.

### Project 3: Visualizing the Logistic Decision Boundary

* **Goal:** Implement and visualize the linear decision boundary found by Logistic Regression on a 2D classification problem.
* **Setup:** Generate synthetic 2D data that is linearly separable.
* **Steps:**
    1.  Fit the Logistic Regression classifier.
    2.  Calculate the probability $P(y=1|\mathbf{x})$ across a 2D grid using the sigmoid function output.
* ***Goal***: Plot the decision boundary (the contour where $P=0.5$). The resulting line must be a straight line that optimally separates the two classes, illustrating the linear nature of the solution.

### Project 4: Multiclass Softmax as an Energy Model

* **Goal:** Implement the Multiclass Softmax function and demonstrate its relation to the Boltzmann energy distribution.
* **Setup:** Define three classes ($K=3$). Define a set of arbitrary linear scores (logits): $z = [z_1, z_2, z_3]$.
* **Steps:**
    1.  Implement the Softmax function: $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$.
    2.  Calculate the probability vector $\mathbf{p} = [p_1, p_2, p_3]$ for two different score vectors: $\mathbf{z}_A = [1.0, 2.0, 3.0]$ and $\mathbf{z}_B = [3.0, 2.0, 1.0]$.
* ***Goal***: Show that for $\mathbf{z}_A$, the probability $p_3$ is highest, and for $\mathbf{z}_B$, $p_1$ is highest. Confirm that the sum of the probabilities is always 1, demonstrating that the function acts as a probabilistic **partition function** over the classes.


