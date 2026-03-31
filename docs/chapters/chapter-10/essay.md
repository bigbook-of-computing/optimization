# **Chapter 10: Regression & Classification — The Linear Family**

# **Introduction**

The Bayesian inference framework of Chapter 9 established probability as the language of uncertainty, showing how rational agents update beliefs from priors to posteriors through the likelihood of observed data. But inference alone does not make predictions—it characterizes what we know about parameters $\mathbf{\theta}$, not what we can forecast about new observations $y_*$. This chapter bridges inference and prediction by applying Maximum Likelihood (ML) and Maximum A Posteriori (MAP) principles to the **linear family** of models—the simplest, most fundamental architectures for regression (continuous outputs) and classification (discrete labels). These models serve as the "harmonic oscillators" of machine learning: analytically tractable, geometrically interpretable, and foundational to all subsequent complexity.

We begin by formalizing the shift from inference (characterizing $p(\mathbf{\theta}|\mathcal{D})$) to prediction (computing $\hat{y} = f_{\mathbf{\theta}}(\mathbf{x})$), revealing that **Linear Regression** is simply Maximum Likelihood estimation under the assumption of i.i.d. Gaussian noise—minimizing squared error is equivalent to maximizing the log-likelihood of a Gaussian model. We explore the **geometry of least squares**, showing that the optimal weight vector $\mathbf{w}^*$ performs an orthogonal projection of target vectors onto the column space of feature matrices, with residual errors perpendicular to the prediction subspace (zero gradient at equilibrium). Regularization (Ridge, LASSO) emerges naturally as the application of Bayesian priors: L2 penalties correspond to Gaussian priors (elastic restraint toward zero), L1 penalties to Laplace priors (sparsity-inducing), and the choice of regularization strength $\lambda$ controls the **bias-variance tradeoff** (underfitting vs overfitting). **Bayesian Linear Regression** extends the framework to full posterior distributions over weights $p(\mathbf{w}|\mathcal{D})$, yielding posterior predictive distributions $p(y_*|\mathbf{x}_*, \mathcal{D})$ that ensemble-average predictions over parameter uncertainty—mirroring the thermal averaging of statistical mechanics. For classification, **Logistic Regression** applies the sigmoid function $\sigma(\mathbf{w}^T \mathbf{x})$ to map linear scores to probabilities, with cross-entropy loss arising from Maximum Likelihood under Bernoulli noise; the resulting decision boundary ($\mathbf{w}^T \mathbf{x} = 0$) is the linear hyperplane separating classes at equilibrium under competing "forces" from misclassified points. **Linear Discriminant Analysis (LDA)** takes a generative approach, modeling class-conditional Gaussians with shared covariance, yielding linear boundaries through Bayes' Theorem. Extensions to multiclass problems (Softmax) and multivariate outputs complete the architecture, with Softmax directly generalizing the Boltzmann distribution (partition function normalization) and providing the standard output layer for deep networks.

By the end of this chapter, you will understand linear models as the practical implementation of Bayesian and optimization principles: regression minimizes elastic potential energy (squared residuals as springs), classification finds equilibrium decision boundaries (zero net force), and regularization controls complexity through energetic penalties (priors as constraints). You will see that the bias-variance tradeoff is a statistical manifestation of the stiffness-fluctuation duality in physics, and that Bayesian posteriors provide ensemble predictions analogous to thermodynamic averaging over microstates. Worked examples demonstrate phase classification in the Ising model (logistic boundary as critical line) and code implementations visualize least-squares projection and sigmoid decision surfaces. These foundations prepare us for Chapter 11, where we extend inference from independent linear models to **Probabilistic Graphical Models**—networks encoding complex conditional dependencies through graph topology, transforming single-model inference into distributed message passing across interconnected systems.

---

# **Chapter 10: Outline**

| **Sec.** | **Title** | **Core Ideas & Examples** |
|:---|:---|:---|
| **10.1** | From Inference to Prediction | Shift from characterizing $p(\mathbf{\theta}|\mathcal{D})$ (inference) to computing $\hat{y} = f_{\mathbf{\theta}}(\mathbf{x})$ (prediction); linear model: $\hat{y} = \mathbf{w}^T \mathbf{x} + b$; regression (continuous $y \in \mathbb{R}$) vs classification (discrete $y \in \{0,1\}$); unifying viewpoint: both are ML estimation under specific noise assumptions; Example: linear models as harmonic oscillators of learning |
| **10.2** | Linear Regression as Maximum Likelihood | Probabilistic model: $y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ (i.i.d. Gaussian noise); likelihood: $p(\mathcal{D}|\mathbf{w}, \sigma^2) = \prod_i \mathcal{N}(y_i | \mathbf{w}^T \mathbf{x}_i, \sigma^2)$; ML estimate: maximize log-likelihood $\iff$ minimize squared error $\sum_i (y_i - \mathbf{w}^T \mathbf{x}_i)^2$; closed-form solution: $\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$; physical analogy: elastic potential energy (springs); Example: least squares as ML under Gaussian assumption |
| **10.3** | Geometry of Least Squares | Vector-space formulation: $\mathbf{y} \approx X\mathbf{w}$; orthogonal projection: $\hat{\mathbf{y}} = X(X^T X)^{-1}X^T \mathbf{y}$ (closest vector in column space of $X$); residual orthogonality: $X^T(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$ (zero gradient condition); projection matrix $P = X(X^T X)^{-1}X^T$; physical analogy: equilibrium under constraint forces, residual perpendicular to subspace; Example: $R^2$ as cosine of angle between $\mathbf{y}$ and $\hat{\mathbf{y}}$ |
| **10.4** | Regularization and the Bias–Variance Tradeoff | Regularization as Bayesian prior: Ridge ($L^2$ penalty, Gaussian prior), LASSO ($L^1$ penalty, Laplace prior); MAP estimation: $\mathbf{w}_{\text{MAP}} = \arg\min [\|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2]$; bias-variance decomposition: $\mathbb{E}[(\hat{y}-y)^2] = (\text{Bias})^2 + \text{Variance} + \sigma^2$; trade-off: high $\lambda$ (rigid, high bias, underfitting) vs low $\lambda$ (flexible, high variance, overfitting); physical analogy: stiffness vs fluctuation, elastic restraint; Example: $\lambda$ controls exploration-exploitation balance |
| **10.5** | Bayesian Linear Regression | Full posterior: $p(\mathbf{w}|\mathcal{D}) = \mathcal{N}(\mathbf{w} | \mathbf{w}_N, S_N)$ (conjugate Gaussian-Gaussian); posterior mean: $\mathbf{w}_N = \beta S_N X^T \mathbf{y}$ (identical to Ridge solution); posterior covariance: $S_N = (\alpha I + \beta X^T X)^{-1}$ (uncertainty in weights); posterior predictive: $p(y_*|\mathbf{x}_*, \mathcal{D}) = \mathcal{N}(y_* | \mathbf{w}_N^T \mathbf{x}_*, \sigma_N^2)$ (ensemble average over $\mathbf{w}$); thermal analogy: prediction as ensemble averaging; Example: full distribution vs point estimate |
| **10.6** | Logistic Regression — From Lines to Boundaries | Binary classification: $y \in \{0,1\}$; sigmoid function: $p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^T \mathbf{x}}}$ (maps linear score to probability); Bernoulli likelihood: ML $\to$ minimize cross-entropy loss $L(\mathbf{w}) = -\sum_i [y_i \ln p_i + (1-y_i)\ln(1-p_i)]$; decision boundary: $\mathbf{w}^T \mathbf{x} = 0$ (linear hyperplane); convex optimization (gradient descent); physical analogy: equilibrium under class forces; Example: sigmoid as Boltzmann factor |
| **10.7** | Linear Discriminant Analysis (LDA) | Generative model: class-conditional Gaussians $p(\mathbf{x}|y=k) = \mathcal{N}(\mathbf{x}|\mathbf{\mu}_k, \Sigma)$ (shared covariance); decision rule: $\arg\max_k p(y=k|\mathbf{x}) \propto \arg\max_k p(\mathbf{x}|y=k) p(y=k)$ (Bayes' Theorem); linear boundary: shared $\Sigma$ forces linear separation; QDA: unique $\Sigma_k$ yields quadratic boundaries; physical analogy: comparing harmonic wells (Gaussian potentials), boundary where free energies equal; Example: generative vs discriminative classification |
| **10.8** | Multiclass and Multivariate Extensions | Multiclass (Softmax): $K$ classes, $p(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_j e^{\mathbf{w}_j^T \mathbf{x}}}$ (Boltzmann distribution, partition function); energy interpretation: $\mathbf{w}_k^T \mathbf{x}$ as negative energy; multivariate regression: $\hat{\mathbf{y}} = W^T \mathbf{x}$ (low-rank linear mapping); connection to PCA (principal axes), bridge to deep networks (stacked linear layers + nonlinearity); Example: Softmax as standard output layer |
| **10.9** | Worked Example — Predicting an Ising Phase | Binary classification of Ising phases: ordered ($T < T_c$, $y=1$) vs disordered ($T > T_c$, $y=0$); features: magnetization $M$, temperature $T$, energy $E$; logistic regression finds decision boundary $\mathbf{w}^T \mathbf{x} = 0$; physical insight: learned boundary as critical line (phase separation), free energy equivalence; Example: ML automating phase transition discovery |
| **10.10** | Code Demo — Linear Regression & Logistic Classification | Linear regression: Python implementation, least-squares fit $y = 3x + 5 + \text{noise}$, visualization of orthogonal projection (red line minimizes squared residuals); logistic regression: 2D classification, sigmoid decision boundary $\mathbf{w}^T \mathbf{x} = 0$ (black contour where $p=0.5$), class probability heatmap; observation: geometric equilibrium (regression residuals perpendicular, classification forces balanced); Example: ML principles visualized |
| **10.11** | Bias–Variance and Generalization | Error decomposition: $\mathbb{E}[(\hat{y}-y)^2] = (\text{Bias})^2 + \text{Variance} + \sigma^2$; bias (underfitting, rigid model, high $\lambda$) vs variance (overfitting, flexible model, low $\lambda$); physical analogy: stiffness (high bias, large spring constant) vs fluctuation (high variance, thermal noise); optimal $\lambda$ minimizes total energy (error); Example: regularization balances potential energy (fit) vs penalty energy (complexity) |
| **10.12** | Takeaways & Bridge to Chapter 11 | Linear models unified by ML: regression (Gaussian noise, squared error), classification (Bernoulli noise, cross-entropy); geometry: regression as orthogonal projection, classification as equilibrium boundary; regularization as Bayesian prior (L2 = Gaussian, L1 = Laplace); bias-variance as stiffness-fluctuation duality; bridge to Chapter 11: from independent features to networks of dependencies (graphical models, message passing, belief propagation); Example: linear models as harmonic oscillators, foundation for complex hierarchies |

## **10.1 From Inference to Prediction**

---

### **Recap: The Shift in Goal**

In **Chapter 9: Bayesian Thinking and Inference**, the goal was purely **inference**: to update beliefs and characterize the full posterior probability distribution $P(\mathbf{\theta}|\mathcal{D})$. The output was knowledge about the parameter space, including uncertainty.

The current goal is **prediction**: to use the inferred relationships to forecast future outcomes. We move from the theoretical characterization of $P(\mathbf{\theta})$ to the practical calculation of $\hat{y}$ given a new $\mathbf{x}$.

---

### **The Predictive Goal: Linear Models**

We focus on the simplest, most fundamental model class: the **linear model**. The model uses a set of weights $\mathbf{w}$ and a bias $b$ (collectively the parameters $\mathbf{\theta}$) to transform the input features $\mathbf{x}$ into a predicted output $\hat{y}$:

$$
\hat{y} = f_{\mathbf{\theta}}(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b
$$

---

### **Two Flavors of Prediction**

Linear models are versatile and form the basis for two essential types of learning tasks:

1.  **Regression:** The predicted output $y$ is a **continuous value** ($y \in \mathbb{R}$).
    * *Example:* Predicting the energy of a molecule or the price of a house.
2.  **Classification:** The predicted output $y$ is a **discrete label** ($y \in \{0, 1, \dots\}$).
    * *Example:* Predicting the phase of an Ising model (ordered or disordered) or whether an email is spam.

---

### **Unifying Viewpoint: Maximum Likelihood**

Despite the difference in output type, both Linear Regression and Logistic Regression (classification) share a crucial conceptual foundation, uniting them with the principles of Chapter 9:

* **Energy Minimization:** Both methods minimize an **expected loss function** (squared error for regression, cross-entropy for classification).
* **Maximum Likelihood (ML) Inference:** Crucially, minimizing these specific loss functions is mathematically **equivalent to performing Maximum Likelihood inference** under a particular assumption about the noise.

The goal of finding the single optimal weight vector $\mathbf{w}^*$ is, therefore, a return to the optimization framework of Part II, but now explicitly justified by the **probabilistic principles of inference** from Part III.

## **10.2 Linear Regression as Maximum Likelihood**

**Linear Regression** is the canonical model for **continuous variable prediction** ($y \in \mathbb{R}$). While often taught purely through the **least-squares** geometric objective, its powerful foundation lies in **probabilistic inference** via the **Maximum Likelihood (ML) principle**.

---

### **The Probabilistic Model**

Linear regression assumes that the output $y_i$ is a noisy linear function of the input vector $\mathbf{x}_i$. The noise is assumed to be **independent and identically distributed (i.i.d.) Gaussian (Normal) noise** ($\epsilon_i$), which has a mean of zero and a fixed variance $\sigma^2$:

$$
y_i = \mathbf{w}^\top \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

This assumption implies that the probability of observing a specific target value $y_i$ for a given input $\mathbf{x}_i$ is a Gaussian distribution centered exactly on the model's prediction $\mathbf{w}^\top \mathbf{x}_i$:

$$
p(y_i|\mathbf{x}_i, \mathbf{w}, \sigma^2) = \mathcal{N}(y_i | \mathbf{w}^\top \mathbf{x}_i, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}\right]
$$

---

### **Maximizing the Likelihood**

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, the overall **likelihood** of the parameters $\mathbf{w}$ (and $\sigma^2$) is the product of the likelihoods for all individual, independent observations:

$$
p(\mathcal{D}|\mathbf{w}, \sigma^2) = \prod_{i=1}^N p(y_i|\mathbf{x}_i, \mathbf{w}, \sigma^2) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}\right]
$$

To find the **Maximum Likelihood Estimate ($\mathbf{w}_{\text{ML}}$)**, we maximize this quantity. It's simpler to maximize the **log-likelihood**, which converts the product into a sum:

$$
\ln p(\mathcal{D}|\mathbf{w}, \sigma^2) = \sum_{i=1}^N \left[ -\frac{1}{2} \ln(2\pi\sigma^2) - \frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2} \right]
$$

---

### **Log-Likelihood ↔ Least Squares**

To maximize this log-likelihood with respect to the weights $\mathbf{w}$, we only need to consider the term that depends on $\mathbf{w}$. Since we seek to *maximize* the negative term in the exponent, this is equivalent to **minimizing the sum of squared errors** (SSE):

$$
\mathbf{w}_{\text{ML}} = \arg\min_{\mathbf{w}} \sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
$$

$$
L(\mathbf{w}) = \sum_i (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 \quad \text{— The Least-Squares Objective}
$$

This establishes a critical conclusion: **Minimizing the Least-Squares error is mathematically equivalent to finding the Maximum Likelihood Estimate (MLE) of the weights, under the fundamental assumption of i.i.d. Gaussian noise**.

---

### **Closed-Form Solution and Physical Analogy**

The least-squares problem is one of the few core optimization challenges that possesses a **closed-form, analytic solution** (requiring no iterative descent methods like those in Part II):

$$
\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}
$$

This solution is found by solving $\nabla L(\mathbf{w}) = 0$.

* **Physical Analogy:** Linear regression is the perfect analogy for a **linear spring model** or **harmonic potential**. The squared error $(y_i - \hat{y}_i)^2$ represents the potential energy stored in a linear spring (elastic energy $E \propto x^2$) connecting the prediction to the true observation. The best-fit line minimizes the total potential energy stored across all data points.

## **10.3 Geometry of Least Squares**

While Section 10.2 established the probabilistic foundation of Linear Regression (MLE under Gaussian noise), the **least-squares objective** also has a profound geometric interpretation rooted in **vector space analysis**. This perspective, crucial in physical modeling, reveals that finding the optimal weight vector $\mathbf{w}^*$ is equivalent to performing a specific **orthogonal projection**.

---

### **Vector-Space Formulation**

Consider the linear regression model written in matrix form, where $X$ is the $N \times D$ **design matrix** (our data matrix from Chapter 1):

$$
\mathbf{y} \approx X\mathbf{w}
$$

* $\mathbf{y}$: The $N$-dimensional vector of target outputs.
* $X$: The matrix whose columns are the feature vectors (the input space).
* $\mathbf{w}$: The $D$-dimensional vector of weights.
* $X\mathbf{w}$: The $N$-dimensional vector of **model predictions** ($\hat{\mathbf{y}}$).

The least-squares objective is to minimize the magnitude of the difference between the actual targets $\mathbf{y}$ and the predictions $\hat{\mathbf{y}}$:

$$
\mathbf{w}^* = \arg\min_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|^2
$$

---

### **Geometric Interpretation: Orthogonal Projection**

The vector of model predictions, $\hat{\mathbf{y}} = X\mathbf{w}$, is a linear combination of the feature columns of $X$. Geometrically, $\hat{\mathbf{y}}$ must lie within the **column space of $X$** (the subspace spanned by the input features).

The least-squares solution $\mathbf{w}^*$ is the one that makes $\hat{\mathbf{y}}$ the **closest vector** in that subspace to the target vector $\mathbf{y}$.

* **Projection:** This closest vector $\hat{\mathbf{y}}$ is the **orthogonal projection** of the target vector $\mathbf{y}$ onto the column space of $X$.
* **Prediction Vector:** The prediction vector $\hat{\mathbf{y}}$ is given by the application of the **projection matrix** $P=X(X^\top X)^{-1}X^\top$ to the target vector $\mathbf{y}$:

$$
\hat{\mathbf{y}} = P\mathbf{y} = X(X^\top X)^{-1}X^\top \mathbf{y}
$$

---

### **Error Orthogonality and Residuals**

A fundamental property of this solution is that the **residual error vector ($\mathbf{y} - \hat{\mathbf{y}}$)** must be **orthogonal** (perpendicular) to every vector in the prediction subspace (the column space of $X$).

Mathematically, this means the dot product between the feature vectors (columns of $X$) and the residual must be zero:

$$
X^\top(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}
$$

This is the condition required for the gradient $\nabla L(\mathbf{w})$ to be zero, confirming that the **least-squares estimate is the unique point of equilibrium** where the projection error is minimized.

---

### **Physical Analogy: Equilibrium Under Constraint**

* **Physical Analogy:** This geometric interpretation is analogous to finding a point of **equilibrium under orthogonal constraint forces**. The model's prediction $\hat{\mathbf{y}}$ is constrained to lie within the space spanned by the input features. The optimal error $\mathbf{y} - \hat{\mathbf{y}}$ is the force (or distance) exerted by the true target. Equilibrium is achieved when the error vector is perfectly orthogonal to the constraint subspace, ensuring **no residual work** can be done by changing the parameters $\mathbf{w}$.
* **Statistical Measure:** The coefficient of determination, $R^2$, is the proportion of variance explained and is also a geometric measure. It is equal to the squared cosine of the angle between the target vector $\mathbf{y}$ and the predicted vector $\hat{\mathbf{y}}$.

Linear regression, therefore, unifies the three pillars of *Volume III*: it is a probabilistic inference (MLE), solved by optimization (minimizing loss), with a direct geometric meaning (orthogonal projection).

## **10.4 Regularization and the Bias–Variance Tradeoff**

The maximum likelihood solution for Linear Regression (least squares) is analytically exact, but often leads to models that suffer from **overfitting**, especially when dealing with noisy or high-dimensional data. **Regularization** is the technique used to stabilize the model and improve its ability to **generalize** to unseen data.

---

### **Regularization as a Bayesian Prior**

As established in Section 9.3, regularization is the explicit introduction of a **Bayesian prior** into the optimization objective. This effectively adds a penalty term that biases the weights $\mathbf{w}$ toward smaller magnitudes, constraining the model's complexity.

The objective of finding the optimal weights $\mathbf{w}^*$ shifts from pure MLE to **Maximum A Posteriori (MAP)** estimation:

$$
\mathbf{w}_{\text{MAP}} = \arg\min_{\mathbf{w}} [\text{Loss}(\mathbf{w}) + \lambda \cdot \text{Penalty}(\mathbf{w})]
$$

where $\text{Loss}(\mathbf{w})$ is the least-squares error, and $\lambda$ is the **regularization strength**.

---

### **Common Regularization Penalties**

The form of the penalty dictates the resulting Bayesian prior:

* **Ridge Regression ($L^2$ Penalty):** This adds a penalty proportional to the squared magnitude of the weights.

$$
L(\mathbf{w}) = \underbrace{\|\mathbf{y} - X\mathbf{w}\|^2}_{\text{Least Squares Loss}} + \lambda \underbrace{\|\mathbf{w}\|^2}_{\text{L2 Penalty}}
$$

    * **Bayesian Analogy:** Minimizing this objective corresponds to using a **Gaussian prior** $p(\mathbf{w})$ centered at zero. It discourages large weights, effectively applying an **elastic restraint** to the parameter space.

* **LASSO (L1 Penalty):** This adds a penalty proportional to the absolute magnitude of the weights.

$$
L(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_1
$$
    * **Bayesian Analogy:** Minimizing this objective corresponds to using a **Laplace prior**. Unlike $L^2$, the $L^1$ penalty encourages some weights to become **exactly zero**, thus promoting sparsity and performing automatic feature selection.

---

### **The Bias–Variance Tradeoff**

Regularization explicitly addresses the **Bias–Variance Tradeoff**, a core concept in statistical learning that relates model complexity to prediction error. The total expected prediction error can be decomposed as:

$$
\mathbb{E}[(\hat{y}-y)^2] = (\text{Bias})^2 + \text{Variance} + \sigma^2
$$

| Component | Description | Effect of $\lambda$ | Analogy |
| :--- | :--- | :--- | :--- |
| **Bias** | The error due to a model being too simple (underfitting). | **Increases.** A large $\lambda$ forces weights toward zero, making the model rigid. | **Stiffness:** Like a very rigid spring that cannot be stretched to fit the data. |
| **Variance** | The error due to a model being too complex (overfitting). | **Decreases.** Regularization constrains the weights, stabilizing the model against noise in the training data. | **Fluctuation:** Prevents model parameters from oscillating wildly. |

The optimal value of $\lambda$ is found by striking a balance: high enough to significantly reduce variance (overfitting) but low enough to avoid introducing excessive bias (underfitting). Regularization, therefore, is analogous to adding an **elastic restraint** that controls the **fluctuation** (variance) of the parameter vector in the energy landscape.

## **10.5 Bayesian Linear Regression**

While Maximum Likelihood (ML) and Maximum A Posteriori (MAP) estimation provide single point estimates for the weights $\mathbf{w}$, **Bayesian Linear Regression** (BLR) utilizes the full Bayesian framework to derive and characterize the complete **posterior distribution** $p(\mathbf{w}|\mathcal{D})$. This approach explicitly quantifies the uncertainty in the model weights and predictions.

---

### **The Setup: Probabilistic Assumptions**

BLR treats the weights $\mathbf{w}$ as **random variables**, rather than fixed constants. We adopt Gaussian distributions for both the prior belief about the weights and the noise corrupting the data:

* **Prior Distribution:** We assume the weights $\mathbf{w}$ are drawn from a Gaussian distribution centered at zero (an $L^2$ penalty equivalent, Section 10.4):

$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \alpha^{-1}I)
$$

```
where $\alpha$ is the precision (inverse variance) of the prior.
```
* **Likelihood/Noise:** We assume the observed targets $y_i$ are generated by the linear model corrupted by i.i.d. Gaussian noise (as in Section 10.2):

$$
p(\mathbf{y}|X, \mathbf{w}) = \mathcal{N}(\mathbf{y} | X\mathbf{w}, \beta^{-1}I)
$$

```
where $\beta$ is the precision of the noise.

```
---

### **The Posterior Distribution: A Gaussian Result**

Since we use a Gaussian prior and a Gaussian likelihood, the two distributions are **conjugate** (Section 9.4). Applying Bayes' theorem yields a **Posterior distribution** that is also a Gaussian distribution:

$$
p(\mathbf{w}|\mathcal{D}) = \mathcal{N}(\mathbf{w} | \mathbf{w}_N, S_N)
$$

The posterior's mean $\mathbf{w}_N$ and covariance matrix $S_N$ are analytically calculable based on the data ($X, \mathbf{y}$) and the prior precisions ($\alpha, \beta$):
* **Posterior Covariance Matrix ($S_N$):** This matrix captures the uncertainty in the weights.

$$
S_N = (\alpha I + \beta X^\top X)^{-1}
$$

* **Posterior Mean Vector ($\mathbf{w}_N$):** This is the single most probable weight vector.

$$
\mathbf{w}_N = \beta S_N X^\top \mathbf{y}
$$

The posterior mean $\mathbf{w}_N$ is identical to the solution found by **Ridge Regression** (MAP estimate with a Gaussian prior, Section 10.4).

---

### **Prediction: The Ensemble Average**

Unlike ML, which uses only the point estimate $\mathbf{w}_{\text{ML}}$, BLR uses the entire posterior $p(\mathbf{w}|\mathcal{D})$ to make predictions. The **Posterior Predictive Distribution** (Section 9.7) for a new input $\mathbf{x}_*$ is also a Gaussian distribution:

$$
p(y_*|\mathbf{x}_*,\mathcal{D}) = \mathcal{N}(y_* | \mathbf{w}_N^\top \mathbf{x}_*, \sigma^2_N)
$$

* **Prediction Mean:** $\mathbf{w}_N^\top \mathbf{x}_*$.
* **Prediction Variance ($\sigma^2_N$):** This variance quantifies the total uncertainty in the prediction. It includes both the inherent noise in the data ($\beta^{-1}$) and the uncertainty in the model fit ($\mathbf{x}_*^\top S_N \mathbf{x}_*$).

---

### **Physical Analogy: Thermal Ensemble of Models**

Bayesian Linear Regression is analogous to analyzing a **thermal ensemble of linear models**.

* The total set of possible models $\mathbf{w}$ is our ensemble of states.
* The posterior $p(\mathbf{w}|\mathcal{D})$ weights each model according to its probability (its evidential energy).
* The prediction process is an **ensemble average** over all these weighted models. The final prediction is the most robust forecast possible because it accounts for the uncertainty and credibility of every single potential model.

## **10.6 Logistic Regression — From Lines to Boundaries**

While Linear Regression (Sections 10.2–10.5) is designed for predicting **continuous outputs ($y \in \mathbb{R}$)** based on the assumption of Gaussian noise, **Logistic Regression** is the canonical linear model for **binary classification ($y \in \{0, 1\}$)**. It applies the linear model output to a nonlinear function to yield a probability, thus bridging continuous energy minimization with probabilistic decision boundaries.

---

### **The Classification Model: The Sigmoid Function**

In classification, the goal is to model the **probability** that an input $\mathbf{x}$ belongs to class 1, denoted $P(y=1|\mathbf{x})$. The output of the linear model, $\mathbf{w}^\top \mathbf{x} + b$, is a continuous value ranging from $-\infty$ to $+\infty$, which cannot directly be interpreted as a probability.

Logistic Regression solves this by passing the linear output through the **Sigmoid (or Logistic) function** $\sigma(z)$:

$$
p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{x}}}
$$

* **Effect:** The sigmoid function maps the continuous input $z = \mathbf{w}^\top \mathbf{x}$ (often called the **logit** or score) to the range $[0, 1]$, resulting in a valid probability.
* **Energy Analogy:** The term $e^{-\mathbf{w}^\top \mathbf{x}}$ relates directly to an **unnormalized energy** or likelihood ratio, placing the model firmly within the exponential family (Chapter 2.2).

---

### **The Loss Function: Maximum Bernoulli Likelihood**

Logistic Regression is an **Maximum Likelihood Estimate (MLE)** under the assumption that the targets $y_i$ are generated by a **Bernoulli distribution** (i.e., a coin flip determined by the probability $p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$).

Maximizing the Bernoulli log-likelihood is equivalent to minimizing the **Cross-Entropy Loss**:

$$
L(\mathbf{w}) = -\sum_i [y_i \ln p_i + (1-y_i)\ln(1-p_i)]
$$

This loss function is **convex** (Section 4.3), ensuring that gradient-based optimization is guaranteed to find the unique global minimum.

---

### **The Decision Boundary and Gradient Dynamics**

The model classifies a new input $\mathbf{x}$ based on whether the probability $p(y=1|\mathbf{x})$ is greater than $0.5$.

* **Boundary Condition:** The probability $p$ equals $0.5$ exactly when the input to the sigmoid is zero: $\sigma(z) = 0.5 \iff z = 0$.
* **Decision Boundary:** The decision surface is therefore defined by the linear equation: $\mathbf{w}^\top \mathbf{x} = 0$. This means Logistic Regression finds a linear hyperplane that optimally separates the two classes in the input space.
* **Optimization:** Unlike linear regression, logistic regression has no closed-form solution. The weights $\mathbf{w}$ must be found iteratively using gradient methods (Chapter 5). The gradient of the loss is used to drive the parameter updates:

$$
\nabla L = X^\top(\sigma(X\mathbf{w}) - \mathbf{y})
$$

---

### **Analogy: Force on a Separating Hyperplane**

* **Physical Analogy:** The optimization process is analogous to finding a state of **equilibrium** for the decision hyperplane. Each misclassified data point exerts a proportional **"force"** on the hyperplane, attempting to rotate it toward the correct orientation.
* **Equilibrium:** The optimal decision boundary ($\mathbf{w}^*$) is the one where the total force from all data points balances out, resulting in minimal loss.

## **10.7 Linear Discriminant Analysis (LDA)**

While **Logistic Regression** (Section 10.6) is a **discriminative model** that directly finds a linear boundary separating classes, **Linear Discriminant Analysis (LDA)** is a **generative model** that achieves linear classification by first modeling the underlying probability distributions of the data within each class. LDA is fundamentally rooted in the Gaussian (Normal) distribution.

---

### **The Generative Model Assumption**

LDA assumes that the data belonging to each class $k$ (e.g., class $y=0$ or $y=1$) is generated by a **Gaussian distribution**:

$$
p(\mathbf{x}|y=k) = \mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma)
$$

The key distinguishing feature of LDA is the assumption that **all classes share the same covariance matrix ($\Sigma$)**. This assumption is critical because it forces the resulting decision boundaries to be linear.

* **$\mathbf{\mu}_k$:** The mean vector (centroid) for class $k$.
* **$\Sigma$:** The single, shared covariance matrix across all classes.

---

### **The Decision Rule: Maximizing Posterior Probability**

The goal of LDA is to classify a new input $\mathbf{x}$ by assigning it to the class $k$ that has the highest **posterior probability** $p(y=k|\mathbf{x})$. This is achieved by maximizing the numerator of Bayes' Theorem (Section 9.2):

$$
\arg\max_k p(y=k|\mathbf{x}) \propto \arg\max_k p(\mathbf{x}|y=k) p(y=k)
$$

* $p(\mathbf{x}|y=k)$: The class-conditional Gaussian likelihood (the generative part).
* $p(y=k)$: The class prior (the marginal probability of seeing class $k$ in the data).

---

### **Connection to Linear Boundaries**

When the Gaussian assumption of shared covariance ($\Sigma$) is plugged into this maximization, the decision boundary between any two classes $k$ and $j$ simplifies to a function that is **linear in $\mathbf{x}$**.

* **Generalization:** If, instead, we allow each class to have its own unique covariance matrix ($\Sigma_k$), the resulting decision boundary becomes quadratic, leading to **Quadratic Discriminant Analysis (QDA)**.

---

### **Analogy: Comparing Harmonic Wells**

LDA's classification process is analogous to **comparing the local potential of two harmonic wells**.

* **Harmonic Well:** Each class $k$ is modeled as a high-probability region (a minimum in the energy landscape) defined by a Gaussian potential $\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k, \Sigma)$.
* **Classification:** To classify a point $\mathbf{x}$, LDA simply measures the relative "energy distance" (or distance in units of the shared covariance) from $\mathbf{x}$ to the center $\mathbf{\mu}_k$ of each Gaussian well, modified by the class priors.
* **The Boundary:** The resulting linear decision boundary represents the line of points where the probability of belonging to one Gaussian well equals the probability of belonging to the other.

LDA is therefore a **Bayesian classifier** under the explicit, generative assumption of Gaussian class structure.

## **10.8 Multiclass and Multivariate Extensions**

The linear models discussed so far—Linear Regression (single continuous output) and Logistic Regression/LDA (binary classification)—represent the simplest, two-state cases. We now extend these foundational principles to problems involving more complex outputs: predicting one of **multiple categories** or predicting **multiple continuous values** simultaneously.

---

### **Multiclass Classification (Softmax)**

When the classification problem involves $K > 2$ possible output labels (e.g., classifying handwritten digits $0$ through $9$), binary Logistic Regression is replaced by its generalized form, often called **Multiclass Logistic Regression** or the **Softmax Classifier**.

* **Architecture:** The model is structured with $K$ separate weight vectors ($\mathbf{w}_1, \dots, \mathbf{w}_K$) and bias terms ($b_1, \dots, b_K$), one for each class. Each weight vector calculates a linear score $z_k = \mathbf{w}_k^\top \mathbf{x}$.
* **The Softmax Function:** The scores $z_k$ are converted into a probability distribution over the $K$ classes using the **Softmax function**:

$$
p(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_j e^{\mathbf{w}_j^\top \mathbf{x}}}
$$

* **Energy Interpretation:** The softmax function is a direct generalization of the logistic function and is related to the **Boltzmann distribution** (Chapter 2.2). The exponential term $e^{\mathbf{w}_k^\top \mathbf{x}}$ is an unnormalized probability, where the score $\mathbf{w}_k^\top \mathbf{x}$ acts as the negative energy for class $k$. The denominator acts as the normalizing **Partition Function** (Chapter 2.2).
* **Bridge to Deep Learning:** This energy model is the standard final output layer for virtually all deep neural networks.

---

### **Multivariate Regression (Multi-Output)**

In many physical systems, the goal is to predict not just a single scalar output, but an entire vector of correlated outputs, $\mathbf{y} \in \mathbb{R}^M$.

* **Architecture:** This is handled by **Multivariate Regression**, where the input vector $\mathbf{x}$ is mapped to the output vector $\mathbf{y}$ using a matrix of weights, $W$, instead of a vector $\mathbf{w}$.

$$
\hat{\mathbf{y}} = W^\top \mathbf{x}
$$

* **Interpretation:** The weight matrix $W$ performs a **low-rank linear mapping**, learning the fundamental, correlated axes of variation in the output space.
* **Connection to PCA:** This is directly related to the concept of **Principal Component Analysis (PCA)** (Chapter 1.3), where the solution attempts to align the mapping with the principal axes that capture the majority of the data's variance. The goal is to find a compact subspace that accurately predicts all correlated outputs simultaneously.

---

### **Bridge to Deep Hierarchies**

These extensions—multiclass Softmax and multivariate linear mapping—provide the architectural building blocks for more complex learning systems.

* **Deep Networks:** A multi-layer neural network (Chapter 12) is simply a series of these transformations stacked together, where each layer learns a unique linear mapping ($W$) followed by a non-linear activation.
* The final output layer often uses the Softmax function to produce a probabilistic output. The subsequent deep learning chapters (Part IV) demonstrate how these basic linear maps can be composed into deep **hierarchies** (Chapter 13) to approximate complex, non-linear functions.

## **10.9 Worked Example — Predicting an Ising Phase**

This worked example grounds the principles of linear classification (Logistic Regression, Section 10.6) in a physical problem: identifying the transition between the **ordered (ferromagnetic) and disordered (paramagnetic) phases** of the 2D Ising model based solely on thermodynamic observables. The classification boundary learned by the model directly maps to the **critical line** separating the physical states.

---

### **The Problem and Data**

* **Data Source:** Simulation outputs from a 2D Ising model run across a range of temperatures $T$, spanning below and above the critical temperature $T_c$.
* **Target (Label $y$):** The output is binary: $y=1$ for the **Ordered Phase** ($T < T_c$) and $y=0$ for the **Disordered Phase** ($T > T_c$).
* **Features (Inputs $\mathbf{x}$):** The model is trained to predict the phase based on macroscopic variables extracted from the simulation snapshots:

$$
\mathbf{x} = [\text{Magnetization } M, \text{ Temperature } T, \text{ Energy } E]
$$

* **Task:** Fit a Logistic Regression model to find the optimal decision boundary $\mathbf{w}^\top \mathbf{x} = 0$ that separates these two phase labels.

---

### **Steps: Building a Phase Classifier**

1.  **Feature Matrix Construction:** Build the input matrix $X$ using the thermodynamic observables $M$, $T$, and $E$.
2.  **Model Training:** Fit the Logistic Regression classifier to maximize the Bernoulli log-likelihood (minimize the cross-entropy loss). This optimization yields the weight vector $\mathbf{w}^*$ that defines the best linear decision boundary.
3.  **Visualization:** Since the phase boundary is best understood in the context of two observables, the result is often visualized as the decision boundary (where $p(y=1|\mathbf{x}) = 0.5$) projected onto the $(M, E)$ or $(M, T)$ feature space.

---

### **Observation and Physical Insight**

The success of the Logistic Regression model provides a powerful **physical insight**:

* **Classification as Phase Separation:** The model successfully learns the highly nonlinear relationship between $M$, $E$, and $T$ and the phase state. The classification boundary ($\mathbf{w}^\top \mathbf{x} = 0$) learned by the model acts like a data-driven **critical line** or **phase boundary**.
* **Free-Energy Parallel:** The decision boundary separates regions of high probability for the ordered state from regions of high probability for the disordered state. This boundary is directly analogous to the line in the state space where the **free energy** of the two phases is equal ($F_{\text{ordered}} = F_{\text{disordered}}$). The logistic model has implicitly learned the most statistically robust linear approximation of this underlying free-energy separation.

!!! example "Logistic Regression as Phase Transition Detector"
```
Training logistic regression on Ising simulation data with features $[M, T, E]$ automatically discovers the critical temperature $T_c$:

* Below $T_c$: High magnetization $M$, low energy $E$ → model predicts $y=1$ (ordered phase)
* Above $T_c$: Near-zero $M$, high $E$ → model predicts $y=0$ (disordered phase)
* Decision boundary: $\mathbf{w}^\top \mathbf{x} = 0$ maps the critical line where free energies balance

This demonstrates ML automating physics discovery—the learned boundary is the data-driven phase diagram.

```
Logistic Regression, therefore, serves as an effective method for **automating the discovery of phase transitions** using supervised learning principles.

## **10.10 Code Demo — Linear Regression & Logistic Classification**

This demo provides a practical implementation of the two fundamental linear models, showing how the principles of **Maximum Likelihood (ML) estimation** translate into distinct geometric solutions for continuous (Regression) and discrete (Classification) data.

---

### **1. Linear Regression (Continuous Prediction)**

This part demonstrates **Linear Regression**, which is the **ML Estimate under Gaussian noise** (minimizing squared error, Section 10.2).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

## Synthetic regression data: y = 3x + 5 + noise

x = np.linspace(0, 10, 50)
y = 3*x + 5 + np.random.randn(50)*2
## Fit the linear model

model = LinearRegression().fit(x.reshape(-1,1), y)

plt.figure(figsize=(9, 4))
plt.scatter(x, y, label='Data Points')
## Plot the predicted line (the orthogonal projection, Section 10.3)

plt.plot(x, model.predict(x.reshape(-1,1)), color='r', lw=2, label='Linear Fit')
plt.title('Linear Regression Fit (Minimizing Squared Error)')
plt.xlabel('Input Feature (x)')
plt.ylabel('Target Output (y)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

#### Observation:

The red line represents the solution $\mathbf{w}^*$. This line minimizes the sum of squared vertical distances (the elastic potential energy) from every data point to the line. Geometrically, the vector of residuals $(\mathbf{y} - \hat{\mathbf{y}})$ is orthogonal to the line, indicating the unique point of equilibrium (Section 10.3).

---

### **2. Logistic Regression (Binary Classification)**

This part demonstrates **Logistic Regression**, which is the **ML Estimate under Bernoulli likelihood** (minimizing cross-entropy loss, Section 10.6).

```python
## Synthetic classification data (2D input, binary output)

np.random.seed(0)
X = np.random.randn(200,2)
## Create a hidden linear boundary to generate labels: y = 1 if 0.5*x1 - 0.7*x2 > 0

y = (X[:,0]*0.5 + X[:,1]*(-0.7) > 0).astype(int)
## Fit the logistic classifier

clf = LogisticRegression().fit(X, y)

plt.figure(figsize=(9, 6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', alpha=0.6, label='Data by Class')

## Define a grid for plotting the decision boundary

xx, yy = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
input_grid = np.c_[xx.ravel(), yy.ravel()]
## Predict probability of class 1 across the grid

Z = clf.predict_proba(input_grid)[:,1].reshape(xx.shape)

## Plot the decision boundary (where p=0.5, or w^T x = 0, Section 10.6)

plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=3, label='Decision Boundary')

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature $x_1$')
plt.ylabel('Feature $x_2$')
plt.show()
```

#### Observation:

The contour plot shows a continuous color transition from one class probability (cool) to the other (warm). The thick black line represents the final, optimal **linear decision boundary** ($\mathbf{w}^\top \mathbf{x} = 0$). This line is the point of equilibrium where the "forces" exerted by misclassified data points balance out (Section 10.6). The two methods visually demonstrate the different geometric consequences of optimizing for continuous versus discrete outcomes.

## **10.11 Bias–Variance and Generalization**

The **Bias–Variance Tradeoff** is a central organizing principle in statistical learning, explaining the tension between a model that is too simple to capture the data structure and one that is too complex and merely memorizes the noise. The goal of any learning process is to achieve optimal **generalization**—performance on unseen data—by balancing these two sources of error.

---

### **The Decomposition of Error**

The total expected prediction error (the squared difference between the true target $y$ and the prediction $\hat{y}$) can be analytically decomposed into three non-negative components:

$$
\mathbb{E}[(\hat{y}-y)^2] = (\text{Bias})^2 + \text{Variance} + \sigma^2
$$

1.  **Bias ($(\text{Bias})^2$):** The error arising from a model being too **rigid** or overly simplified (e.g., using a linear model for highly nonlinear data). A high bias leads to **underfitting**.
2.  **Variance ($ \text{Variance}$):** The error arising from a model being too **flexible**. High variance means the model's output changes drastically based on small, random fluctuations in the training data, leading to **overfitting**.
3.  **Irreducible Error ($\sigma^2$):** The inherent noise or complexity in the data itself (e.g., measurement error) that no model can eliminate.

!!! tip "Regularization Controls the Bias-Variance Tradeoff"
```
The regularization strength $\lambda$ directly controls the bias-variance balance:

* **High $\lambda$** (strong regularization): Adds stiffness → high bias, low variance → underfitting
* **Low $\lambda$** (weak regularization): Allows flexibility → low bias, high variance → overfitting
* **Optimal $\lambda$**: Minimizes total error $= (\text{Bias})^2 + \text{Variance} + \sigma^2$

This is the statistical manifestation of the stiffness-fluctuation duality in physics.

```
---

### **The Trade-off and Physical Analogy**

The challenge lies in the inverse relationship between bias and variance, which is governed by the model's complexity (e.g., the degree of a polynomial or the magnitude of regularization $\lambda$).

| Scenario | Model Complexity | Bias | Variance | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Simple Model** | Low (High $\lambda$) | High | Low | **Underfitting** (Rigid, poor fit) |
| **Complex Model** | High (Low $\lambda$) | Low | High | **Overfitting** (Fits noise, poor generalization) |

**Physical Analogy: Stiffness vs. Fluctuation**

This statistical tension directly mirrors physical dynamics:
* **High Bias** is analogous to a system with too much **stiffness** (e.g., a high elastic spring constant $k$). The model is too constrained (too rigid) to deform and fit the data.
* **High Variance** is analogous to a system with excessive **fluctuation** (e.g., high thermal noise). The model parameters oscillate wildly in response to microscopic data changes.

---

### **Minimizing Total Energy**

The goal of regularization (Section 10.4) is precisely to control this trade-off. By setting the regularization strength ($\lambda$), we balance the **potential energy** associated with the error (low bias) against the **penalty energy** associated with complexity (low variance). The optimal model is the one that minimizes the **total expected energy (error)**, achieving the best possible compromise between rigidity and flexibility.

??? question "Why Can't We Eliminate Both Bias and Variance?"
```
The bias-variance tradeoff is fundamental because reducing one typically increases the other:

* **Reducing bias** requires a more flexible model (more parameters, lower $\lambda$)
* **More flexibility** increases sensitivity to training data variations (higher variance)
* **Reducing variance** requires constraints (higher $\lambda$, simpler model)
* **More constraints** reduce the model's ability to fit the true function (higher bias)

This is not a limitation of algorithms but a fundamental property of finite data:
$\text{Complexity}(\text{Model}) \times \text{Size}(\text{Data}) = \text{const}$

```
The Bias–Variance Tradeoff reappears throughout every layer of machine learning, from linear models (this chapter) to neural networks (Chapter 12–13) to reinforcement learning (Chapter 15).

## **10.12 Takeaways & Bridge to Chapter 11**

This chapter successfully applied the principles of **probabilistic inference** (Chapter 9) and **optimization** (Part II) to the foundational predictive models: **Linear Regression** and **Logistic Regression**.

---

### **Key Takeaways from Chapter 10**

* **Inference Unifies Linear Models:** Both Linear Regression (for continuous output) and Logistic Regression (for binary output) are unified by the principle of **Maximum Likelihood (ML) Estimation**. Minimizing the familiar least-squares error is equivalent to ML under the assumption of **Gaussian noise**, while minimizing cross-entropy loss is equivalent to ML under **Bernoulli noise**.
* **Prediction is Geometric:** Finding the optimal weight vector $\mathbf{w}^*$ is a geometric problem.
    * In regression, the solution vector $\hat{\mathbf{y}}$ is the **orthogonal projection** of the target vector $\mathbf{y}$ onto the subspace spanned by the data features.
    * In classification, the solution is the **linear decision boundary** that achieves equilibrium by balancing the "forces" exerted by the different classes.
* **Priors and Energy Control:** Regularization (like $L^1$ or $L^2$) is the direct application of a **Bayesian prior** (Section 9.3) that introduces an energetic penalty on complexity. This penalty controls the **Bias–Variance Tradeoff**, balancing model rigidity against fluctuation.
* **Foundational Analogy:** Linear models serve as the "harmonic oscillators" of learning. They show that regularization is a form of **elastic restraint**, and fitting minimizes the **total potential (error) energy**.

---

### **Bridge to Chapter 11: Networks of Dependencies**

The linear family of models (Regression, Logistic, LDA) assumes that features contribute to the output largely independently, or through simple, local linear combinations.

However, in complex physical and informational systems, variables are linked in intricate, non-linear **networks of dependencies** (Chapter 1.1).

* A complex spin system is not just $N$ independent spins; it is $N$ spins linked by local **couplings** $J_{ij}$.
* A causal system involves chains of conditional dependencies (e.g., $A \to B \to C$).

In **Chapter 11: "Graphical Models & Probabilistic Graphs,"** we will generalize the principles of inference to these interconnected systems. We will use graph theory to explicitly model these dependencies, and we will see how inference transforms into a process of **message passing** (belief propagation) across the network—the statistical dynamics of a large, collectively interacting system.

---

## **References**

[1] **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Comprehensive treatment of linear models, Bayesian regression, and classification]
[2] **Murphy, K. P.** (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. [Modern probabilistic perspective on regression and classification]
[3] **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*. Springer. [Definitive reference for bias-variance tradeoff and regularization]
[4] **MacKay, D. J. C.** (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press. [Bayesian linear regression and information-theoretic perspective]
[5] **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. [Foundation for understanding linear models as building blocks of neural networks]
[6] **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2013). *An Introduction to Statistical Learning*. Springer. [Accessible introduction with practical examples]
[7] **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. [Bayesian treatment of regression with uncertainty quantification]
[8] **Mehta, P., et al.** (2019). "A high-bias, low-variance introduction to Machine Learning for physicists." *Physics Reports*. [Physics-oriented perspective on ML fundamentals]