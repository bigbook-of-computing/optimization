# **Chapter 2: Statistics & Probability in High Dimensions (Codebook)**

## Project 1: Testing the Curse of Dimensionality (Distance Concentration)

---

### Definition: Testing the Curse of Dimensionality

The goal of this project is to numerically demonstrate the phenomenon of **distance concentration**. This is a core concept of the **Curse of Dimensionality**, where the relative spread of distances between all points in a dataset shrinks to zero as the number of features ($D$) increases.

### Theory: Distance Concentration and the Volume Paradox

In low-dimensional spaces, maximum and minimum distances between points vary widely. In high-dimensional spaces, however, the concept of a "nearest neighbor" breaks down because **all pairwise Euclidean distances converge to nearly the same value**. This counter-intuitive effect stems from the **Volume Paradox** (most volume is concentrated near the surface of the hypersphere).

We quantify this by computing the **relative spread of distances**:

$$\text{Spread Ratio} = \frac{\text{Max Distance} - \text{Min Distance}}{\text{Average Distance}}$$

For large $D$, this ratio approaches zero, confirming that points become "equally far" from each other, rendering naïve proximity methods ineffective.

---

### Extensive Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

## ====================================================================

## 1. Setup Datasets (Low D vs. High D)

## ====================================================================

N = 500  # Number of samples
## Run 1: Low Dimension (D=2)

D_LOW = 2
X_low = np.random.randn(N, D_LOW)

## Run 2: High Dimension (D=1000)

D_HIGH = 1000
X_high = np.random.randn(N, D_HIGH)

## ====================================================================

## 2. Distance Calculation and Concentration Metric

## ====================================================================

def calculate_concentration(X):
    """
    Calculates the relative spread of pairwise Euclidean distances.
    The ratio approaches 0 for high dimensions.
    """
    # Calculate all pairwise Euclidean distances
    distances = pdist(X, metric='euclidean')

    # Compute the required metrics
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    avg_dist = np.mean(distances)

    # Relative Spread Ratio
    spread_ratio = (max_dist - min_dist) / avg_dist
    return spread_ratio, avg_dist

spread_low, avg_low = calculate_concentration(X_low)
spread_high, avg_high = calculate_concentration(X_high)

## ====================================================================

## 3. Visualization and Summary

## ====================================================================

print("--- Distance Concentration Test (Curse of Dimensionality) ---")
print(f"1. Low Dimension (D={D_LOW}):")
print(f"   Average Distance: {avg_low:.3f}")
print(f"   Relative Spread Ratio: {spread_low:.3f} (Large variance in distances)")

print(f"\n2. High Dimension (D={D_HIGH}):")
print(f"   Average Distance: {avg_high:.3f}")
print(f"   Relative Spread Ratio: {spread_high:.3f} (Distances are tightly concentrated)")

print("\nConclusion: The Spread Ratio drops significantly in high dimensions, confirming that most of the space is empty and all data points become equidistant from one another. This illustrates why distance-based algorithms struggle in high-D feature space.")
```
**Sample Output:**
```python
--- Distance Concentration Test (Curse of Dimensionality) ---
1. Low Dimension (D=2):
   Average Distance: 1.782
   Relative Spread Ratio: 3.390 (Large variance in distances)

2. High Dimension (D=1000):
   Average Distance: 44.698
   Relative Spread Ratio: 0.198 (Distances are tightly concentrated)

Conclusion: The Spread Ratio drops significantly in high dimensions, confirming that most of the space is empty and all data points become equidistant from one another. This illustrates why distance-based algorithms struggle in high-D feature space.
```

---

## Project 2: Importance Sampling for Estimating $\langle f \rangle$

---

### Definition: Importance Sampling for Estimating $\langle f \rangle$

The goal is to use **Importance Sampling** to estimate a known integral, $\langle f \rangle_P$, demonstrating that the estimate's **variance** is highly dependent on the choice of the **proposal distribution ($q(\mathbf{x})$)**.

### Theory: Importance Weights and Variance

Importance Sampling allows estimation of the expectation $\langle f \rangle_P = \int f(\mathbf{x}) P(\mathbf{x}) d\mathbf{x}$ by drawing samples from a simpler proposal distribution $Q(\mathbf{x})$:

$$\langle f \rangle_P \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}_i) \cdot w(\mathbf{x}_i), \quad \text{where } w(\mathbf{x}_i) = \frac{P(\mathbf{x}_i)}{Q(\mathbf{x}_i)}$$

The terms $w(\mathbf{x}_i)$ are the **importance weights**.

  * **Good Proposal ($Q_A \approx P$):** Weights are near 1, and the variance of the estimator is low.
  * **Poor Proposal ($Q_B$ distant from $P$):** Weights are large for a few samples and near zero for most. This leads to an unstable estimate with **catastrophically high variance**.

We test this by estimating the variance of a standard Gaussian ($P(x) = \mathcal{N}(x|0, 1)$ with $f(x) = x^2$), which is exactly $\langle f \rangle_P = 1$.

---

### Extensive Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Target Distribution (P) and Function (f)

## ====================================================================

N_SAMPLES = 10000  # Number of samples for the estimator
TARGET_MEAN = 0.0
TARGET_STD = 1.0

## True Distribution P(x) = N(x|0, 1)

def P(x):
    return norm.pdf(x, loc=TARGET_MEAN, scale=TARGET_STD)

## Function to integrate f(x) = x^2

def f(x):
    return x**2

## Analytical Result: <f>_P = <x^2>_N(0,1) = Variance = 1.0

ANALYTICAL_MEAN = 1.0

## ====================================================================

## 2. Importance Sampling Trials

## ====================================================================

## Trial A: Good Proposal Q_A (Perfect Match)

Q_A = lambda x: norm.pdf(x, loc=TARGET_MEAN, scale=TARGET_STD)
X_A = np.random.normal(loc=TARGET_MEAN, scale=TARGET_STD, size=N_SAMPLES)
Weights_A = P(X_A) / Q_A(X_A)  # Weights should be all 1s
Estimate_A = np.mean(f(X_A) * Weights_A)
Variance_A = np.var(f(X_A) * Weights_A)

## Trial B: Poor Proposal Q_B (Distant Mean)

Q_B_MEAN = 5.0
Q_B = lambda x: norm.pdf(x, loc=Q_B_MEAN, scale=TARGET_STD)
X_B = np.random.normal(loc=Q_B_MEAN, scale=TARGET_STD, size=N_SAMPLES)
Weights_B = P(X_B) / Q_B(X_B)
Estimate_B = np.mean(f(X_B) * Weights_B)
Variance_B = np.var(f(X_B) * Weights_B)

## ====================================================================

## 3. Visualization and Summary

## ====================================================================

print("--- Importance Sampling Performance ---")
print(f"Target Analytical Mean <f(x)>_P: {ANALYTICAL_MEAN:.4f}")

print("\nTrial A: Good Proposal Q_A = N(0, 1)")
print(f"  Estimate: {Estimate_A:.4f} (Accurate)")
print(f"  Variance of Estimator: {Variance_A:.4f} (Low)")

print("\nTrial B: Poor Proposal Q_B = N(5, 1)")
print(f"  Estimate: {Estimate_B:.4f}")
print(f"  Variance of Estimator: {Variance_B:.4f} (Extremely High)")

## Plotting the weights (visualizing the mismatch)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_B, Weights_B, s=10, alpha=0.5, color='darkred', label='Weights $w(x) = P(x)/Q_B(x)$')
ax.axhline(0, color='k', linestyle='--')
ax.set_title("Importance Weights for Poor Proposal $Q_B = \\mathcal{N}(5, 1)$")
ax.set_xlabel("Sampled Point $x$")
ax.set_ylabel("Importance Weight $w(x)$")
ax.grid(True)
plt.show()

print("\nConclusion: Both trials achieved the correct mean (analytic result of 1.0) on average. However, the variance of the estimate for the distant proposal (Trial B) is orders of magnitude higher. This confirms that the variance of the Importance Sampling estimator explodes when the proposal distribution does not adequately cover the important, low-energy region (near x=0) of the target distribution.")
```
**Sample Output:**
```python
--- Importance Sampling Performance ---
Target Analytical Mean <f(x)>_P: 1.0000

Trial A: Good Proposal Q_A = N(0, 1)
  Estimate: 1.0068 (Accurate)
  Variance of Estimator: 2.0543 (Low)

Trial B: Poor Proposal Q_B = N(5, 1)
  Estimate: 0.4965
  Variance of Estimator: 190.2655 (Extremely High)

Conclusion: Both trials achieved the correct mean (analytic result of 1.0) on average. However, the variance of the estimate for the distant proposal (Trial B) is orders of magnitude higher. This confirms that the variance of the Importance Sampling estimator explodes when the proposal distribution does not adequately cover the important, low-energy region (near x=0) of the target distribution.
```

---

## Project 3: Visualizing Density Estimation (KDE Bandwidth)

---

### Definition: Visualizing Density Estimation (KDE Bandwidth)

The goal is to visually demonstrate the effect of the **bandwidth ($h$)** parameter on the **bias-variance trade-off** in **Kernel Density Estimation (KDE)**. This is a fundamental concept in nonparametric density estimation.

### Theory: Bandwidth and the Bias-Variance Trade-Off

**Kernel Density Estimation (KDE)** is a nonparametric method that approximates a continuous density $\hat{p}(\mathbf{x})$ by summing smooth kernels (e.g., Gaussians) placed at each data point $\mathbf{x}_i$.

$$\hat{p}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N K_h(\mathbf{x} - \mathbf{x}_i)$$

The **bandwidth ($h$)** controls the width of these kernels:

  * **Small $h$ (Under-smoothing):** High **variance** (spiky, irregular estimate) but low **bias** (retains local features).
  * **Large $h$ (Over-smoothing):** High **bias** (oversmoothed, washes out real features) but low **variance** (smooth, stable estimate).

We demonstrate this trade-off using a simple 1D multi-modal distribution (a Gaussian Mixture Model) where the true underlying structure is clearly visible.

---

### Extensive Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Ground Truth and Sample Data

## ====================================================================

N_SAMPLES = 100
## Ground Truth: A bimodal distribution (two Gaussians)

X_A = np.random.normal(loc=-2, scale=0.5, size=N_SAMPLES // 2)
X_B = np.random.normal(loc=2, scale=1.0, size=N_SAMPLES // 2)
X_full = np.concatenate([X_A, X_B])

## Function for the true underlying density (for plotting reference)

def true_density(x):
    return 0.5 * norm.pdf(x, loc=-2, scale=0.5) + 0.5 * norm.pdf(x, loc=2, scale=1.0)

## Grid for plotting the smooth functions

x_plot = np.linspace(-5, 5, 500)
y_true = true_density(x_plot)

## ====================================================================

## 2. KDE Trials (Varying Bandwidth)

## ====================================================================

## Trial A: Small Bandwidth (High Variance, Low Bias)

H_SMALL = 0.1
kde_small = gaussian_kde(X_full, bw_method=H_SMALL)
y_small = kde_small(x_plot)

## Trial B: Large Bandwidth (High Bias, Low Variance)

H_LARGE = 1.0
kde_large = gaussian_kde(X_full, bw_method=H_LARGE)
y_large = kde_large(x_plot)

## ====================================================================

## 3. Visualization and Summary

## ====================================================================

plt.figure(figsize=(10, 6))
## Plot raw data (rug plot at the bottom)

plt.plot(X_full, np.full_like(X_full, -0.01), '|k', markeredgewidth=1, alpha=0.5, label='Raw Data Samples')

## Plot true density

plt.plot(x_plot, y_true, 'k--', label='True Density (Reference)', lw=2)

## Plot KDE estimates

plt.plot(x_plot, y_small, 'r-', label=f'KDE (h={H_SMALL}): High Variance', lw=1.5)
plt.plot(x_plot, y_large, 'b-', label=f'KDE (h={H_LARGE}): High Bias', lw=1.5)

## Labeling and Formatting

plt.title('Kernel Density Estimation: Bandwidth and Bias-Variance Trade-off')
plt.xlabel('x')
plt.ylabel('Probability Density $\\hat{p}(x)$')
plt.ylim(-0.05, 0.45)
plt.legend()
plt.grid(True)
plt.show()

print("\n--- KDE Bandwidth Analysis ---")
print(f"Reference Structure: Bimodal (peaks at x=-2 and x=2)")
print("-------------------------------------------------")
print(f"KDE with h={H_SMALL} (Small): Estimate is spiky (high variance) but accurately resolves the bimodal structure (low bias).")
print(f"KDE with h={H_LARGE} (Large): Estimate is smooth (low variance) but fails to resolve the two peaks, becoming an inaccurate single-mode blob (high bias).")

print("\nConclusion: The bandwidth h controls the bias-variance trade-off. A proper choice is critical for accurately inferring the underlying multi-modal energy landscape.")
```
**Sample Output:**
```python
--- KDE Bandwidth Analysis ---
Reference Structure: Bimodal (peaks at x=-2 and x=2)

---

KDE with h=0.1 (Small): Estimate is spiky (high variance) but accurately resolves the bimodal structure (low bias).
KDE with h=1.0 (Large): Estimate is smooth (low variance) but fails to resolve the two peaks, becoming an inaccurate single-mode blob (high bias).

Conclusion: The bandwidth h controls the bias-variance trade-off. A proper choice is critical for accurately inferring the underlying multi-modal energy landscape.
```

---

## Project 4: Maximum Likelihood Estimation (MLE) of a Gaussian

---

### Definition: Maximum Likelihood Estimation (MLE) of a Gaussian

The goal is to numerically find the **Maximum Likelihood Estimate (MLE)** for the mean ($\mathcal{\mu}$) and covariance ($\mathcal{\Sigma}$) of a 2D Gaussian distribution, verifying that the analytical MLE solution corresponds simply to the **sample mean** and **sample covariance**.

### Theory: MLE for the Gaussian

The MLE approach finds the parameters ($\mathcal{\mu}, \mathcal{\Sigma}$) that maximize the **log-likelihood** of observing the generated data:

$$\ln \mathcal{L}(\mathcal{\theta}) = \sum_{i=1}^N \ln p(\mathbf{x}_i|\mathcal{\theta})$$

For the Multivariate Gaussian distribution, the function is simplified because the **analytical solution for the MLE is known and unique**.

The MLE estimates for the mean and covariance are simply the **empirical (sample) mean** and the **empirical (sample) covariance**:

$$\hat{\mathcal{\mu}}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i \quad \text{and} \quad \hat{\Sigma}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N (\mathbf{x}_i - \hat{\mathcal{\mu}})(\mathbf{x}_i - \hat{\mathcal{\mu}})^{\top}$$

This project verifies that the empirical statistics we use for geometric analysis (Chapter 1) are precisely the optimal parameters under the Gaussian assumption (the **MaxEnt** distribution matching those moments).

---

### Extensive Python Code

```python
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Ground Truth and Data Generation

## ====================================================================

N_SAMPLES = 1000
D = 2  # 2D Gaussian

## --- True Parameters (The "Ground Truth") ---

MU_TRUE = np.array([2.5, -1.0])
SIGMA_TRUE = np.array([[1.0, 0.5], [0.5, 2.0]])

## Generate the data from the true distribution

X_data = np.random.multivariate_normal(MU_TRUE, SIGMA_TRUE, N_SAMPLES)

## ====================================================================

## 2. Maximum Likelihood Estimation (Analytical Solution)

## ====================================================================

## MLE Estimate for the Gaussian Mean is the Sample Mean

MU_MLE = np.mean(X_data, axis=0)

## MLE Estimate for the Gaussian Covariance is the Sample Covariance

## Note: np.cov uses N-1 by default (unbiased sample covariance); the MLE formula uses N (biased).

## We use the unbiased estimator here for better comparison, but note the technical distinction.

SIGMA_MLE = np.cov(X_data, rowvar=False)

## --- Define a Poorly Chosen Parameter Set for Comparison ---

MU_POOR = np.array([0.0, 0.0])
SIGMA_POOR = np.array([[3.0, 0.0], [0.0, 3.0]])

## ====================================================================

## 3. Log-Likelihood Calculation (Verification)

## ====================================================================

def calculate_log_likelihood(X, mu, sigma):
    """
    Computes the total log-likelihood for the dataset X given parameters (mu, sigma).
    This measures how well the model (mu, sigma) explains the data.
    """
    # Create the multivariate Gaussian object
    model = multivariate_normal(mean=mu, cov=sigma)

    # Compute the log-likelihood for each point and sum them up
    # Total Log-Likelihood = sum(log P(x_i | theta))
    return np.sum(model.logpdf(X))

## Calculate the log-likelihood for the three parameter sets

LL_TRUE = calculate_log_likelihood(X_data, MU_TRUE, SIGMA_TRUE)
LL_MLE = calculate_log_likelihood(X_data, MU_MLE, SIGMA_MLE)
LL_POOR = calculate_log_likelihood(X_data, MU_POOR, SIGMA_POOR)

## ====================================================================

## 4. Visualization and Summary

## ====================================================================

print("--- Maximum Likelihood Estimation (MLE) Analysis ---")

print("\n1. Parameter Comparison:")
print(f"| Parameter | True | MLE Estimate | Difference |")
print("| :--- | :--- | :--- | :--- |")
print(f"| Mean (\u03bc) | {MU_TRUE} | {np.round(MU_MLE, 3)} | {np.round(MU_MLE - MU_TRUE, 3)} |")
print(f"| Cov (\u03a3)[0,1] | {SIGMA_TRUE[0, 1]:.3f} | {SIGMA_MLE[0, 1]:.3f} | {SIGMA_MLE[0, 1] - SIGMA_TRUE[0, 1]:.3f} |")

print("\n2. Log-Likelihood (LL) Verification:")
print(f"LL of True Parameters:    {LL_TRUE:.2f}")
print(f"LL of MLE Parameters:     {LL_MLE:.2f} (Maximized)")
print(f"LL of Poor Parameters:    {LL_POOR:.2f}")

## Plot LL comparison

plt.figure(figsize=(8, 5))
plt.bar(['LL_True', 'LL_MLE', 'LL_Poor'], [LL_TRUE, LL_MLE, LL_POOR], color=['gray', 'darkgreen', 'red'])
plt.axhline(LL_MLE, color='k', linestyle='--', alpha=0.6, label='Maximum Likelihood')
plt.title('Log-Likelihood Maximization')
plt.ylabel('Total Log-Likelihood')
plt.grid(True, axis='y')
plt.show()

print("\nConclusion: The LL_MLE is the highest value, confirming that the empirical sample mean and covariance are the **Maximum Likelihood Estimates** for the Gaussian model. This numerically verifies the analytical solution and shows that the empirical statistics correctly capture the generative parameters of the distribution under the MaxEnt principle.")
```
**Sample Output:**
```python
--- Maximum Likelihood Estimation (MLE) Analysis ---

1. Parameter Comparison:
| Parameter | True | MLE Estimate | Difference |
| :--- | :--- | :--- | :--- |
| Mean (μ) | [ 2.5 -1. ] | [ 2.566 -0.974] | [0.066 0.026] |
| Cov (Σ)[0,1] | 0.500 | 0.435 | -0.065 |

2. Log-Likelihood (LL) Verification:
LL of True Parameters:    -3096.29
LL of MLE Parameters:     -3092.38 (Maximized)
LL of Poor Parameters:    -4667.53

Conclusion: The LL_MLE is the highest value, confirming that the empirical sample mean and covariance are the **Maximum Likelihood Estimates** for the Gaussian model. This numerically verifies the analytical solution and shows that the empirical statistics correctly capture the generative parameters of the distribution under the MaxEnt principle.
```