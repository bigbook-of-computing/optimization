# Source: Optimization/chapter-9/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

alpha, beta_ = 2, 2  # prior (Beta(2, 2) is a weak, symmetric prior)
n, k = 20, 12        # data: 20 tosses, 12 heads (Empirical bias = 0.6)

theta = np.linspace(0, 1, 200)
prior = beta.pdf(theta, alpha, beta_)
# Posterior is Beta(alpha + k, beta + n - k)
posterior = beta.pdf(theta, alpha + k, beta_ + n - k)

plt.figure(figsize=(9, 6))
plt.plot(theta, prior, '--', label='Prior: Beta(2, 2)', color='gray')
plt.plot(theta, posterior, label=f'Posterior: Beta({alpha+k}, {beta_+n-k})', color='darkorange', lw=2)
plt.axvline(k/n, color='darkgreen', linestyle=':', label=f'Empirical Freq. ({k/n})')

plt.title('Bayesian Update for Coin Bias')
plt.xlabel(r'Coin Bias ($\theta$)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
