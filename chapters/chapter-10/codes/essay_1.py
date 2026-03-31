# Source: Optimization/chapter-10/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# Synthetic regression data: y = 3x + 5 + noise
x = np.linspace(0, 10, 50)
y = 3*x + 5 + np.random.randn(50)*2
# Fit the linear model
model = LinearRegression().fit(x.reshape(-1,1), y)

plt.figure(figsize=(9, 4))
plt.scatter(x, y, label='Data Points')
# Plot the predicted line (the orthogonal projection, Section 10.3)
plt.plot(x, model.predict(x.reshape(-1,1)), color='r', lw=2, label='Linear Fit')
plt.title('Linear Regression Fit (Minimizing Squared Error)')
plt.xlabel('Input Feature (x)')
plt.ylabel('Target Output (y)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
