# Source: Optimization/chapter-12/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

# Linearly separable data (100 samples in 2D)
np.random.seed(0)
X = np.random.randn(100, 2)
# True separation: A line defined by the equation 0.8*x1 - 0.5*x2 + 0.2 = 0
y = np.sign(X[:,0]*0.8 + X[:,1]*(-0.5) + 0.2)

w = np.zeros(2) # Initialize weights w = [0, 0]
b = 0           # Initialize bias b = 0
eta = 0.1       # Learning rate

# Training Loop
for epoch in range(20):
    for i in range(len(X)):
        # 1. Forward Pass: Compute the activation score
        activation_score = np.dot(w, X[i]) + b
        # 2. Prediction: Apply the sign function (the non-differentiable step)
        y_pred = np.sign(activation_score)
        
        # 3. Update (Backward Pass): Perceptron Learning Rule
        if y_pred != y[i]:
            # Corrective impulse is applied ONLY on misclassified points
            w += eta * y[i] * X[i] # Adjust weight vector
            b += eta * y[i]        # Adjust bias
            
# Plotting the result
xx, yy = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
# The decision boundary is where the score is zero: w[0]*x + w[1]*y + b = 0
Z = np.sign(w[0]*xx + w[1]*yy + b)

plt.figure(figsize=(9, 6))
# Scatter plot of data points, colored by true class y
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', alpha=0.7)
# Plot the learned decision boundary (contour where Z=0)
plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='-', linewidths=3)
plt.title('Perceptron Decision Boundary')
plt.xlabel('Feature $x_1$')
plt.ylabel('Feature $x_2$')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
