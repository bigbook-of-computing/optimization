# Source: Optimization/chapter-5/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

# Define the noisy gradient function
def grad(theta):
    # True gradient is 2.0 * theta
    # We add Gaussian noise to simulate the stochasticity of a mini-batch
    noise = np.random.randn() * 0.2
    return 2.0 * theta + noise

# Set optimization hyperparameters
eta = 0.05       # Learning rate
theta = 5.0      # Initial parameter (starting position)
trajectory = [theta] # List to store the history of theta

# Run the SGD optimization
for t in range(100):
    theta = theta - eta * grad(theta) # The SGD update rule
    trajectory.append(theta)

# Plot the trajectory over time
plt.figure(figsize=(8, 5))
plt.plot(trajectory)
plt.title('Stochastic Gradient Descent on Noisy Quadratic')
plt.xlabel('Iteration (t)')
plt.ylabel(r'Parameter Value ($\theta$)')
plt.axhline(0, color='r', linestyle='--', label='True Minimum')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
