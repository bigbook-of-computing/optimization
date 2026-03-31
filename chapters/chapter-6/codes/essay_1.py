# Source: Optimization/chapter-6/essay.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Loss Gradient (Ravine Function) ---
# L(x, y) = 0.5*x^2 + 5*y^2
# Gradient is: [dL/dx, dL/dy] = [x, 10*y]
def grad(theta):
    # theta is a 2D vector [x, y]
    x, y = theta
    return np.array([x, 10*y])

# --- 2. Helper Function to Run Optimization ---
def run_optimizer(update_func, theta0=np.array([3.0, 3.0]), steps=100):
    """Runs a given optimization update function for a set number of steps."""
    traj = [theta0.copy()]
    theta = theta0.copy()
    for _ in range(steps):
        # The update function is responsible for calculating the next theta
        theta = update_func(theta)
        traj.append(theta.copy())
    return np.array(traj)

# --- 3. Optimization Algorithms ---

eta = 0.05 # Learning rate, compromised by the stiff direction (y/theta_2)

# 3.1. Gradient Descent (GD)
def gd_update(theta):
    """Standard Gradient Descent: theta_t+1 = theta_t - eta * grad(theta_t)"""
    return theta - eta * grad(theta)

# 3.2. Momentum
v = np.zeros(2) # Global state: velocity vector
beta = 0.9      # Momentum coefficient (inertia)
def momentum_update(theta):
    """Momentum: v_t+1 = beta*v_t - eta*grad, theta_t+1 = theta_t + v_t+1"""
    # NOTE: The global 'v' must be updated within the function's closure context
    global v
    v = beta * v - eta * grad(theta)
    return theta + v

# 3.3. Adam (Simplified)
m = np.zeros(2)  # First moment estimate (momentum)
v2 = np.zeros(2) # Second moment estimate (squared gradient history)
b1, b2 = 0.9, 0.999
eps = 1e-8
# Use a mutable list to track time step t for bias correction (t is passed as [1] initially)
def adam_update(theta, t_counter=[1]):
    """Adam: Combines momentum (m) and adaptive scaling (v2) with bias correction."""
    global m, v2
    g = grad(theta)
    t = t_counter[0] # Current time step

    # 1. Update biased moment estimates
    m = b1 * m + (1-b1)*g
    v2 = b2 * v2 + (1-b2)*(g*g)

    # 2. Bias correction (required especially early on)
    m_hat = m / (1-b1**t)
    v_hat = v2 / (1-b2**t)

    # 3. Final adaptive update
    theta -= eta * m_hat / (np.sqrt(v_hat)+eps)
    t_counter[0] += 1 # Increment time step
    return theta

# --- 4. Run Trajectories ---
# Reset global state variables before each run
traj_gd = run_optimizer(gd_update)
v[:] = 0 # Reset momentum velocity state
traj_m = run_optimizer(momentum_update)
m[:] = 0; v2[:] = 0 # Reset Adam moment states
traj_a = run_optimizer(adam_update)

# --- 5. Visualization ---
# Plotting the loss contours
t1 = np.linspace(-4, 4, 100)
t2 = np.linspace(-4, 4, 100)
T1, T2 = np.meshgrid(t1, t2)
L_loss = 0.5 * T1**2 + 5 * T2**2

plt.figure(figsize=(9, 7))
CS = plt.contour(T1, T2, L_loss, levels=np.logspace(-1, 3, 20), cmap='magma')
plt.plot(traj_gd[:,0], traj_gd[:,1], '-o', label='GD (Zigzagging)', alpha=0.8, markersize=3, lw=1.5, color='royalblue')
plt.plot(traj_m[:,0], traj_m[:,1], '-o', label='Momentum (Coasting)', alpha=0.8, markersize=3, lw=1.5, color='darkorange')
plt.plot(traj_a[:,0], traj_a[:,1], '-o', label='Adam (Adaptive)', alpha=0.8, markersize=3, lw=1.5, color='mediumseagreen')

plt.scatter(0, 0, marker='*', s=300, color='gold', label='Minimum')
plt.scatter(traj_gd[0,0], traj_gd[0,1], marker='s', s=80, color='black', label='Start')

plt.title('Optimization Trajectories on a Ravine Function')
plt.xlabel(r'Parameter $\theta_1$ (Sloppy/Flat Direction)')
plt.ylabel(r'Parameter $\theta_2$ (Stiff/Steep Direction)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
