# Source: Optimization/chapter-16/codebook.md -- Block 1

import torch
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ====================================================================
# 1. Setup Conceptual PDE (1D Heat Equation)
# ====================================================================

# PDE: u_t = alpha * u_xx (Heat Equation)
ALPHA = 0.5
N_COLLOCATION = 100

# Conceptual Collocation Points (x, t) for training the physics loss
X_COLLOC = torch.rand((N_COLLOCATION, 2), requires_grad=True)

# Conceptual PINN Output (u_theta(x, t)) - Placeholder for the NN
def u_theta(x_t):
    # Simplified placeholder: u(x, t) = sin(\pi x) * exp(-t * \pi^2 * alpha)
    x = x_t[:, 0:1]
    t = x_t[:, 1:2]
    return torch.sin(np.pi * x) * torch.exp(-ALPHA * np.pi**2 * t)

# ====================================================================
# 2. Automatic Differentiation (AD) and Residual Calculation
# ====================================================================

def calculate_physics_loss(X_colloc, alpha=ALPHA):
    # Output of the neural network (u)
    u = u_theta(X_colloc)
    
    # Step 1: Calculate First Derivatives (u_t, u_x)
    # grad_outputs=torch.ones_like(u) is used for vector-valued outputs
    grads = torch.autograd.grad(u, X_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    # Step 2: Calculate Second Derivative (u_xx)
    u_xx = torch.autograd.grad(u_x, X_colloc, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    # Step 3: Compute the Residual (R)
    # R = u_t - alpha * u_xx
    R = u_t - alpha * u_xx
    
    # Step 4: Compute the Physics Loss (L_phys = MSE of R)
    L_phys = torch.mean(R**2)
    
    return L_phys.item()

# --- Final Function Test ---
physics_loss_value = calculate_physics_loss(X_COLLOC)

print("--- Physics Loss Calculation Summary (Conceptual) ---")
print(f"PDE: u_t = {ALPHA} * u_xx")
print(f"Calculated Physics Loss L_phys: {physics_loss_value:.4e} (Should be near zero if NN is accurate)")
