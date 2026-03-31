# Source: Optimization/chapter-16/codebook.md -- Block 2

import torch
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ====================================================================
# 1. Setup Constraints and Loss Components
# ====================================================================

# Constants
ALPHA = 0.5  # Known diffusion constant
L = 1.0      # Domain length
T_FINAL = 1.0

# --- Data Points (Conceptual) ---
N_IC = 50   # Points for Initial Condition
N_BC = 50   # Points for Boundary Conditions
N_PHYS = 500 # Points for Physics/Collocation

# 1. Initial Condition Data (IC): u(x, 0) = sin(\pi x)
X_IC = torch.cat([torch.rand((N_IC, 1)) * L, torch.zeros((N_IC, 1))], dim=1).requires_grad_(True)
U_IC = torch.sin(np.pi * X_IC[:, 0:1])

# 2. Boundary Condition Data (BC): u(0, t) = u(L, t) = 0
X_BC_L = torch.cat([torch.zeros((N_BC, 1)), torch.rand((N_BC, 1)) * T_FINAL], dim=1).requires_grad_(True)
X_BC_R = torch.cat([torch.full((N_BC, 1), L), torch.rand((N_BC, 1)) * T_FINAL], dim=1).requires_grad_(True)
X_BC_tf = torch.cat([X_BC_L, X_BC_R], dim=0)
U_BC = torch.zeros((N_BC * 2, 1))

# 3. Collocation Points (Physics): Random points in the domain
X_PHYS_tf = (torch.rand((N_PHYS, 2)) * torch.tensor([L, T_FINAL])).requires_grad_(True)

# --- Conceptual Model (NN Placeholder) ---
def u_theta(x_t):
    # Use a non-linear function so second derivatives are non-zero and tracked
    x = x_t[:, 0:1]
    t = x_t[:, 1:2]
    return torch.sin(np.pi * x) * torch.exp(-ALPHA * np.pi**2 * t)

# ====================================================================
# 2. PINN Loss Function (Composite Loss)
# ====================================================================

def calculate_total_loss(alpha=ALPHA, w_ic=1.0, w_bc=1.0):
    # A. Physics Loss (L_phys) - Uses AD on collocation points
    u_phys = u_theta(X_PHYS_tf)
    
    # First derivatives
    grads = torch.autograd.grad(u_phys, X_PHYS_tf, grad_outputs=torch.ones_like(u_phys), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    # Second derivative u_xx
    u_xx = torch.autograd.grad(u_x, X_PHYS_tf, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    # R = u_t - alpha * u_xx
    R = u_t - alpha * u_xx
    L_phys = torch.mean(R**2)
    
    # B. Initial Condition Loss (L_IC)
    u_ic_pred = u_theta(X_IC)
    L_IC = torch.mean((u_ic_pred - U_IC)**2)
    
    # C. Boundary Condition Loss (L_BC)
    u_bc_pred = u_theta(X_BC_tf)
    L_BC = torch.mean((u_bc_pred - U_BC)**2)
    
    # Total Loss
    L_total = L_phys + w_ic * L_IC + w_bc * L_BC
    
    return L_total.item(), L_phys.item(), L_IC.item(), L_BC.item()

# --- Final Function Test ---
L_total, L_phys, L_IC, L_BC = calculate_total_loss()

print("\n--- Solving the Forward Problem (Conceptual Multi-Loss) ---")
print(f"Total Loss (L_total): {L_total:.4f}")
print(f"Physics Loss (L_phys): {L_phys:.4f}")
print(f"Initial Condition Loss (L_IC): {L_IC:.4f}")
print(f"Boundary Condition Loss (L_BC): {L_BC:.4f}")
