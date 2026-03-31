# Source: Optimization/chapter-16/codebook.md -- Block 3

import torch
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ====================================================================
# 1. Setup Data and True Parameter (The Target)
# ====================================================================

ALPHA_TRUE = 0.5 # The true diffusion constant to be inferred
N_DATA_OBS = 100 # Small number of noisy observation points
N_PHYS = 500

# Conceptual Data Observations (Conceptual u(x, t) data points)
def exact_solution(x, t, alpha=ALPHA_TRUE):
    return np.exp(-alpha * t * np.pi**2) * np.sin(np.pi * x)

# Generate small, noisy dataset D
X_DATA = np.random.rand(N_DATA_OBS, 2)
X_DATA[:, 0] *= 1.0 # x \in [0, 1]
X_DATA[:, 1] *= 0.5 # t \in [0, 0.5]
U_DATA = exact_solution(X_DATA[:, 0:1], X_DATA[:, 1:2]) + np.random.normal(0, 0.05, (N_DATA_OBS, 1))

# --- Trainable Variable Setup ---
# The parameter \alpha is now a trainable variable, starting from a bad guess.
ALPHA_GUESS_INIT = 0.1
alpha_trainable = torch.tensor(ALPHA_GUESS_INIT, requires_grad=True)

# Collocation points for physics loss
X_PHYS_tf = torch.rand((N_PHYS, 2), requires_grad=True)

# --- Conceptual Model (NN Placeholder) ---
def u_theta(x_t, alpha):
    # Placeholder for NN output
    x = x_t[:, 0:1]
    t = x_t[:, 1:2]
    return torch.sin(np.pi * x) * torch.exp(-t * alpha * np.pi**2)

# ====================================================================
# 2. Inverse Loss Function (L_total with trainable \alpha)
# ====================================================================

def calculate_inverse_loss():
    X_DATA_tf = torch.tensor(X_DATA, dtype=torch.float32)
    U_DATA_tf = torch.tensor(U_DATA, dtype=torch.float32)
    
    # 1. Data Loss (L_data) - Enforces fit to noisy observations
    U_pred_data = u_theta(X_DATA_tf, alpha_trainable)
    L_data = torch.mean((U_pred_data - U_DATA_tf)**2)
    
    # 2. Physics Loss (L_phys) - Uses the trainable alpha
    u_phys = u_theta(X_PHYS_tf, alpha_trainable)
    
    # First derivatives
    grads = torch.autograd.grad(u_phys, X_PHYS_tf, grad_outputs=torch.ones_like(u_phys), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    # Second derivative u_xx
    u_xx = torch.autograd.grad(u_x, X_PHYS_tf, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    # R = u_t - alpha * u_xx
    R = u_t - alpha_trainable * u_xx
    L_phys = torch.mean(R**2)
    
    # Total Loss (L_total)
    L_total = L_data + L_phys
    
    # The gradient of L_total w.r.t. \alpha is what drives the inference
    L_total.backward()
    dL_dalpha = alpha_trainable.grad
    
    return L_total.item(), alpha_trainable.item(), dL_dalpha.item()

# --- Final Function Test ---
L_total, alpha_inferred, dL_dalpha = calculate_inverse_loss()

print("\n--- Solving the Inverse Problem (Parameter Inference) ---")
print(f"True Diffusion Constant (\u03b1_true): {ALPHA_TRUE:.3f}")
print(f"Initial Guess (\u03b1_guess): {ALPHA_GUESS_INIT:.3f}")
print("---------------------------------------------------------------")
print(f"Total Loss L_total (at \u03b1_guess): {L_total:.4f}")
print(f"Gradient w.r.t. \u03b1 (\u2202L/\u2202\u03b1): {dL_dalpha:.4f}")

print("\nConclusion: The gradient \u2202L/\u2202\u03b1 is non-zero, indicating that the optimization process would successfully adjust the trainable parameter \u03b1 in the direction of the true value (\u03b1_true=0.5) to minimize the total physics-constrained data error.")
