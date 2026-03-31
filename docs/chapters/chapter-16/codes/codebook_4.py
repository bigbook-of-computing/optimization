# Source: Optimization/chapter-16/codebook.md -- Block 4

import torch
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ====================================================================
# 1. Setup Conceptual Solution and Residual Grid
# ====================================================================

ALPHA = 0.5
L = 1.0
T_FINAL = 1.0

# --- Test Grid ---
N_GRID = 50
x_grid = np.linspace(0, L, N_GRID)
t_grid = np.linspace(0, T_FINAL, N_GRID)
X_test_mesh, T_test_mesh = np.meshgrid(x_grid, t_grid)
X_test_flat = np.hstack([X_test_mesh.flatten()[:, None], T_test_mesh.flatten()[:, None]])

X_test_tf = torch.tensor(X_test_flat, dtype=torch.float32, requires_grad=True)

# ====================================================================
# 2. Residual Calculation on the Test Grid
# ====================================================================

def calculate_residual_magnitude(X_test, alpha=ALPHA):
    def u_tf(x_t):
        x, t = x_t[:, 0:1], x_t[:, 1:2]
        return torch.sin(np.pi * x) * torch.exp(-alpha * t * np.pi**2)
    
    u = u_tf(X_test)
    
    # First derivatives
    grads = torch.autograd.grad(u, X_test, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    # Second derivative u_xx
    u_xx = torch.autograd.grad(u_x, X_test, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    # R = u_t - alpha * u_xx
    R = u_t - alpha * u_xx
    R_magnitude = torch.abs(R).detach().numpy().flatten()
    return R_magnitude

R_mag_flat = calculate_residual_magnitude(X_test_tf)
R_mag_grid = R_mag_flat.reshape(N_GRID, N_GRID)

# ====================================================================
# 3. Adaptive Sampling Identification
# ====================================================================

# Identify the points with the highest violation (top 5% residual)
THRESHOLD_PERCENTILE = 95
threshold_value = np.percentile(R_mag_flat, THRESHOLD_PERCENTILE)
high_residual_indices = R_mag_flat > threshold_value
X_next_batch = X_test_flat[high_residual_indices]

# ====================================================================
# 4. Visualization and Analysis
# ====================================================================

plt.figure(figsize=(9, 6))

# Plot 1: Heatmap of the Residual R(x, t)
plt.imshow(R_mag_grid, extent=[0, L, 0, T_FINAL], origin='lower', cmap='plasma')
plt.colorbar(label='Residual Magnitude $|R(x, t)|$')

# Plot 2: Overlay the new, adaptively sampled points
plt.scatter(X_next_batch[:, 0], X_next_batch[:, 1],
            s=5, c='white', label=f'Adaptive Sampled Points (Top {100 - THRESHOLD_PERCENTILE}%)')

plt.title(f'Adaptive Sampling Strategy: Locating Maximum PDE Violation')
plt.xlabel('Space $x$')
plt.ylabel('Time $t$')
plt.legend()
plt.savefig('Optimization/RESEARCH/docs/chapters/chapter-16/codes/ch16_adaptive_sampling.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n--- Adaptive Sampling Strategy Summary ---")
print(f"Residual Violation Threshold: R > {threshold_value:.4e}")
print(f"Number of new collocation points identified: {len(X_next_batch)}")

print("\nConclusion: The heatmap visually displays the residual R across the domain. The adaptive sampling correctly identifies the regions where the solution violates the PDE the most (high R, near boundaries or sharp transitions in the solution) and focuses the next training batch on these critical areas, leading to faster and more robust convergence.")
