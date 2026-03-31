# Source: Optimization/chapter-14/essay.md -- Block 1

import torch, matplotlib.pyplot as plt
import numpy as np # Used for plotting grid setup

# --- 1. Synthetic Data (Two Metastable States) ---
# Creates two Gaussian clusters centered at (-2, 0) and (2, 0)
# These represent two high-probability, low-energy regions (basins).
data = torch.cat([
    torch.randn(1000,2)*0.3 + torch.tensor([2,0]),
    torch.randn(1000,2)*0.3 + torch.tensor([-2,0])
])

# --- 2. Energy Function Definition E(x, w) ---
# E is parameterized by 'w' (our neural network weights)
E = lambda x, w: 0.5*(x**2).sum(1) + w[0]*torch.sin(1.2*x[:,0]) + w[1]*torch.cos(1.2*x[:,1])

w = torch.randn(2, requires_grad=True)
opt = torch.optim.Adam([w], lr=0.05)

# --- 3. Training Loop (Contrastive Divergence / Gradient Approximation) ---
# Minimizes the difference between the average energy of real data and fake data.
for _ in range(500):
    # a) Positive Phase (Data Term): Calculate average energy of real samples
    Eb = E(data, w).mean()
    
    # b) Negative Phase (Model Term): Sample pure noise (an approximation)
    # This sample represents non-data configurations that should have high energy.
    x_fake = torch.randn_like(data)
    Em = E(x_fake, w).mean()
    
    # Loss: Minimize E_data - E_model. This pulls E(data) down and pushes E(fake) up.
    loss = Eb - Em 
    
    # Optimization step
    opt.zero_grad(); loss.backward(); opt.step()

# --- 4. Visualization: Plotting the Learned Landscape ---
X_grid, Y_grid = torch.meshgrid(torch.linspace(-4,4,100), torch.linspace(-4,4,100), indexing='xy')
# Evaluate the learned energy function across the entire 2D grid
Z = E(torch.stack([X_grid.flatten(), Y_grid.flatten()],1), w).detach().reshape(100,100)

plt.figure(figsize=(9, 7))
# Plot the energy contours (low energy is dark, high energy is light)
plt.contourf(X_grid.numpy(), Y_grid.numpy(), Z.numpy(), levels=50, cmap='magma')
plt.colorbar(label=r'Learned Energy $E_{\mathbf{w}}(\mathbf{x})$')
# Overlay the original data points
plt.scatter(data[:,0].numpy(), data[:,1].numpy(), s=2, c='cyan', alpha=0.5, label='Original Data')

plt.title('Learned Energy Landscape via EBM Training')
plt.xlabel(r'$x_1$'); plt.ylabel(r'$x_2$')
plt.legend()
plt.show()
