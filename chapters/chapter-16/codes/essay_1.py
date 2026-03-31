# Source: Optimization/chapter-16/essay.md -- Block 1

import torch, torch.nn as nn

# --- 1. Define the Neural Network (The Trial Solution, u_theta) ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # The network takes two inputs (x, t) and outputs one scalar (u)
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),  # Layer 1
            nn.Linear(64, 64), nn.Tanh(),  # Layer 2
            nn.Linear(64, 1)  # Output layer: The field solution u(x, t)
        )
    def forward(self, x, t):
        # Concatenate x and t to form a single input vector
        return self.net(torch.cat([x, t], dim=1))

# --- 2. Initialize Model and Collocation Points ---
model = PINN()

# Define input coordinates (collocation points) for calculating the PDE residual
# We need to set 'requires_grad=True' so that AD can compute derivatives w.r.t. these inputs
x = torch.rand(100, 1, requires_grad=True)
t = torch.rand(100, 1, requires_grad=True)

# --- 3. Forward Pass (Compute the solution u) ---
u = model(x, t)

# --- 4. Automatic Differentiation (AD) for Derivatives ---
# We use torch.autograd.grad to compute partial derivatives w.r.t. x and t.
# create_graph=True is necessary to compute SECOND derivatives.

# First derivative w.r.t time (u_t)
u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

# First derivative w.r.t space (u_x)
u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

# Second derivative w.r.t. space (u_xx) - AD applied to the result of u_x
u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

# --- 5. Construct the Physics Residual ---
# Heat Equation: u_t - alpha * u_xx = 0. Here, alpha is set to 0.1
alpha = 0.1
residual = u_t - alpha * u_xx

# --- 6. Physics Loss Term ---
# The physics loss L_phys is the Mean Squared Error (MSE) of the residual
loss_phys = (residual**2).mean()
