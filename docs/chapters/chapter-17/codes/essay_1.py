# Source: Optimization/chapter-17/essay.md -- Block 1

import torch, torch.nn as nn

class NQS(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        # W: Coupling matrix between visible (spins) and hidden (entanglement) units.
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden)*0.1)
        # a, b: Bias fields for visible and hidden units, respectively.
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))
        
    def forward(self, s):
        # Implements the RBM wavefunction ansatz (Section 17.3):
        # psi(s) = sum_h exp(a^T s + b^T h + s^T W h)
        
        # 1. Linear term: a^T s
        linear_term = self.a @ s 
        
        # 2. Coupling term: b + s @ W 
        # This is the contribution to the hidden unit energies.
        coupling_term = self.b + s @ self.W
        
        # 3. Summation over hidden units (logsumexp handles the sum efficiently):
        # The exponential of the RBM energy involves a sum over all possible hidden states h.
        log_sum_exp = torch.logsumexp(coupling_term, dim=1)
        
        # 4. Final log-amplitude (unnormalized log|psi| or log psi)
        return torch.exp(linear_term + log_sum_exp)

# Example energy estimation (schematic)
# Define 100 sample spin configurations (s) for a system of 10 spins.
s = torch.randint(0,2,(100,10)).float()*2-1
psi = NQS(10,5) # N=10 spins, 5 hidden units
log_psi = psi(s)
# Gradient-based updates on <H> would follow with sampled local energies
