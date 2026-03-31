# Source: Optimization/chapter-6/codebook.md -- Block 1

import numpy as np
import random
import matplotlib.pyplot as plt

# ====================================================================
# 1. Ising Core Functions (from Chapter 2)
# ====================================================================

# Standard 2D Ising Lattice functions with Periodic Boundary Conditions (PBCs)

def create_lattice(N, initial_state='random'):
    """Initializes an N x N lattice with spins (+1 or -1)."""
    if initial_state == '+1':
        return np.ones((N, N), dtype=np.int8)
    return np.random.choice([-1, 1], size=(N, N), dtype=np.int8)

def get_neighbors(N, i, j):
    """Returns the coordinates of the four nearest neighbors (PBCs)."""
    return [
        ((i + 1) % N, j), ((i - 1 + N) % N, j), 
        (i, (j + 1) % N), (i, (j - 1 + N) % N)  
    ]

def calculate_delta_E_local(lattice, i, j, J=1.0, H=0.0):
    """Computes the O(1) change in energy for flipping spin (i, j)."""
    N = lattice.shape[0]
    spin_ij = lattice[i, j]
    sum_nn = sum(lattice[ni, nj] for ni, nj in get_neighbors(N, i, j))
    
    # Delta E = 2J * spin_ij * sum_nn + 2H * spin_ij
    return 2 * J * spin_ij * sum_nn + 2 * H * spin_ij

def metropolis_update_step(lattice, beta, J=1.0, H=0.0):
    """Performs one Monte Carlo Sweep (N*N attempts) using single-spin flip."""
    N = lattice.shape[0]
    total_spins = N * N
    
    for _ in range(total_spins):
        i, j = random.randrange(N), random.randrange(N)
        delta_E = calculate_delta_E_local(lattice, i, j, J, H)
        
        if delta_E <= 0 or random.random() < np.exp(-beta * delta_E):
            lattice[i, j] *= -1
            
def calculate_magnetization(lattice):
    """Calculates the absolute magnetization per spin |M|."""
    return np.mean(np.abs(lattice))

# ====================================================================
# 2. Autocorrelation Analysis Functions
# ====================================================================

def autocorr_func(x, lag):
    """Calculates the Autocorrelation Function C(tau)."""
    N = len(x)
    mean_x = np.mean(x)
    var_x = np.var(x)

    if var_x == 0:
        return 1.0 if lag == 0 else 0.0

    # Cov(O_t, O_{t+tau})
    cov = np.sum((x[:N - lag] - mean_x) * (x[lag:] - mean_x)) / (N - lag)
    return cov / var_x

def estimate_tau_int(x, max_lag_limit=300):
    """Estimates the integrated autocorrelation time (tau_int) from C(tau)."""
    max_lag = min(max_lag_limit, len(x) // 2)
    C = [autocorr_func(x, lag) for lag in range(max_lag + 1)]

    # ESS Denominator (G) = 1 + 2 * sum(C(tau)) with a practical cutoff
    G = 1.0
    for c_tau in C[1:]:
        # Stop summing when C(tau) is small or negative (practical cutoff)
        if c_tau < 0.05:
            G += 2 * c_tau
            break
        G += 2 * c_tau

    tau_int = 0.5 if G <= 1.0 else (G - 1.0) / 2.0
    return tau_int, C

# ====================================================================
# 3. Simulation and Quantification
# ====================================================================

LATTICE_SIZE = 32
MCS_RUN = 15000  # Long run to observe slow dynamics
EQUILIBRATION_MCS = 500

# Critical inverse temperature: beta_c = ln(1 + sqrt(2)) / 2 approx 0.4407
T_C = 2.269185 
BETA_C = 1.0 / T_C

# Temperatures of Interest (Control Parameters)
TEMPS = {
    'T_low (Ordered)': 1.5,
    'T_c (Critical)': T_C,
    'T_high (Disordered)': 3.5
}

J = 1.0
H = 0.0
results = {}

print(f"Quantifying critical slowing down for L={LATTICE_SIZE}...")

for label, T in TEMPS.items():
    beta = 1.0 / T
    
    # Initialize lattice to all up (+1)
    lattice = create_lattice(LATTICE_SIZE, initial_state='+1')
    
    # Thermalization
    for _ in range(EQUILIBRATION_MCS):
        metropolis_update_step(lattice, beta, J, H)
        
    # Measurement
    M_series = []
    for _ in range(MCS_RUN):
        metropolis_update_step(lattice, beta, J, H)
        M_series.append(calculate_magnetization(lattice))
    
    # Analysis
    M_series = np.array(M_series)
    tau_int, C_plot = estimate_tau_int(M_series)
    
    results[label] = {
        'T': T,
        'M_series': M_series,
        'C_plot': C_plot,
        'tau_int': tau_int
    }
    print(f"Finished {label}. Tau_int: {tau_int:.2f} MCS.")

# ====================================================================
# 4. Visualization and Comparison
# ====================================================================

# Extract results for comparison plot
tau_values = [res['tau_int'] for res in results.values()]
labels = list(results.keys())

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
markers = ['o', 's', '^']

# Plot 1: Autocorrelation Function C_M(tau) for each T
for i, (label, res) in enumerate(results.items()):
    ax[0].plot(res['C_plot'][:100], marker='.', markersize=3, 
               linestyle='-', lw=1.5, alpha=0.8,
               label=f"{label} ($\\\\tau_{{\\\\text{{int}}}} \\\\approx {res['tau_int']:.1f}$ MCS)")

ax[0].axhline(0, color='k', linestyle='--')
ax[0].set_title('Autocorrelation of Magnetization $C_M(\\tau)$')
ax[0].set_xlabel('Time Lag $\\tau$ (MCS)')
ax[0].set_ylabel('Autocorrelation $C(\\tau)$')
ax[0].set_xlim(0, 50)
ax[0].legend()
ax[0].grid(True, which='both', linestyle=':')

# Plot 2: Autocorrelation Time Comparison
ax[1].bar(labels, tau_values, color=['skyblue', 'darkred', 'lightgreen'])
ax[1].set_title('Integrated Autocorrelation Time $\\tau_{\\text{int}}$')
ax[1].set_xlabel('Temperature Regime')
ax[1].set_ylabel('$\\tau_{\\text{int}}$ (MCS)')
ax[1].grid(True, which='major', axis='y', linestyle=':')

plt.tight_layout()
plt.show()

# Final Analysis
print("\n--- Critical Slowing Down Analysis ---")
for label, res in results.items():
    print(f"| {label:<20} | T={res['T']:.3f} | Tau_int: {res['tau_int']:.2f} MCS |")
print("-" * 50)
print(f"Conclusion: The autocorrelation time $\\tau_{{\\text{{int}}}}$ is highest at the critical temperature ($T_c \u2248 2.269$). This confirms that the single-spin Metropolis algorithm suffers from **critical slowing down**, requiring significantly more Monte Carlo sweeps (MCS) to generate independent samples when the system is near its phase transition boundary.")
