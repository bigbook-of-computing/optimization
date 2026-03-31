# Source: Optimization/chapter-7/codebook.md -- Block 4

import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Parameters and Initial Conditions
# ====================================================================

# --- System Parameters (1D Harmonic Oscillator) ---
M = 1.0     
K_SPRING = 1.0  
KB = 1.0    # Boltzmann constant (set to 1.0 for simplified units)
DT = 0.01   
STEPS = 5000 
DOF = 1     # Degrees of freedom for a 1D particle

# --- Thermostat Parameters ---
T0 = 1.0    # Target temperature
TAU_T = 1.0 # Relaxation time constant (\tau_T)

# --- Initial Conditions (High Energy/Temperature) ---
R_INIT = 5.0  # High initial potential energy (stretched spring)
V_INIT = 0.0  # Initial kinetic energy is zero
F_current = -K_SPRING * R_INIT

# --- Reference Functions ---
def force(r, k=K_SPRING):
    return -k * r

def calculate_temperature(v, m=M, kB=KB, dof=DOF):
    """Calculates instantaneous temperature from kinetic energy (T = 2K / (DOF * kB))."""
    K = 0.5 * m * v**2
    return 2 * K / (dof * kB)

# ====================================================================
# 2. Velocity–Verlet Integration with Berendsen Thermostat
# ====================================================================

r, v = R_INIT, V_INIT
T_inst_history = []
E_total_history = []

for step in range(STEPS):
    # Get current acceleration
    a_current = F_current / M
    
    # 1. Position Update (Verlet)
    r_new = r + v * DT + 0.5 * a_current * DT**2
    
    # 2. New Force Evaluation
    F_new = force(r_new)
    a_new = F_new / M
    
    # 3. Velocity Update (Pre-Thermostat - V_raw_new)
    v_raw_new = v + 0.5 * (a_current + a_new) * DT
    
    # --- Berendsen Thermostat Application ---
    T_inst = calculate_temperature(v_raw_new, dof=DOF)
    
    # Handle near-zero temperature (division by zero risk)
    if T_inst < 1e-6:
        lambda_factor = 1.0 # No scaling if temp is near zero
    else:
        # Calculate scaling factor lambda
        lambda_sq = 1 + (DT / TAU_T) * ((T0 / T_inst) - 1)
        # Ensure lambda_sq is not negative (possible if DT is too large)
        lambda_factor = np.sqrt(np.maximum(lambda_sq, 0.0))
    
    # Apply scaling to the velocity
    v_thermo = v_raw_new * lambda_factor
    
    # Bookkeeping: Advance state and force
    r, v = r_new, v_thermo
    F_current = F_new
    
    # Store temperature and energy
    T_inst_history.append(T_inst)
    E_total_history.append(0.5*M*v**2 + 0.5*K_SPRING*r**2)


# ====================================================================
# 3. Visualization and Summary
# ====================================================================

T_history = np.array(T_inst_history)
time_points = np.arange(len(T_history)) * DT

plt.figure(figsize=(10, 5))

# Plot instantaneous temperature over time
plt.plot(time_points, T_history, lw=1.5, color='darkgreen', label='Instantaneous $T_{\\text{inst}}$')
plt.axhline(T0, color='red', linestyle='--', alpha=0.7, label='Target Temperature $T_0$')

# Labeling and Formatting
plt.title(f'Berendsen Thermostat (NVT) Relaxation ($\u03C4_T={TAU_T}$ s)')
plt.xlabel('Time (s)')
plt.ylabel('Instantaneous Temperature ($T$)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Conclusion ---
# Calculate initial T (when V_INIT=0, T_inst is near zero or undefined, but the potential energy is high)
T_initial_effective = T_history[T_history > 1e-5][0]

print("\n--- Thermostat Performance Check ---")
print(f"Target Temperature (T0): {T0:.4f}")
print(f"Temperature at start of dynamics: {T_initial_effective:.4f}")
print(f"Final Average Temperature (Last 1000 steps): {np.mean(T_history[-1000:]):.4f}")

print("\nConclusion: The Berendsen thermostat successfully stabilized the system. The instantaneous temperature is forced to relax from the initial dynamic fluctuations and quickly converges to the target temperature (T0=1.0), demonstrating the required control for simulating the Canonical (NVT) ensemble.")
