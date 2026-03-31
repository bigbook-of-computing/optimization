# Source: Optimization/chapter-7/codebook.md -- Block 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Conceptual Trajectory Generation (Simulating a Diffusive System)
# ====================================================================

# We simulate a random walk trajectory as a proxy for a complex MD simulation
N_PARTICLES = 100       
DT = 0.01               
TOTAL_STEPS = 5000      
TRAJECTORY_LENGTH = TOTAL_STEPS + 1
DIMENSIONS = 3          # Use 3D for the 6*tau denominator in Einstein relation

R_history = np.zeros((TRAJECTORY_LENGTH, N_PARTICLES, DIMENSIONS))

# Simulate the diffusion process (random walk)
for t in range(1, TRAJECTORY_LENGTH):
    # Small, random displacement
    random_displacement = np.random.normal(0, 0.1, size=(N_PARTICLES, DIMENSIONS))
    R_history[t] = R_history[t-1] + random_displacement

# ====================================================================
# 2. Mean-Squared Displacement (MSD) Calculation
# ====================================================================

MAX_LAG = TOTAL_STEPS // 2
msd_history = np.zeros(MAX_LAG)

# Calculate MSD by averaging over all time origins (t) and all particles (i)
for tau in range(1, MAX_LAG):
    # Calculate displacement vector: dr(t) = R(t+tau) - R(t)
    dr = R_history[tau:] - R_history[:-tau]
    
    # Squared displacement: sum |dr|^2 over dimensions (axis=2)
    dr_sq = np.sum(dr**2, axis=2)
    
    # Mean: Average over all particles (axis=1) and all time origins (axis=0)
    msd_history[tau] = np.mean(dr_sq)

# Time axis for the MSD plot
time_lags = np.arange(MAX_LAG) * DT

# Identify the linear regime for fitting (long time)
FIT_START_LAG = 500 # Starting the fit after the initial ballistic/sub-diffusive regime

# ====================================================================
# 3. Diffusion Coefficient (D) Extraction
# ====================================================================

# Filter data for linear fitting
X_fit = time_lags[FIT_START_LAG:MAX_LAG:20] # Sample sparsely for clean fitting
Y_fit = msd_history[FIT_START_LAG:MAX_LAG:20]

# Perform linear regression: MSD(tau) = slope*tau + C
slope, intercept, r_value, p_value, std_err = linregress(X_fit, Y_fit)

# Extract Diffusion Coefficient D from the slope (D = slope / (2 * DIMENSIONS))
D_CALCULATED = slope / (2 * DIMENSIONS) 

# Create the best-fit line data for visualization
fit_line = intercept + slope * X_fit

# ====================================================================
# 4. Visualization
# ====================================================================

fig, ax = plt.subplots(figsize=(8, 5))

# Plot the raw MSD curve
ax.plot(time_lags[1:MAX_LAG], msd_history[1:MAX_LAG], lw=2, color='darkblue', alpha=0.8, label='MSD($\\tau$) Simulation')

# Plot the linear fit line
ax.plot(X_fit, fit_line, 'r--', 
        label=f'Linear Fit (Slope = 6D = {slope:.3f})')

# Labeling and Formatting
ax.set_title('Mean-Squared Displacement (MSD) and Diffusion Coefficient')
ax.set_xlabel('Time Lag $\\tau$ (s)')
ax.set_ylabel('MSD ($\mathregular{r^2}$) / $\\langle|\\mathbf{r}(t)-\\mathbf{r}(0)|^2\\rangle$')
ax.text(0.65, 0.2, f'Diffusion Coeff. $D \\approx {D_CALCULATED:.4f}$', 
        transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# --- Conclusion ---
print("\n--- Diffusion Coefficient Analysis Summary ---")
print(f"Calculated MSD Slope (6D): {slope:.4f}")
print(f"Calculated Diffusion Coefficient (D): {D_CALCULATED:.5f}")
print(f"R-squared of Fit (Linearity Check): {r_value**2:.4f}")

print("\nConclusion: The Mean-Squared Displacement (MSD) curve shows the expected linear growth at long times. The Diffusion Coefficient (D) is accurately extracted from the slope of this linear regime using the Einstein relation, confirming the transport properties of the simulated system.")
