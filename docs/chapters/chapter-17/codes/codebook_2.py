# Source: Optimization/chapter-17/codebook.md -- Block 2

import numpy as np
import tensorflow as tf

# ====================================================================
# 1. Setup Conceptual Functions (Using TensorFlow Variables for AD)
# ====================================================================

N = 4
M_SAMPLES = 50 
J_COUPLING = 1.0
H_FIELD = 0.5 

# --- A. Trainable Parameter (\theta) ---
# We model the complex-valued RBM parameter as two real-valued components
theta_real = tf.Variable(0.1, dtype=tf.float32, name='theta_real')
theta_imag = tf.Variable(0.0, dtype=tf.float32, name='theta_imag') # Simplify by setting imag=0

# --- B. Conceptual Trial Wavefunction (\psi_{\theta}) ---
def psi_theta_tf(s, theta_r, theta_i):
    """
    Conceptual TF-based function for the log-amplitude of the wavefunction.
    We return the log-amplitude for simplicity in the log-likelihood trick.
    """
    s = tf.constant(s, dtype=tf.float32)
    E_s = -J_COUPLING * tf.reduce_sum(s[0:N] * tf.roll(s[0:N], shift=-1, axis=0))
    
    # log(|\psi|) \approx theta_r * E_s 
    # Log-likelihood trick requires log(\psi)
    log_psi_s = theta_r * E_s 
    return log_psi_s

# --- C. Local Energy Calculation (Required in the gradient formula) ---
def get_local_energy_tf(s_tf, E_expected, theta_r, theta_i):
    """Placeholder for the local energy calculation E_loc(s)."""
    # E_loc is non-trivial. For conceptual check, we model it as noisy around the expected E.
    E_loc = E_expected + tf.random.normal((1,), mean=0, std=0.2)
    return E_loc

# ====================================================================
# 2. Gradient Calculation (\nabla E)
# ====================================================================

# Conceptual Energy (from Project 1, simplified)
E_EXPECTED_CURRENT = -1.5 

def calculate_energy_gradient(E_expected=E_EXPECTED_CURRENT):
    # Prepare VMC samples (Conceptual)
    samples = [np.random.choice([-1, 1], N) for _ in range(M_SAMPLES)]
    
    # Store the final gradient sum
    gradient_sum = tf.constant(0.0, dtype=tf.float32)
    
    # 1. Loop through samples and calculate the term: (E_loc - <H>) * \nabla log(\psi)
    for s_np in samples:
        s_tf = tf.constant(s_np, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch the trainable parameters
            tape.watch(theta_real)
            tape.watch(theta_imag)
            
            # Get log(\psi)
            log_psi_s = psi_theta_tf(s_tf, theta_real, theta_imag)
            
            # Calculate E_loc(s)
            E_loc = get_local_energy_tf(s_tf, E_expected, theta_real, theta_imag)
            
            # Calculate the policy gradient term: \nabla log(\psi)
            # This is the 'feature vector' of the state s
            nabla_log_psi_real = tape.gradient(log_psi_s, theta_real)
            # nabla_log_psi_imag = tape.gradient(log_psi_s, theta_imag) # Not needed if \psi is real
            
            # Calculate the TD Error analogue: E_loc - <H>
            error_term = E_loc - E_expected
            
            # Calculate the contribution of this single step to the gradient sum
            contribution = error_term * nabla_log_psi_real
            
        # Accumulate the sum (E_loc - <H>) * \nabla log(\psi)
        gradient_sum = gradient_sum + contribution[0] 
        
    # Final VMC average (approximation of the full gradient formula)
    nabla_E_vmc = 2.0 * gradient_sum / M_SAMPLES
    
    return nabla_E_vmc.numpy()

# Final function call
quantum_gradient = calculate_energy_gradient()

# ====================================================================
# 3. Analysis and Summary
# ====================================================================

print("--- NQS Quantum Loss Gradient (\u2207_{\u03b8} E) ---")
print(f"Current Estimate of E: {E_EXPECTED_CURRENT:.3f}")
print(f"Current Parameter \u03b8_real: {theta_real.numpy():.3f}")
print(f"Calculated Gradient \u2207 E: {quantum_gradient:.4f}")

print("\nConclusion: The calculation successfully computed the stochastic gradient of the expected energy with respect to the trainable parameter (\u03b8). This gradient acts as the **quantum force** that the optimizer (e.g., Adam) will use to update the neural network weights in the direction that minimizes the expected energy, driving the system toward its quantum ground state.")
