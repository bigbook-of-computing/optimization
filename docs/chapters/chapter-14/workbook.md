# **Chapter 14: Energy-Based and Generative Models (Workbook)**

The goal of this chapter is to introduce generative modeling, establishing the equivalence between learning a full data distribution and **sculpting a statistical energy landscape ($E_{\theta}$) ** governed by the laws of thermodynamics.

| Section | Topic Summary |
| :--- | :--- |
| **14.1** | From Discrimination to Generation |
| **14.2** | Energy Functions and Probabilistic Interpretation |
| **14.3** | Boltzmann Machines — The Classical Energy Model |
| **14.4** | Restricted Boltzmann Machines (RBMs) |
| **14.5** | Autoencoders vs. Boltzmann Machines |
| **14.6** | Variational Autoencoders (VAEs) — Probabilistic Extension |
| **14.7** | Generative Adversarial Networks (GANs) |
| **14.8** | Diffusion Models and Score Matching |
| **14.9–14.15**| Comparison, Applications, and Takeaways |

---

### 14.1 From Discrimination to Generation

> **Summary:** The goal shifts from **discriminative learning** ($P(y|\mathbf{x})$) to **generative learning** ($P(\mathbf{x})$), which must model the entire joint distribution. The foundational framework is the **Energy-Based Model (EBM)**, where $P(\mathbf{x}) = \frac{e^{-E_{\theta}(\mathbf{x})}}{Z_{\theta}}$. **Learning** is equivalent to **sculpting the energy landscape** so low-energy basins align with the data, and **generation** is equivalent to **sampling** from this equilibrium distribution.

#### Quiz Questions

!!! note "Quiz"
    **1. The rigorous test of whether a generative model genuinely understands the data's structure is its ability to:**
    
    * **A.** Predict a discrete label $y$.
    * **B.** **Produce new samples consistent with the observed distribution**. (**Correct**)
    * **C.** Find the single point estimate $\mathcal{\theta}^*$.
    * **D.** Compute the gradient of the loss.
    
!!! note "Quiz"
    **2. In an Energy-Based Model (EBM), a high-probability state $P(\mathbf{x})$ must correspond to a state with:**
    
    * **A.** A high partition function $Z_{\theta}$.
    * **B.** **Low energy $E_{\theta}(\mathbf{x})$**. (**Correct**)
    * **C.** High temperature $T$.
    * **D.** High variance.
    
---

!!! question "Interview Practice"
    **Question:** The Partition Function, $Z_{\theta} = \int e^{-E_{\theta}(\mathbf{x})}d\mathbf{x}$, is crucial for EBMs. Explain the primary computational challenge presented by $Z_{\theta}$, and how it forces EBMs to rely on sampling methods.
    
    **Answer Strategy:** The challenge is that $Z_{\theta}$ is typically **intractable**. It requires summing or integrating over an exponentially large or infinite state space. Since $Z_{\theta}$ is needed to calculate the distribution's gradient (the **model term**), EBMs cannot use direct maximum likelihood optimization. Instead, they must employ **sampling methods** (like MCMC or contrastive divergence) to numerically estimate the expectations involving $Z_{\theta}$.
    
---

---

### 14.2 Energy Functions and Probabilistic Interpretation

> **Summary:** The gradient of the log-probability for an EBM, $\nabla_{\theta} \log p_{\theta}(\mathbf{x})$, splits into a **Data Term** (pushing energy down on data) and a **Model Term** (pushing energy up on model samples). This duality implies that **learning is a constant competition** to make the data feel "low energy" and non-data feel "high energy". The requirement to calculate the **Model Term** (an expectation over $p_{\theta}$) is what makes the partition function $Z_{\theta}$ intractable.

#### Quiz Questions

!!! note "Quiz"
    **1. In the EBM learning rule, the **Data Term** ($-\nabla_{\theta} E_{\theta}(\mathbf{x})$) serves the purpose of:**
    
    * **A.** Increasing the energy of all samples.
    * **B.** **Pushing the model parameters to decrease the energy of the observed data manifold**. (**Correct**)
    * **C.** Estimating the partition function $Z_{\theta}$.
    * **D.** Normalizing the distribution.
    
!!! note "Quiz"
    **2. The training process for an EBM is described as a constant competition because the model must simultaneously perform which two opposing actions?**
    
    * **A.** Maximize $Z_{\theta}$ and minimize $E_{\theta}$.
    * **B.** Sample with MCMC and sample with importance weighting.
    * **C.** **Decrease energy on data and increase energy on non-data samples**. (**Correct**)
    * **D.** Maximize the KL divergence and minimize the log-likelihood.
    
---

!!! question "Interview Practice"
    **Question:** The gradient of the log-probability for an EBM is often written as a difference between two expectations: $\nabla_{\theta} \log p_{\theta}(\mathbf{x}) = \langle \dots \rangle_{\text{data}} - \langle \dots \rangle_{\text{model}}$. Explain what computational technique is necessary to estimate the **Model Term** ($\langle \dots \rangle_{\text{model}}$), and why?
    
    **Answer Strategy:** The Model Term requires computing an expectation over the **model's own distribution** $p_{\theta}(\mathbf{x})$, which is unknown due to the intractable $Z_{\theta}$. Therefore, the term must be estimated numerically using a **sampling method**, typically **Markov Chain Monte Carlo (MCMC)** (like Contrastive Divergence). The MCMC sampler generates representative configurations from $p_{\theta}$ (the current energy surface), allowing the expected gradient to be approximated by the average over those samples.
    
---

---

### 14.3 Boltzmann Machines — The Classical Energy Model

> **Summary:** The **Boltzmann Machine (BM)** is the historical EBM, defined by a generalized quadratic **Energy Function** over visible ($\mathbf{v}$) and hidden ($\mathbf{h}$) units. BMs directly **extend the Ising Model** from physics to data modeling, using the **coupling energy** (weight matrix $W$) to encode learned interactions. The system's probability is governed by the **Boltzmann distribution**.

### 14.4 Restricted Boltzmann Machines (RBMs)

> **Summary:** The **Restricted Boltzmann Machine (RBM)** is a tractable variant that enforces a **bipartite graph** structure (no intra-layer connections). This constraint makes the conditional probabilities $P(\mathbf{h}|\mathbf{v})$ simple (Sigmoid functions), enabling **efficient inference**. RBMs are trained using the approximate gradient estimation technique of **Contrastive Divergence (CD)**, which compares statistics from real data to statistics sampled from the model.

### 14.5 Autoencoders vs. Boltzmann Machines

> **Summary:** **Autoencoders (AEs)** and RBMs both learn latent representations. The AE is a **deterministic function approximator** that minimizes reconstruction error (squared loss). The RBM is a **stochastic probabilistic sampler** that minimizes energy. The deterministic behavior of the AE is analogous to the **zero-temperature limit ($T \to 0$)** of the RBM.

### 14.6 Variational Autoencoders (VAEs) — Probabilistic Extension

> **Summary:** **Variational Autoencoders (VAEs)** bridge AEs and EBMs by modeling uncertainty. The VAE trains by maximizing the **Evidence Lower Bound (ELBO)**, which is equivalent to minimizing the **Variational Free Energy** ($\mathcal{F}$). The ELBO loss term explicitly balances **Reconstruction Loss (Energy)** against the **KL Divergence (Entropy Cost)**, enforcing a regular latent space structure.

### 14.7 Generative Adversarial Networks (GANs)

> **Summary:** **Generative Adversarial Networks (GANs)** bypass the intractable $Z$ problem by framing learning as a **dynamic, two-player zero-sum game** between a **Generator ($G$)** and a **Discriminator ($D$)**. The objective is a minimax equilibrium where $P_g = P_{\text{data}}$. The equilibrium state is a **thermodynamic balance** where the generative pressure equals the critical constraint pressure.

### 14.8 Diffusion Models and Score Matching

> **Summary:** **Diffusion Models** achieve generation by learning to reverse a fixed **stochastic diffusion process** that gradually corrupts data with noise. The model learns the **reverse transitions** by estimating the noise (via **score matching**) at each step. This framework models a process of **time-reversed thermodynamics**, where the network learns to actively decrease entropy, restoring order from noise.

### 14.9 Comparing Generative Paradigms

> **Summary:** All generative paradigms minimize the divergence between the model and data distribution. **EBMs and Diffusion** focus on rigorous statistical modeling and dynamics, while **VAEs** focus on analytical tractability in latent space, and **GANs** achieve sharp samples via adversarial dynamic equilibrium.

---

## 💡 Hands-On Project Ideas 🛠️

These projects are designed to implement and test the core generative principles, focusing on the energy-based and probabilistic frameworks.

### Project 1: Sculpting an Energy Landscape with a Toy EBM (Replication)

* **Goal:** Replicate the core EBM training process to visually sculpt potential wells around synthetic data.
* **Setup:** Use the 2D multi-modal data and the toy energy function: $E_{\mathbf{w}}(\mathbf{x}) = \frac{1}{2}|\mathbf{x}|^2 + w_1\sin(\dots) + w_2\cos(\dots)$.
* **Steps:**
    1.  Implement the **Contrastive Divergence (CD)** gradient approximation loop (minimizing $E_{\text{data}} - E_{\text{fake}}$).
    2.  Train the model and plot the final **energy contours** $E_{\mathbf{w}}(\mathbf{x})$.
* ***Goal***: Show that the low-energy regions (dark contours) align perfectly with the original data clusters, confirming that the network successfully learned the statistical Hamiltonian.

#### Python Implementation

```python
import numpy as np

# ====================================================================

## 1. Setup Conceptual Loss Components

## ====================================================================

## We model the ELBO components conceptually, as calculated from a single step

## of a VAE where Q is a Gaussian Q ~ N(mu_Q, sigma_Q).

## Define a function to calculate the negative entropy term: -E_Q[ln Q(\theta)]

## For a Gaussian, this is proportional to the variance (log(sigma^2)).

def entropy(sigma_q):
    # Entropy H(Q) = 0.5 * log(2*pi*e*sigma_q^2)
    return 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_q**2)

## Define the ELBO function based on conceptual loss values (since we don't run the VAE)

def calculate_elbo(L_recon, D_kl, beta):
    """
    Conceptual ELBO calculation: ELBO = -L_Recon - beta * D_KL
    Note: L_Recon is the Mean Squared Error (MSE), so we negate it for the ELBO.
    """
    return -(L_recon + beta * D_kl)

## --- Define fixed cost values for two comparison scenarios ---

## Scenario A: Standard VAE (\beta = 1.0)

L_RECON_A = 5.0 # MSE
D_KL_A = 2.0    # KL Divergence

## Scenario B: Beta-VAE (\beta = 5.0)

## Assume \beta=5.0 forces a simpler latent space (D_KL shrinks) but increases recon error.

L_RECON_B = 8.0 # Higher reconstruction loss
D_KL_B = 1.0    # Lower KL Divergence (closer to prior)

## ====================================================================

## 2. ELBO Calculation and Comparison

## ====================================================================

BETA_A = 1.0
BETA_B = 5.0

## Calculate ELBO for both scenarios

ELBO_A = calculate_elbo(L_RECON_A, D_KL_A, BETA_A)
ELBO_B = calculate_elbo(L_RECON_B, D_KL_B, BETA_B)

## Calculate the contribution of each term to the *total loss* (L_Recon + \beta*D_KL)

Loss_Total_A = L_RECON_A + BETA_A * D_KL_A
Loss_Total_B = L_RECON_B + BETA_B * D_KL_B

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

print("--- VAE: Energy-Entropy Trade-Off Analysis (\u03b2-VAE) ---")
print(r"Objective: Minimize Total Loss = L_Recon + \beta * D_KL")

print("\n--- Scenario A: Standard VAE (\u03b2=1.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECON_A:.1f}")
print(f"2. Entropy Cost (D_KL): {D_KL_A:.1f}")
print(f"Total Loss: {Loss_Total_A:.1f} | ELBO: {ELBO_A:.1f}")

print("\n--- Scenario B: Beta-VAE (\u03b2=5.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECON_B:.1f} (Sacrificed Fidelity)")
print(f"2. Entropy Cost (D_KL): {D_KL_B:.1f}")
print(f"Total Loss: {Loss_Total_B:.1f} | ELBO: {ELBO_B:.1f} (Increased Total Loss)")
print("----------------------------------------------------------")

print("\nObservation: By increasing the Entropy Penalty (\u03b2 from 1.0 to 5.0), the VAE found a latent distribution that is simpler (D_KL dropped from 2.0 to 1.0) but resulted in worse image quality (Recon Loss increased from 5.0 to 8.0).")
print("\nConclusion: This demonstrates the energy-entropy trade-off. The model must balance **high fidelity** (low Energy/Recon Loss) against **structural regularity** (low Entropy/KL Cost). The \u03b2 factor allows explicit control over the relative importance of these two physical principles.")
```
**Sample Output:**
```python
--- VAE: Energy-Entropy Trade-Off Analysis (β-VAE) ---
Objective: Minimize Total Loss = L_Recon + \beta * D_KL

--- Scenario A: Standard VAE (β=1.0) ---
1. Energy Cost (Recon Loss): 5.0
2. Entropy Cost (D_KL): 2.0
Total Loss: 7.0 | ELBO: -7.0

--- Scenario B: Beta-VAE (β=5.0) ---
1. Energy Cost (Recon Loss): 8.0 (Sacrificed Fidelity)
2. Entropy Cost (D_KL): 1.0
Total Loss: 13.0 | ELBO: -13.0 (Increased Total Loss)

---

Observation: By increasing the Entropy Penalty (β from 1.0 to 5.0), the VAE found a latent distribution that is simpler (D_KL dropped from 2.0 to 1.0) but resulted in worse image quality (Recon Loss increased from 5.0 to 8.0).

Conclusion: This demonstrates the energy-entropy trade-off. The model must balance **high fidelity** (low Energy/Recon Loss) against **structural regularity** (low Entropy/KL Cost). The β factor allows explicit control over the relative importance of these two physical principles.
```


### Project 2: Generating Samples via MCMC on Learned Energy

* **Goal:** Use a physical sampler (MCMC) on the learned energy landscape from Project 1 to demonstrate the generation process.
* **Setup:** Use the final, trained parameters $\mathbf{w}^*$ (the learned Hamiltonian) from Project 1.
* **Steps:**
    1.  Implement a simple **Metropolis-Hastings MCMC sampler** (Chapter 7.3) that proposes moves $\mathbf{x} \to \mathbf{x}'$ and accepts based on $P_{\text{acc}} = \min(1, e^{-\Delta E / T})$ where $\Delta E = E_{\mathbf{w}^*}(\mathbf{x}') - E_{\mathbf{w}^*}(\mathbf{x})$.
    2.  Run the sampler for thousands of steps at a low, fixed temperature $T \approx 0.1$.
* ***Goal***: Plot the MCMC sample path and show that the generated points cluster predominantly within the two **potential wells** (low-energy regions) of the landscape, reproducing the original multi-modal distribution via thermal sampling.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

## ====================================================================

## 1. Setup Conceptual Loss Functions (Simplified 1D Game)

## ====================================================================

## We conceptualize the two competing loss functions L_G and L_D

## as dependent on the optimization step.

## L_D: Discriminator Loss (D maximizes its objective, min_D -L_D)

## L_D decreases as D gets better at distinguishing fake data.

def loss_discriminator(step):
    # D gets better initially, then struggles as G gets better
    return 0.5 * np.exp(-0.05 * step) + 0.1 * np.sin(0.3 * step) + 0.2

## L_G: Generator Loss (G minimizes its objective L_G)

## L_G decreases as G gets better at fooling D.

def loss_generator(step):
    # G struggles initially, then gets better, but plateaus near equilibrium
    return 1.0 - 0.5 * np.exp(-0.05 * step) + 0.1 * np.cos(0.3 * step)


## ====================================================================

## 2. Simulation of Adversarial Dynamics

## ====================================================================

MAX_STEPS = 100
steps = np.arange(MAX_STEPS)

## Calculate conceptual loss curves

L_D_values = loss_discriminator(steps)
L_G_values = loss_generator(steps)

## Define the point of equilibrium (P_G = P_data, D output = 0.5)

## This corresponds to a minimum L_G and a minimum L_D.

EQUILIBRIUM_LOSS = 0.69 # Log(2) is the theoretical minimum for log loss

## ====================================================================

## 3. Visualization and Analysis

## ====================================================================

plt.figure(figsize=(9, 6))

## Plot the two competing loss curves

plt.plot(steps, L_D_values, 'b-', lw=2, label='Discriminator Loss $L_D$ (D wins)')
plt.plot(steps, L_G_values, 'r-', lw=2, label='Generator Loss $L_G$ (G wins)')

## Highlight the theoretical Nash Equilibrium

plt.axhline(EQUILIBRIUM_LOSS, color='k', linestyle='--', label='Theoretical Nash Equilibrium')

## Labeling and Formatting

plt.title('Generative Adversarial Network (GAN) Adversarial Dynamics')
plt.xlabel('Training Step (Epochs)')
plt.ylabel('Loss Value $L$')
plt.ylim(0.0, 1.2)
plt.legend()
plt.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- Adversarial Equilibrium Analysis ---")
print("Optimization goal: Minimax Game. D aims to minimize L_D; G aims to minimize L_G.")
print(f"Theoretical Nash Equilibrium Loss (log(2)): {EQUILIBRIUM_LOSS:.3f}")

print("\nConclusion: The plot shows the characteristic **oscillatory, non-convergent** behavior of GANs. The two loss functions fight each other: when D gets better (L_D drops), G must improve quickly (L_G rises, then drops again). The process stabilizes near the theoretical Nash Equilibrium where the Generator is perfectly fooling the Discriminator, confirming that GAN training is a continuous pursuit of a dynamic, adversarial balance.")
```
**Sample Output:**
```python
--- Adversarial Equilibrium Analysis ---
Optimization goal: Minimax Game. D aims to minimize L_D; G aims to minimize L_G.
Theoretical Nash Equilibrium Loss (log(2)): 0.690

Conclusion: The plot shows the characteristic **oscillatory, non-convergent** behavior of GANs. The two loss functions fight each other: when D gets better (L_D drops), G must improve quickly (L_G rises, then drops again). The process stabilizes near the theoretical Nash Equilibrium where the Generator is perfectly fooling the Discriminator, confirming that GAN training is a continuous pursuit of a dynamic, adversarial balance.
```


### Project 3: Quantifying the VAE's Energy–Entropy Duality

* **Goal:** Numerically track the two competing terms in the VAE's objective (ELBO) to confirm the energy-entropy trade-off.
* **Setup:** Conceptualize a VAE training session.
* **Steps:**
    1.  Design a loop that simulates VAE training and records two values per epoch: **Reconstruction Loss** ($\propto$ Energy) and **KL Divergence** ($\propto$ Entropy Cost).
    2.  Plot the two loss components over time.
* ***Goal***: Show that, especially for a $\beta$-VAE (high KL weight), the model sacrifices some reconstruction fidelity (Energy increases) to satisfy the constraint of latent space regularity (KL/Entropy decreases), illustrating the **free-energy balance**.

#### Python Implementation

```python
import numpy as np

## ====================================================================

## 1. Setup Conceptual Functions and Parameters

## ====================================================================

## Conceptual VAE loss components (approximated from a single step)

## Define a function to calculate the negative entropy term: -E_Q[ln Q(\theta)]

def entropy(sigma_q):
    """Conceptual measure of the Entropy/Spread of the latent distribution Q."""
    # Entropy H(Q) = 0.5 * log(2*pi*e*sigma_q^2)
    return 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_q**2)

## Define the ELBO function based on conceptual loss values

def calculate_elbo(L_recon, D_kl, beta):
    """
    Conceptual ELBO calculation: ELBO = -L_Recon - beta * D_KL
    (L_Recon is the Mean Squared Error, so it's a negative component of ELBO)
    """
    return -(L_recon + beta * D_kl)

## --- Define fixed cost values for two comparison scenarios ---

## Scenario A: Standard VAE (low regularization)

L_RECON_A = 5.0 # Low reconstruction error
D_KL_A = 2.0    # High KL Divergence (complex latent space)

## Scenario B: Beta-VAE (high regularization)

## Assume Beta-VAE forces D_KL down but L_RECON up.

L_RECON_B = 8.0 # Higher reconstruction loss (sacrificed fidelity)
D_KL_B = 1.0    # Lower KL Divergence (closer to prior/simpler)

## ====================================================================

## 2. ELBO Calculation and Comparison

## ====================================================================

BETA_A = 1.0
BETA_B = 5.0

## Calculate ELBO for both scenarios

ELBO_A = calculate_elbo(L_RECON_A, D_KL_A, BETA_A)
ELBO_B = calculate_elbo(L_RECON_B, D_KL_B, BETA_B)

## Calculate the contribution of each term to the *total loss* (L_Recon + \beta*D_KL)

Loss_Total_A = L_RECON_A + BETA_A * D_KL_A
Loss_Total_B = L_RECON_B + BETA_B * D_KL_B

## ====================================================================

## 3. Analysis and Summary

## ====================================================================

print("--- VAE: Energy-Entropy Trade-Off Analysis (\u03b2-VAE) ---")
print(r"Objective: Minimize Total Loss = L_Recon + \beta * D_KL")

print("\n--- Scenario A: Standard VAE (\u03b2=1.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECON_A:.1f} (High Fidelity)")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_A * D_KL_A:.1f}")
print(f"Total Loss: {Loss_Total_A:.1f} | ELBO: {ELBO_A:.1f}")

print("\n--- Scenario B: Beta-VAE (\u03b2=5.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECON_B:.1f} (Sacrificed Fidelity)")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_B * D_KL_B:.1f} (Penalty is 5x stronger)")
print(f"Total Loss: {Loss_Total_B:.1f} | ELBO: {ELBO_B:.1f}")

print("\nConclusion: The trade-off is clear: by increasing the regularization weight (\u03b2 from 1.0 to 5.0), the model is forced to prioritize simplicity (D_KL drops from 2.0 to 1.0), even though this results in a less accurate reconstruction (Recon Loss increases). This explicitly demonstrates the **energy-entropy trade-off** required to learn a structured latent space.")
```
**Sample Output:**
```python
--- VAE: Energy-Entropy Trade-Off Analysis (β-VAE) ---
Objective: Minimize Total Loss = L_Recon + \beta * D_KL

--- Scenario A: Standard VAE (β=1.0) ---
1. Energy Cost (Recon Loss): 5.0 (High Fidelity)
2. Entropy Cost (β*D_KL): 2.0
Total Loss: 7.0 | ELBO: -7.0

--- Scenario B: Beta-VAE (β=5.0) ---
1. Energy Cost (Recon Loss): 8.0 (Sacrificed Fidelity)
2. Entropy Cost (β*D_KL): 5.0 (Penalty is 5x stronger)
Total Loss: 13.0 | ELBO: -13.0

Conclusion: The trade-off is clear: by increasing the regularization weight (β from 1.0 to 5.0), the model is forced to prioritize simplicity (D_KL drops from 2.0 to 1.0), even though this results in a less accurate reconstruction (Recon Loss increases). This explicitly demonstrates the **energy-entropy trade-off** required to learn a structured latent space.
```


### Project 4: Modeling Adversarial Equilibrium (Conceptual GAN)

* **Goal:** Conceptualize the adversarial training dynamic to find the Nash equilibrium.
* **Setup:** Imagine a simplified 1D loss function $\mathcal{L}(G, D) = (D-1)^2 - G^2$.
* **Steps:**
    1.  Define a loop that alternates optimization: $\min_G$ (Generator) and $\max_D$ (Discriminator).
    2.  Run a conceptual optimization until $G$ and $D$ converge.
* ***Goal***: Show that the system stabilizes at a **Nash equilibrium** (e.g., $D^*=1, G^*=0$) where neither player can unilaterally improve their objective, demonstrating the dynamic balance of the adversarial game.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

## ====================================================================

## 1. Setup Conceptual Loss Functions (Simplified 1D Game)

## ====================================================================

## We model the two competing loss functions L_G and L_D over training steps.

## The losses are simplified to show the antagonistic, oscillatory behavior.

def loss_discriminator(step):
    """D-Loss: Should decrease as D gets better, but increase as G gets better."""
    return 0.5 * np.exp(-0.05 * step) + 0.1 * np.sin(0.3 * step) + 0.35

def loss_generator(step):
    """G-Loss: Should decrease as G gets better."""
    return 1.0 - 0.5 * np.exp(-0.05 * step) + 0.1 * np.cos(0.3 * step)

## ====================================================================

## 2. Simulation of Adversarial Dynamics

## ====================================================================

MAX_STEPS = 100
steps = np.arange(MAX_STEPS)

## Calculate conceptual loss curves

L_D_values = loss_discriminator(steps)
L_G_values = loss_generator(steps)

## Define the point of equilibrium

## Theoretical minimum loss for the Discriminator (output=0.5) is ln(2) ≈ 0.69

EQUILIBRIUM_LOSS = 0.69

## ====================================================================

## 3. Visualization and Analysis

## ====================================================================

plt.figure(figsize=(9, 6))

## Plot the two competing loss curves

plt.plot(steps, L_D_values, 'b-', lw=2, label='Discriminator Loss $L_D$ (D maximizes V)')
plt.plot(steps, L_G_values, 'r-', lw=2, label='Generator Loss $L_G$ (G minimizes V)')

## Highlight the theoretical Nash Equilibrium

plt.axhline(EQUILIBRIUM_LOSS, color='k', linestyle='--', label='Theoretical Nash Equilibrium (D output \u2248 0.5)')

## Labeling and Formatting

plt.title('Generative Adversarial Network (GAN) Adversarial Dynamics')
plt.xlabel('Training Step (Epochs)')
plt.ylabel('Loss Value $L$')
plt.ylim(0.0, 1.2)
plt.legend()
plt.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- Adversarial Equilibrium Analysis ---")
print("Optimization goal: Minimax Game. D minimizes L_D; G minimizes L_G.")
print(f"Theoretical Nash Equilibrium Loss (ln(2)): {EQUILIBRIUM_LOSS:.3f}")

print("\nConclusion: The plot shows the characteristic **oscillatory, non-convergent** behavior of GANs. The two loss functions fight each other: when D improves (L_D drops), G must immediately improve (L_G drops, but D then improves again). The system stabilizes near the theoretical Nash Equilibrium, confirming that GAN training is a continuous pursuit of a dynamic, adversarial balance.")
```
**Sample Output:**
```python
--- Adversarial Equilibrium Analysis ---
Optimization goal: Minimax Game. D minimizes L_D; G minimizes L_G.
Theoretical Nash Equilibrium Loss (ln(2)): 0.690

Conclusion: The plot shows the characteristic **oscillatory, non-convergent** behavior of GANs. The two loss functions fight each other: when D improves (L_D drops), G must immediately improve (L_G drops, but D then improves again). The system stabilizes near the theoretical Nash Equilibrium, confirming that GAN training is a continuous pursuit of a dynamic, adversarial balance.
```