# **Chapter 13: Hierarchical Representation Learning (Workbook)**

The goal of this chapter is to study specialized **deep architectures** that embed geometric knowledge (space, time) into their structure, revealing how networks build a hierarchy of abstraction analogous to physical coarse-graining.

| Section | Topic Summary |
| :--- | :--- |
| **13.1** | From Perceptrons to Hierarchies |
| **13.2** | The Geometry of Deep Representations |
| **13.3** | Convolutional Neural Networks (CNNs) — Spatial Structure |
| **13.4** | Hierarchical Feature Maps |
| **13.5** | Temporal Hierarchies — Recurrent Neural Networks (RNNs) |
| **13.6** | Autoencoders — Learning Internal Coordinates |
| **13.7** | Variational Autoencoders (VAEs) — Probabilistic Extension |
| **13.8** | Spatial–Temporal Hybrids |
| **13.9** | Regularization and Generalization in Deep Networks |
| **13.10–13.15**| Visualization, Worked Example, and Takeaways |

---

### 13.1 From Perceptrons to Hierarchies

> **Summary:** Deep networks achieve computational power by stacking the non-linear unit (perceptron), where **composition of non-linear mappings** yields immense expressive power. The sequential transformation (layer $l \to l+1$) is analogous to the **Renormalization Group (RG) framework** in physics, as each layer performs a step of **coarse-graining** to extract effective, abstract variables.

#### Quiz Questions

!!! note "Quiz"
    **1. The Universal Approximation Theorem proves that a single hidden layer (of sufficient size) and non-linear activation can approximate any continuous function. The added benefit of *stacking* layers (depth) is to facilitate:**
    
    * **A.** The loss of all variance.
    * **B.** **The hierarchical extraction of emergent features**. (**Correct**)
    * **C.** The reduction of the learning rate $\eta$.
    * **D.** The calculation of the geodesic distance.
    
!!! note "Quiz"
    **2. The process where a deep network iteratively removes microscopic details and redundancy to encode a new, effective state variable $\mathbf{h}^{(l+1)}$ is directly analogous to which physical framework?**
    
    * **A.** The Ising model Hamiltonian.
    * **B.** **The Renormalization Group (RG) framework**. (**Correct**)
    * **C.** Quantum tunneling.
    * **D.** Orthogonal projection.
    
---

!!! question "Interview Practice"
    **Question:** Explain the philosophical significance of viewing each layer of a deep neural network as a step of **coarse-graining** in the Renormalization Group sense?
    
    **Answer Strategy:** The significance is that it provides a physical justification for the network's structure. In physics, coarse-graining is necessary to reveal macroscopic laws by ignoring high-frequency, microscopic noise. By performing this iteratively, the network is automatically learning a **hierarchical theory of the data**; it progresses from raw input to abstract **order parameters** that genuinely describe the system's macroscopic state, without any human defining those intermediate features.
    
---

---

### 13.2 The Geometry of Deep Representations

> **Summary:** Deep networks manipulate the **geometry of the data space** by performing successive **change of coordinates**. The primary goal is to **unfold, disentangle, and flatten** the complex, high-dimensional data manifold ($\mathcal{M}$). The network seeks a final, deep representation where data belonging to different classes are trivially **linearly separable**. This hierarchical process is analogous to **progressively flattening a rough potential surface** for easier optimization dynamics.

#### Quiz Questions

!!! note "Quiz"
    **1. The primary geometric goal of the successive transformations performed by the hidden layers in a deep network is to:**
    
    * **A.** Preserve the Euclidean distance between all points.
    * **B.** **Disentangle the structure and flatten the manifold for linear separation**. (**Correct**)
    * **C.** Increase the overall dimension $D$.
    * **D.** Maximize the total energy.
    
!!! note "Quiz"
    **2. In the physical analogy, the process of hierarchical learning is compared to applying successive transformations that gradually flatten a curved potential surface. The final state of this flattening allows the solution to be found by:**
    
    * **A.** Introducing a high learning rate $\eta$.
    * **B.** **A simple linear classifier**. (**Correct**)
    * **C.** Quantum annealing.
    * **D.** Calculating the partition function.
    
---

!!! question "Interview Practice"
    **Question:** How does the design goal of a deep network (to flatten the data manifold) relate to the concept of the final output layer being a simple **linear classifier** (Chapter 10)?
    
    **Answer Strategy:** The high-level layers of a deep network are doing the hard work: they are warping the space through non-linearity until the data, which was highly complex in the raw input space, becomes **linearly separable** in the final hidden feature space. Once the data is linearly separable, the final classification layer only needs to apply a simple linear model (like Logistic Regression or a Perceptron) to draw a straight line or plane to separate the classes, making the final classification step computationally trivial.
    
---

---

### 13.3 Convolutional Neural Networks (CNNs) — Spatial Structure

> **Summary:** CNNs are specialized architectures that embed knowledge of **spatial structure**. The core **convolution layer** replaces dense weights with a small, shared **kernel**. This design incorporates two **inductive biases**: **Locality** (small receptive fields mimic nearest-neighbor interaction) and **Weight Sharing** (enforcing translation invariance). **Pooling** layers perform the coarse-graining step. The entire process is analogous to **real-space renormalization** in physics.

#### Quiz Questions

!!! note "Quiz"
    **1. The CNN feature of **Weight Sharing** is a computational mechanism that directly enforces the physical property of:**
    
    * **A.** Entropy reduction.
    * **B.** **Translation symmetry (invariance)**. (**Correct**)
    * **C.** Temporal recurrence.
    * **D.** Orthogonal projection.
    
!!! note "Quiz"
    **2. In the CNN analogy to physics, the small, local **kernel** in the convolution layer models which structural feature?**
    
    * **A.** Global correlation.
    * **B.** **Nearest-neighbor interaction**. (**Correct**)
    * **C.** The bias-variance trade-off.
    * **D.** The time step $\Delta t$.
    
---

!!! question "Interview Practice"
    **Question:** The CNN uses both **Convolution** and **Pooling** layers. Explain the respective function of each layer in terms of the **Renormalization Group (RG)** framework.
    
    **Answer Strategy:**
    1.  **Convolution $\leftrightarrow$ Feature Extraction:** The convolution layer applies local filters to extract fundamental patterns (e.g., edges, textures) from the neighborhood, identifying **short-range correlations**.
    2.  **Pooling $\leftrightarrow$ Coarse-Graining:** The pooling layer summarizes and reduces the spatial dimension (e.g., taking the maximum activation over a $2 \times 2$ block). This step is the direct analogue of **coarse-graining** in RG theory: removing high-frequency, microscopic noise and redundancy to reveal the simplified, lower-dimensional structure.
    
---

---

### 13.4 Hierarchical Feature Maps

> **Summary:** Deep CNNs build a hierarchy of features, progressing from **low-level features** (microscopic: edges, lines) to **high-level features** (macroscopic: semantic objects, order parameters). This progression mirrors the **hierarchy of excitations** in physical systems. The most active neurons in deep layers often act as **emergent order parameters**, which are abstract variables that characterize the system's macroscopic state.

### 13.5 Temporal Hierarchies — Recurrent Neural Networks (RNNs)

> **Summary:** **Recurrent Neural Networks (RNNs)** are specialized architectures for **sequential data** (time series, language). They achieve **memory** by introducing **feedback**, where the hidden state $\mathbf{h}_t$ depends on both the current input $\mathbf{x}_t$ and the previous state $\mathbf{h}_{t-1}$. The recurrent weight matrix ($W_h$) defines the **transition function** of this non-linear **dynamical system**. **LSTMs** and **GRUs** use internal gating mechanisms to solve the **vanishing gradient problem** and maintain long-term memory.

### 13.6 Autoencoders — Learning Internal Coordinates

> **Summary:** **Autoencoders (AEs)** are architectures for **unsupervised learning** and dimensionality reduction. The AE uses an **Encoder** to map high-D input to a low-D **latent code ($\mathbf{z}$)** and a **Decoder** to reconstruct the input. The objective is to minimize **reconstruction loss**, forcing $\mathbf{z}$ to become the optimal **information compressor**. The entire AE operation is analogous to modeling the **Renormalization Group (RG) flow** forward (encoder) and backward (decoder).

### 13.7 Variational Autoencoders (VAEs) — Probabilistic Extension

> **Summary:** **Variational Autoencoders (VAEs)** embed **probabilistic inference** into the AE structure. VAEs train by maximizing the **Evidence Lower Bound (ELBO)**, which explicitly balances two terms: **Reconstruction Loss (Energy)** and the **KL Divergence** (forcing the latent space to adhere to a regular prior, analogous to minimizing **Entropy Cost**). The VAE's objective function directly embodies the thermodynamic **Energy–Entropy duality**.

### 13.8 Spatial–Temporal Hybrids

> **Summary:** **ConvRNNs** (Convolutional RNNs) combine CNN kernels (for spatial locality) and recurrent connections (for temporal memory) to model systems coupled across both space and time (e.g., fluid dynamics, video prediction). These models act as **probabilistic numerical solvers** that learn the **transition function** (equations of motion) of a continuous physical field.

### 13.9 Regularization and Generalization in Deep Networks

> **Summary:** Deep networks use **regularization** to control complexity and improve **generalization** by managing the **Bias–Variance Tradeoff**. **L2 Weight Decay** acts as a Gaussian prior (energetic constraint). **Dropout** acts as **entropy injection**, forcing the network to average over an ensemble of thinned sub-networks. **Early Stopping** acts as **cooling control**, halting training at a point of optimal thermal balance.

---

## 💡 Hands-On Project Ideas 🛠️

These projects focus on implementing and testing the specialized architectures and their physical analogies.

### Project 1: Simulating RNN Memory and Dynamics

* **Goal:** Implement a simple RNN and demonstrate its ability to maintain memory over a short sequence.
* **Setup:** Define a simple, small RNN with a single recurrent layer (Section 13.5).
* **Steps:**
    1.  Train the RNN on a simple sequential prediction task (e.g., predicting the next item in a repeating sequence like `1, 0, 1, 0, ...`).
    2.  Test the trained network by feeding it a sequence with a gap (e.g., `1, 0, 1, [Gap], 1`).
* ***Goal***: Show that the hidden state $\mathbf{h}_t$ successfully carries the necessary information (memory) across the temporal gap to correctly predict the missing element, illustrating its function as a dynamical system.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model # Import Model for visualization

# Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup and Pre-trained Model (Simple CNN for MNIST)

## ====================================================================

## Load and preprocess MNIST data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train, num_classes=10)

## Define a simple, pre-trainable CNN model

def create_and_train_cnn():
    model = Sequential([
        # Layer 1: Low-level features
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv_1'),
        MaxPooling2D((2, 2), name='pool_1'), # Coarse-graining step

        # Layer 2: Higher-level abstraction
        Conv2D(64, (3, 3), activation='relu', name='conv_2'),
        MaxPooling2D((2, 2), name='pool_2'), # Second coarse-graining step

        Flatten(),
        Dense(10, activation='softmax')
    ])

    # Train quickly to get functional weights (essential for visualization)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train[:1000], y_train_cat[:1000], epochs=1, verbose=0)
    return model

## Train the model

model = create_and_train_cnn()

## Select an input image to analyze (e.g., the digit '4')

input_image = x_test[2:3]
true_label = np.argmax(y_test[2])

## ====================================================================

## 2. Feature Map Extraction

## ====================================================================

## Define models to output the feature maps at specific layers

## Model for Layer 1: Output after the first convolution

layer1_output = model.get_layer('conv_1').output
layer1_model = Model(inputs=model.inputs, outputs=layer1_output)
features_l1 = layer1_model.predict(input_image)[0]

## Model for Layer 2: Output after the second pooling layer (high abstraction/compression)

layer2_output = model.get_layer('pool_2').output
layer2_model = Model(inputs=model.inputs, outputs=layer2_output)
features_l2 = layer2_model.predict(input_image)[0]

## ====================================================================

## 3. Visualization

## ====================================================================

## Display the input image

plt.figure(figsize=(2, 2))
plt.imshow(input_image[0, :, :, 0], cmap='gray')
plt.title(f"Input Image (Digit: {true_label})")
plt.axis('off')
plt.show()

## --- Plot 1: Low-Level Features (Layer 1: Edge Detection) ---

n_filters_l1 = features_l1.shape[-1]
fig1, axs1 = plt.subplots(4, 8, figsize=(10, 5))
fig1.suptitle('Layer 1 Feature Maps (Microscopic/Local Features)', fontsize=14)

for i in range(min(n_filters_l1, 32)):
    row, col = i // 8, i % 8
    axs1[row, col].imshow(features_l1[:, :, i], cmap='viridis')
    axs1[row, col].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

## --- Plot 2: High-Level Features (Layer 2: Abstract/Coarse-Grained) ---

n_filters_l2 = features_l2.shape[-1]
fig2, axs2 = plt.subplots(8, 8, figsize=(10, 10))
fig2.suptitle('Layer 2 Feature Maps (Macroscopic/Abstract Abstraction)', fontsize=14)

for i in range(min(n_filters_l2, 64)):
    row, col = i // 8, i % 8
    axs2[row, col].imshow(features_l2[:, :, i], cmap='plasma')
    axs2[row, col].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

## --- Analysis Summary ---

print("\n--- Hierarchical Abstraction Analysis ---")
print("Layer 1 (Pre-Pooling): Maps are large (26x26) and highlight simple elements like diagonal and horizontal edges. This is the **low-level, microscopic** view of the data.")
print("Layer 2 (Post-Pooling): Maps are highly compressed (5x5) and contain highly abstract patterns. The network has successfully integrated local edges into features representing macroscopic shapes, confirming the **coarse-graining principle** of the architecture.")
```

### Project 2: Autoencoder Manifold Discovery (Non-Linear PCA)

* **Goal:** Use an Autoencoder (AE) to discover the non-linear manifold of a simple dataset, contrasting it with PCA.
* **Setup:** Generate a 2D **curved** dataset (e.g., a simple spiral or C-shaped manifold in $\mathbb{R}^2$).
* **Steps:**
    1.  Design a simple AE with an Encoder and Decoder (MLPs with ReLU) and a 1D latent code ($\mathbf{z} \in \mathbb{R}^1$).
    2.  Train the AE to minimize reconstruction loss.
    3.  Extract the 1D latent code $\mathbf{z}$ for all samples and plot the raw data colored by the value of $\mathbf{z}$.
* ***Goal***: Show that the AE learns a continuous, single coordinate that successfully "unrolls" the curved structure, performing a **non-linear dimensionality reduction** superior to linear PCA.

#### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

## Set seed for reproducibility

np.random.seed(42)

## ====================================================================

## 1. Setup Data Generation (2D Input, 1D Latent Manifold)

## ====================================================================

## Generate 2D Gaussian clusters (Input Dimension D=2)

X, _ = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=0.5, random_state=42)
X = MinMaxScaler().fit_transform(X) # Scale data to [0, 1]

## Set latent dimension (d=1), forcing compression

LATENT_DIM = 1
INPUT_DIM = X.shape[1]

## ====================================================================

## 2. Autoencoder Architecture

## ====================================================================

## 1. Encoder

input_layer = Input(shape=(INPUT_DIM,), name='encoder_input')
encoded = Dense(2, activation='relu', name='hidden_layer_1')(input_layer)
latent_layer = Dense(LATENT_DIM, activation='relu', name='latent_space')(encoded)

## 2. Decoder

decoded = Dense(2, activation='relu', name='hidden_layer_2')(latent_layer)
output_layer = Dense(INPUT_DIM, activation='sigmoid', name='decoder_output')(decoded)

## Autoencoder Model

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

## Encoder Model (for projection visualization)

encoder = Model(inputs=input_layer, outputs=latent_layer)

## ====================================================================

## 3. Model Training and Latent Space Mapping

## ====================================================================

## Train the model (simplified)

history = autoencoder.fit(X, X, epochs=10, batch_size=32, shuffle=True, verbose=0)

## Project the 2D input data down to the 1D latent space

X_latent = encoder.predict(X)

## ====================================================================

## 4. Visualization

## ====================================================================

## Plot the 1D latent representation

plt.figure(figsize=(9, 4))

## Use y-axis for visualization spacing

plt.scatter(X_latent[:, 0], np.zeros_like(X_latent[:, 0]), c=X[:, 0], cmap='viridis', s=20)

plt.title(f'Autoencoder Latent Manifold (D={INPUT_DIM} \u2192 d={LATENT_DIM})')
plt.xlabel('Latent Variable $z_1$ (Compressed Representation)')
plt.yticks([]) # Hide y-axis since it is 1D
plt.colorbar(label='Original $x_1$ Value')
plt.grid(True)
plt.show()

## --- Analysis Summary ---

print("\n--- Autoencoder Latent Manifold Summary ---")
final_loss = history.history['loss'][-1]
print(f"Final Reconstruction Loss (MSE): {final_loss:.4f}")

print("\nConclusion: The Autoencoder successfully learned a compressed, one-dimensional representation (the latent manifold) of the two-dimensional data. The latent variable $z_1$ captures the most important variance in the input, confirming that the network is forced to find the most efficient 'internal coordinates' to represent the data structure with minimal reconstruction error.")
```

### Project 3: Quantifying the VAE's Energy–Entropy Balance

* **Goal:** Implement the VAE objective and analyze the trade-off between its two loss terms (Energy/Reconstruction vs. Entropy/KL).
* **Setup:** Design a simple VAE and train it on a small dataset (e.g., MNIST digits).
* **Steps:**
    1.  Track and plot the value of the **Reconstruction Loss** (the energy term) and the **KL Divergence** (the entropy cost) separately over the training epochs.
    2.  Introduce a large weighting factor ($\beta \gg 1$) on the KL term (a "$\\beta$-VAE").
* ***Goal***: Show that a large $\beta$ forces the KL term down (low entropy cost) but significantly *increases* the Reconstruction Loss (poor image quality), demonstrating the explicit **energy–entropy trade-off** governed by the $\beta$ parameter.

#### Python Implementation

```python
import numpy as np

## ====================================================================

## 1. Setup Conceptual Loss Components

## ====================================================================

## Conceptual values for the three losses from a single training step

## Assume the optimization is running, and these values are calculated.

## Cost 1: Reconstruction Error (Energy Term)

## Lower is better (we maximize -L_reconstruction)

L_RECONSTRUCTION = 5.0

## Cost 2: KL Divergence (Entropy Cost)

## Lower is better (we maximize -D_KL)

D_KL = 2.0

## Total Loss (Objective) is L_RECONSTRUCTION + \beta * D_KL

## The ELBO is ELBO = -Total Loss

## --- Scenario A: Standard VAE (\beta = 1.0) ---

BETA_A = 1.0
ELBO_A = - (L_RECONSTRUCTION + BETA_A * D_KL)

## --- Scenario B: High Entropy Cost (\beta = 5.0, forcing simpler latent space) ---

BETA_B = 5.0
## Assume that forcing the KL term higher (BETA_B=5) leads to higher reconstruction error

## because the model is more constrained.

L_RECONSTRUCTION_B = 8.0
ELBO_B = - (L_RECONSTRUCTION_B + BETA_B * D_KL)

## ====================================================================

## 2. Analysis and Summary

## ====================================================================

print("--- VAE: Energy-Entropy Trade-Off Analysis ---")
print(r"Objective: Maximize ELBO = - (L_Recon + \beta * D_KL)")

print("\n--- Scenario A: Standard VAE (\u03b2=1.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECONSTRUCTION:.1f}")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_A * D_KL:.1f}")
print(f"Total Loss: {L_RECONSTRUCTION + BETA_A * D_KL:.1f} | ELBO: {ELBO_A:.1f}")

print("\n--- Scenario B: Beta-VAE (\u03b2=5.0) ---")
print(f"1. Energy Cost (Recon Loss): {L_RECONSTRUCTION_B:.1f} (Increased)")
print(f"2. Entropy Cost (\u03b2*D_KL): {BETA_B * D_KL:.1f} (Penalty is 5x stronger)")
print(f"Total Loss: {L_RECONSTRUCTION_B + BETA_B * D_KL:.1f} | ELBO: {ELBO_B:.1f}")
print("----------------------------------------------------------")

print("\nObservation: The high \u03b2 (5.0) in Scenario B drastically increased the penalty for the complexity of the latent representation (Entropy Cost = 10.0), even though it resulted in a worse image quality (Energy Cost = 8.0).")
print("\nConclusion: This demonstrates the **Energy-Entropy Trade-Off**: the \u03b2 parameter allows the user to balance the desire for **high fidelity** (low Energy Cost) against the desire for a **simple, structured latent space** (low Entropy Cost). This control is crucial for disentangling the generative factors of the data.")
```
**Sample Output:**
```python
--- VAE: Energy-Entropy Trade-Off Analysis ---
Objective: Maximize ELBO = - (L_Recon + \beta * D_KL)

--- Scenario A: Standard VAE (β=1.0) ---
1. Energy Cost (Recon Loss): 5.0
2. Entropy Cost (β*D_KL): 2.0
Total Loss: 7.0 | ELBO: -7.0

--- Scenario B: Beta-VAE (β=5.0) ---
1. Energy Cost (Recon Loss): 8.0 (Increased)
2. Entropy Cost (β*D_KL): 10.0 (Penalty is 5x stronger)
Total Loss: 18.0 | ELBO: -18.0

---

Observation: The high β (5.0) in Scenario B drastically increased the penalty for the complexity of the latent representation (Entropy Cost = 10.0), even though it resulted in a worse image quality (Energy Cost = 8.0).

Conclusion: This demonstrates the **Energy-Entropy Trade-Off**: the β parameter allows the user to balance the desire for **high fidelity** (low Energy Cost) against the desire for a **simple, structured latent space** (low Entropy Cost). This control is crucial for disentangling the generative factors of the data.
```


### Project 4: Simulating Coarse-Graining in a CNN (Feature Map)

* **Goal:** Visually demonstrate the **hierarchical abstraction** of a CNN (Section 13.4).
* **Setup:** Use a pre-trained simple CNN model (like the one in the demo) on an image dataset (e.g., MNIST).
* **Steps:**
    1.  Select a single input image (e.g., the digit '4').
    2.  Extract and visualize the feature maps after the **first convolutional layer** (low level).
    3.  Extract and visualize the feature maps after the **final pooling layer** (high level).
* ***Goal***: Show that the first layer's maps highlight simple, raw edges (microscopic), while the final layer's maps are highly compressed and abstract, showing only the presence of the full object (macroscopic), confirming the **RG coarse-graining principle**.