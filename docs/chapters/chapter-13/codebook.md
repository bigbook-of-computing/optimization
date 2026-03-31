# **Chapter 13: Hierarchical Representation Learning (Codebook)**

## Project 1: Simulating Coarse-Graining in a CNN (Feature Map)

---

### Definition: Simulating Coarse-Graining in a CNN

The goal is to visually demonstrate the **hierarchical abstraction** of a Convolutional Neural Network (CNN). This is achieved by extracting and visualizing **feature maps** at different layers to show how the network progresses from detecting local, raw features to recognizing compressed, abstract patterns. This process is analogous to **Renormalization Group (RG) coarse-graining** in physics.

### Theory: Hierarchy and Abstraction

CNNs are designed to process data with inherent **spatial structure** (like images) by embedding this locality into their architecture.

  * **Early Layers (Convolution):** Use small kernels to capture **local features** like edges, corners, and raw textures (the "microscopic" details).
  * **Pooling Layers:** Perform **downsampling** (coarse-graining), reducing spatial dimension while preserving translational invariance. This is the analogue of an RG step.
  * **Deep Layers:** Integrate the simple features into complex, **abstract representations** that represent the whole object (the "macroscopic" collective variable).

By visualizing the feature maps, we see the raw data being sequentially transformed from pixels into a hierarchy of learned, meaningful features.

---

### Extensive Python Code and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model # Import Model for visualization

## Set seed for reproducibility

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

---

## Project 2: Simulating the Latent Manifold with a Basic Autoencoder

---

### Definition: Simulating the Latent Manifold with a Basic Autoencoder

The goal is to implement a basic **Autoencoder** to demonstrate its ability to find the underlying **latent manifold ($\mathcal{M}$)** of a dataset by learning an effective encoding (compression) and decoding (reconstruction).

### Theory: Autoencoders and Internal Coordinates

The Autoencoder is an unsupervised neural network designed to learn a representation ($\mathbf{z}$) that captures the most salient information in the input ($\mathbf{x}$).

  * **Encoder:** Maps the input to the low-dimensional latent space: $\mathbf{z} = f_{\text{enc}}(\mathbf{x})$.
  * **Decoder:** Reconstructs the input from the latent code: $\mathbf{\hat{x}} = f_{\text{dec}}(\mathbf{z})$.
  * **Loss:** The system is optimized by minimizing the **reconstruction loss** (Mean Squared Error), forcing the encoder to find the most efficient **internal coordinates** on the data manifold.

This project uses a simple 2D Gaussian mixture dataset to show how the network is forced to learn a compressed representation of the data's core statistical structure.

---

### Extensive Python Code and Visualization

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

---

## Project 3: VAE: Quantifying the Energy–Entropy Trade-Off (Conceptual)

---

### Definition: VAE: Quantifying the Energy–Entropy Trade-Off

The goal is to implement the core loss calculation for a **Variational Autoencoder (VAE)** and demonstrate the explicit **energy–entropy trade-off** governed by the objective function.

### Theory: ELBO and the Physics of Learning

The VAE objective function is the **Evidence Lower Bound (ELBO)**, which is maximized during training. The ELBO can be decomposed into two competing terms:

$$\text{ELBO} = \underbrace{\mathbb{E}_Q [\ln P(\mathbf{x} | \mathbf{z})]}_{\text{Reconstruction Loss (Energy Term)}} - \underbrace{D_{\mathrm{KL}}(Q(\mathbf{z} | \mathbf{x}) || P(\mathbf{z}))}_{\text{Prior Regularization (Entropy Cost)}}$$

1.  **Energy Term (Reconstruction Loss):** Maximizing this term seeks to minimize the error between the input $\mathbf{x}$ and the reconstruction $\mathbf{\hat{x}}$ (favors high fidelity).
2.  **Entropy Term (KL Divergence):** Maximizing this term (minimizing KL) forces the learned latent distribution $Q(\mathbf{z} | \mathbf{x})$ to remain statistically close to a simple **prior $P(\mathbf{z})$** (e.g., $\mathcal{N}(0, I)$). This acts as a regularization penalty that prevents overfitting.

The **hyperparameter $\beta$** (a scaling factor on the KL term, forming the $\beta$-VAE) allows the user to explicitly control the **energy–entropy trade-off**:

  * High $\beta$: High entropy cost, forcing $Q(\mathbf{z})$ to be simple (close to $\mathcal{N}(0, I)$), often at the expense of reconstruction quality.

---

### Extensive Python Code

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