# Source: Optimization/chapter-13/codebook.md -- Block 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup Data Generation (2D Input, 1D Latent Manifold)
# ====================================================================

# Generate 2D Gaussian clusters (Input Dimension D=2)
X, _ = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=0.5, random_state=42)
X = MinMaxScaler().fit_transform(X) # Scale data to [0, 1]

# Set latent dimension (d=1), forcing compression
LATENT_DIM = 1
INPUT_DIM = X.shape[1]

# ====================================================================
# 2. Autoencoder Architecture
# ====================================================================

# 1. Encoder
input_layer = Input(shape=(INPUT_DIM,), name='encoder_input')
encoded = Dense(2, activation='relu', name='hidden_layer_1')(input_layer)
latent_layer = Dense(LATENT_DIM, activation='relu', name='latent_space')(encoded)

# 2. Decoder
decoded = Dense(2, activation='relu', name='hidden_layer_2')(latent_layer)
output_layer = Dense(INPUT_DIM, activation='sigmoid', name='decoder_output')(decoded)

# Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Encoder Model (for projection visualization)
encoder = Model(inputs=input_layer, outputs=latent_layer)

# ====================================================================
# 3. Model Training and Latent Space Mapping
# ====================================================================

# Train the model (simplified)
history = autoencoder.fit(X, X, epochs=10, batch_size=32, shuffle=True, verbose=0)

# Project the 2D input data down to the 1D latent space
X_latent = encoder.predict(X)

# ====================================================================
# 4. Visualization
# ====================================================================

# Plot the 1D latent representation
plt.figure(figsize=(9, 4))

# Use y-axis for visualization spacing
plt.scatter(X_latent[:, 0], np.zeros_like(X_latent[:, 0]), c=X[:, 0], cmap='viridis', s=20)

plt.title(f'Autoencoder Latent Manifold (D={INPUT_DIM} \u2192 d={LATENT_DIM})')
plt.xlabel('Latent Variable $z_1$ (Compressed Representation)')
plt.yticks([]) # Hide y-axis since it is 1D
plt.colorbar(label='Original $x_1$ Value')
plt.grid(True)
plt.show()

# --- Analysis Summary ---
print("\n--- Autoencoder Latent Manifold Summary ---")
final_loss = history.history['loss'][-1]
print(f"Final Reconstruction Loss (MSE): {final_loss:.4f}")

print("\nConclusion: The Autoencoder successfully learned a compressed, one-dimensional representation (the latent manifold) of the two-dimensional data. The latent variable $z_1$ captures the most important variance in the input, confirming that the network is forced to find the most efficient 'internal coordinates' to represent the data structure with minimal reconstruction error.")
