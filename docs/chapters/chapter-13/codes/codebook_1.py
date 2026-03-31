# Source: Optimization/chapter-13/codebook.md -- Block 1

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model # Import Model for visualization

# Set seed for reproducibility
np.random.seed(42)

# ====================================================================
# 1. Setup and Pre-trained Model (Simple CNN for MNIST)
# ====================================================================

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train, num_classes=10)

# Define a simple, pre-trainable CNN model
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

# Train the model
model = create_and_train_cnn()

# Select an input image to analyze (e.g., the digit '4')
input_image = x_test[2:3]
true_label = np.argmax(y_test[2])

# ====================================================================
# 2. Feature Map Extraction
# ====================================================================

# Define models to output the feature maps at specific layers
# Model for Layer 1: Output after the first convolution
layer1_output = model.get_layer('conv_1').output
layer1_model = Model(inputs=model.inputs, outputs=layer1_output)
features_l1 = layer1_model.predict(input_image)[0]

# Model for Layer 2: Output after the second pooling layer (high abstraction/compression)
layer2_output = model.get_layer('pool_2').output
layer2_model = Model(inputs=model.inputs, outputs=layer2_output)
features_l2 = layer2_model.predict(input_image)[0]

# ====================================================================
# 3. Visualization
# ====================================================================

# Display the input image
plt.figure(figsize=(2, 2))
plt.imshow(input_image[0, :, :, 0], cmap='gray')
plt.title(f"Input Image (Digit: {true_label})")
plt.axis('off')
plt.show()

# --- Plot 1: Low-Level Features (Layer 1: Edge Detection) ---
n_filters_l1 = features_l1.shape[-1]
fig1, axs1 = plt.subplots(4, 8, figsize=(10, 5))
fig1.suptitle('Layer 1 Feature Maps (Microscopic/Local Features)', fontsize=14)

for i in range(min(n_filters_l1, 32)):
    row, col = i // 8, i % 8
    axs1[row, col].imshow(features_l1[:, :, i], cmap='viridis')
    axs1[row, col].axis('off')
    
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Plot 2: High-Level Features (Layer 2: Abstract/Coarse-Grained) ---
n_filters_l2 = features_l2.shape[-1]
fig2, axs2 = plt.subplots(8, 8, figsize=(10, 10))
fig2.suptitle('Layer 2 Feature Maps (Macroscopic/Abstract Abstraction)', fontsize=14)

for i in range(min(n_filters_l2, 64)):
    row, col = i // 8, i % 8
    axs2[row, col].imshow(features_l2[:, :, i], cmap='plasma')
    axs2[row, col].axis('off')
    
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Analysis Summary ---
print("\n--- Hierarchical Abstraction Analysis ---")
print("Layer 1 (Pre-Pooling): Maps are large (26x26) and highlight simple elements like diagonal and horizontal edges. This is the **low-level, microscopic** view of the data.")
print("Layer 2 (Post-Pooling): Maps are highly compressed (5x5) and contain highly abstract patterns. The network has successfully integrated local edges into features representing macroscopic shapes, confirming the **coarse-graining principle** of the architecture.")
