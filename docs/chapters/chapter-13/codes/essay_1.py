# Source: Optimization/chapter-13/essay.md -- Block 1

import tensorflow as tf
from tensorflow.keras import layers, models

# Build model
model = models.Sequential([
    # Layer 1: Convolution + Non-linearity + Coarse-Graining
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    # Layer 2: Deeper Feature Extraction
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    # Transition to Dense/Classification
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
