import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the output
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (FPS optimization or not)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Using binary crossentropy since it's a binary classification
                  metrics=['accuracy'])

    return model
