import cv2
import numpy as np

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess the game frame to match the input size and format of the trained model.
    """
    # Resize frame to the input size expected by the model
    frame_resized = cv2.resize(frame, target_size)

    # Convert to RGB (model trained with RGB images)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Normalize the image (if you used normalization during training)
    frame_normalized = frame_rgb / 255.0  # Normalize pixel values to [0, 1]

    return frame_normalized
