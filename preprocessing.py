import os
import cv2
import numpy as np

# Function to load and preprocess the images
def load_and_preprocess_images(data_dir, image_size=(224, 224)):
    images = []
    labels = []  # You'll need to add your own logic for labels

    # Iterate through files and read images
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Construct the full image path
        img_path = os.path.join(data_dir, image_file)

        # Read the image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is not None:  # Check if the image was successfully loaded
            # Resize the image to the desired size
            image = cv2.resize(image, image_size)

            # Normalize pixel values to [0, 1]
            image = image / 255.0

            # Append the image to the list
            images.append(image)

            # Example label: Replace with your actual FPS labels (0 or 1 for now)
            labels.append(np.random.randint(0, 2))  # Random label for this example

    return np.array(images), np.array(labels)

# Example usage
data_dir = "dataset/"  # Path to the dataset folder
images, labels = load_and_preprocess_images(data_dir)

# You can save or return the processed images and labels for further use
print(f"Processed {len(images)} images.")
