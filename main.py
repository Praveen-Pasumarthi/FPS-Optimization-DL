import os
import cv2
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_images  # Import the preprocessing function
from model import build_model
from keras.callbacks import EarlyStopping

# Set dataset path
data_dir = "dataset/"

# Load and show a sample image from the dataset
def load_and_show_image():
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        img_path = os.path.join(data_dir, image_files[0])  # Load the first image
        print(f"Loading image: {img_path}")

        # Ensure OpenCV reads images correctly
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"❌ Error: Unable to load image {img_path}. Check the file format and integrity.")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title("Sample Game Frame")
            plt.axis("off")
            plt.show()
    else:
        print("❌ Error: No valid image files (.png, .jpg, .jpeg) found in dataset/. Please check the folder path.")

# Load and preprocess images for model
def preprocess_for_model():
    images, labels = load_and_preprocess_images(data_dir)  # Preprocess all images in the dataset

    print(f"Processed {len(images)} images for model training.")
    print(f"Sample image shape: {images[0].shape}")  # Check the shape of a sample image

    return images, labels  # Return images and labels for training

# Train the model
def train_model(images, labels):
    model = build_model(input_shape=images.shape[1:])  # Model with input shape based on image data

    # Split the data into training and validation sets (80% train, 20% validation)
    split = int(0.8 * len(images))
    X_train, X_val = images[:split], images[split:]
    y_train, y_val = labels[:split], labels[split:]

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Save the trained model
    model.save('fps_optimization_model.h5')

    print("Model training complete.")

# Main function
if __name__ == "__main__":
    load_and_show_image()  # Show sample image
    images, labels = preprocess_for_model()  # Load and preprocess data
    train_model(images, labels)  # Train the model
