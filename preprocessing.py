import os
import cv2
import numpy as np

# load and preprocess the images
def load_and_preprocess_images(data_dir, image_size=(224, 224)):
    images = []
    labels = []

    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        img_path = os.path.join(data_dir, image_file)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is not None:  
           
            image = cv2.resize(image, image_size)

            image = image / 255.0

            images.append(image)

            labels.append(np.random.randint(0, 2)) 

    return np.array(images), np.array(labels)

data_dir = "dataset/"
images, labels = load_and_preprocess_images(data_dir)

print(f"Processed {len(images)} images.")
