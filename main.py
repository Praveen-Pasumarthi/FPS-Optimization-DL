import os
import cv2
import pyautogui
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_images 
from model import build_model  
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from frame_preprocessing import preprocess_frame 
from model_loading import load_trained_model

# Dataset path
data_dir = "dataset/"

# Screen capture Func
def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    
    screenshot = np.array(screenshot)
    
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    
    return screenshot

# shows the captured frame
def show_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Captured Game Frame")
    plt.axis("off")
    plt.show(block=False) 
    plt.pause(0.1)  
    plt.close() 

# Predicts FPS optimization based on the frame
def predict_fps_optimization(model, frame):
    
    frame_preprocessed = preprocess_frame(frame)
    
    frame_input = np.expand_dims(frame_preprocessed, axis=0)

    # Predict FPS optimization requirement
    prediction = model.predict(frame_input)
    prediction_value = prediction[0][0] 

    print(f"Prediction value: {prediction_value}")  

    if prediction_value > 0.7: 
        print("⚠️ FPS optimization required!")
        return True
    else:
        print("✅ FPS is fine.")
        return False
    
# Function to load, preprocess images, and train model
def preprocess_and_train_model():
    model = build_model(input_shape=(224, 224, 3)) 

    model = load_trained_model('fps_optimization_model.h5')
    
    region = (0, 0, 1920, 1080) 
    frames_to_capture = 10  
    frame_count = 0
    optimization_needed = False

    for _ in range(50):  
        frame = capture_screen(region)  
        frame_count += 1

        processed_frame = preprocess_frame(frame)

        prediction = model.predict(np.expand_dims(processed_frame, axis=0))

        print(f"FPS Optimization Prediction: {prediction}")

        show_frame(frame)

        if frame_count >= frames_to_capture:
            prediction_value = prediction[0][0]  
            print(f"Prediction value: {prediction_value}")

            if prediction_value > 0.7:  
                print("⚠️ FPS optimization required!")
                optimization_needed = True
                break  

            frame_count = 0  

    if optimization_needed:
        print("Capture process stopped after detecting FPS optimization need.")
    else:
        print("Capture process completed without detecting FPS optimization need.")

if __name__ == "__main__":
    preprocess_and_train_model()