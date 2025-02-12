import os
import cv2
import pyautogui
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_images
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from frame_preprocessing import preprocess_frame
from model_loading import load_trained_model

st.title("🎮 FPS Optimization Detection using Deep Learning")

# Dataset path
data_dir = "dataset/"
model_path = "fps_optimization_model.h5"

# Ensure the model exists before loading
if not os.path.exists(model_path):
    st.warning(f"⚠️ Model file '{model_path}' not found! Training a new model...")
    model = build_model()
    model.save(model_path)  # Save the trained model
    st.success("✅ Model trained and saved successfully!")

# Load the trained model
model = load_trained_model(model_path)
st.sidebar.success("✅ Model loaded successfully!")

# Screen capture function
def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

# Show captured frame
def show_frame(frame):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    st.pyplot(fig)

# Predict FPS optimization requirement
def predict_fps_optimization(model, frame):
    frame_preprocessed = preprocess_frame(frame)
    frame_input = np.expand_dims(frame_preprocessed, axis=0)

    prediction = model.predict(frame_input)
    prediction_value = prediction[0][0]

    st.sidebar.write(f"Prediction value: {prediction_value:.4f}")

    if prediction_value > 0.7:
        st.error("⚠️ FPS optimization required!")
        return True
    else:
        st.success("✅ FPS is fine.")
        return False

# Sidebar settings
st.sidebar.header("Settings")
capture_frames = st.sidebar.slider("Number of Frames to Capture", min_value=1, max_value=50, value=10)

# Buttons for controlling FPS check
start_button = st.sidebar.button("▶️ Start FPS Optimization Check")
stop_button = st.sidebar.button("🛑 Stop Check")

# Session state to handle stopping
if "stop_check" not in st.session_state:
    st.session_state.stop_check = False

# If Stop button is clicked, update session state
if stop_button:
    st.session_state.stop_check = True

if start_button:
    st.session_state.stop_check = False  # Reset stop flag
    st.sidebar.write("📸 Capturing screen frames...")

    region = (0, 0, 1920, 1080)
    optimization_needed = False

    for _ in range(capture_frames):
        if st.session_state.stop_check:
            st.sidebar.warning("🛑 FPS check stopped by user!")
            break  # Stop loop if Stop button is clicked

        frame = capture_screen(region)
        show_frame(frame)

        if predict_fps_optimization(model=model, frame=frame):
            optimization_needed = True
            break  # Stop capture if FPS optimization is required

    if optimization_needed:
        st.sidebar.error("⚠️ Capture process stopped: FPS optimization required!")
    elif not st.session_state.stop_check:
        st.sidebar.success("✅ Capture process completed: No FPS optimization needed.")

st.sidebar.info("Click '▶️ Start' to begin or '🛑 Stop' to stop the process.")