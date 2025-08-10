import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------
# CONFIG
# -----------------------
MODEL_DIR = "models"  # relative to repo root
DATASET_PATH = os.path.join("images.cv_jzk6llhf18tm3k0kyttxz", "data", "test")
IMG_SIZE = (224, 224)

# Automatically detect class names from test dataset folders
try:
    CLASS_NAMES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
except FileNotFoundError:
    st.error(f"Error: The dataset path '{DATASET_PATH}' was not found. Please ensure your folder structure is correct.")
    st.stop()

# -----------------------
# HELPER FUNCTIONS
# -----------------------
@st.cache_resource
def load_model_cached(model_path):
    # Add your custom layers or functions here if needed, e.g.:
    # from your_custom_module import CustomLayer
    # custom_objects = {"CustomLayer": CustomLayer}
    custom_objects = {}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array / 255.0

def predict(image, model):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence_scores = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    return predicted_class, confidence_scores

# -----------------------
# UI LAYOUT
# -----------------------
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Fish Image Classifier")

# Find all model files
try:
    MODEL_FILES = [f for f in os.listdir(MODEL_DIR) if f.endswith((".h5", ".keras"))]
    if not MODEL_FILES:
        st.error(f"No model files found in the '{MODEL_DIR}' directory.")
        st.stop()
except FileNotFoundError:
    st.error(f"The model directory '{MODEL_DIR}' was not found. Please ensure it exists.")
    st.stop()

# -----------------------
# MODEL SELECTION
# -----------------------
model_choice = st.selectbox("Select Model:", MODEL_FILES)
model = load_model_cached(os.path.join(MODEL_DIR, model_choice))

# -----------------------
# MODE SELECTOR
# -----------------------
mode = st.radio("Choose Mode:", ["📤 Upload Image", "📂 Browse Dataset"])

if mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Classifying...")
        predicted_class, confidence_scores = predict(image, model)
        st.success(f"Predicted Class: **{predicted_class}**")

        st.write("### Confidence Scores")
        for cls, score in confidence_scores.items():
            st.write(f"{cls}: {score:.2%}")

elif mode == "📂 Browse Dataset":
    all_images = []
    for root, _, files in os.walk(DATASET_PATH):
        for f in files:
            if f.lower().endswith(('jpg', 'jpeg', 'png')):
                all_images.append(os.path.join(root, f))

    if all_images:
        selected_image_path = st.selectbox("Select an image:", all_images)
        image = Image.open(selected_image_path)
        st.image(image, caption=f"Selected: {os.path.basename(selected_image_path)}", use_column_width=True)

        st.write("Classifying...")
        predicted_class, confidence_scores = predict(image, model)
        st.success(f"Predicted Class: **{predicted_class}**")

        st.write("### Confidence Scores")
        for cls, score in confidence_scores.items():
            st.write(f"{cls}: {score:.2%}")
    else:
        st.warning("No images found in the dataset folder.")
