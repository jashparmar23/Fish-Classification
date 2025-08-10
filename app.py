import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- CONFIG ---
MODEL_DIR = "models"
MODEL_NAME = "your_model.h5"  # or SavedModel folder
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
DATASET_PATH = "images.cv_jzk6llhf18tm3k0kyttxz/data/test"
IMG_SIZE = (224, 224)

# Replace with your actual class names
CLASS_NAMES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    st.write(f"Model loaded. Number of inputs: {len(model.inputs)}")
    st.write(f"Model inputs shapes: {[inp.shape for inp in model.inputs]}")
    return model

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]  # drop alpha channel if present
    image_array = image_array / 255.0  # normalize to [0,1]
    return np.expand_dims(image_array, axis=0)

def predict(image: Image.Image, model):
    processed_img = preprocess_image(image)
    if len(model.inputs) == 1:
        preds = model.predict(processed_img)
    else:
        # Feed same input to all model inputs (adjust if you know otherwise)
        inputs = [processed_img for _ in range(len(model.inputs))]
        preds = model.predict(inputs)
    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]
    confidences = {cls: float(preds[0][i]) for i, cls in enumerate(CLASS_NAMES)}
    return predicted_class, confidences

def main():
    st.title("Fish Classification")
    model = load_model()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                predicted_class, confidences = predict(image, model)
            st.success(f"Prediction: {predicted_class}")
            st.write("Confidence Scores:")
            st.json(confidences)

if __name__ == "__main__":
    main()
