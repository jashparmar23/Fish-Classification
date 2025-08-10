import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_DIR = "models"
DATASET_PATH = "images.cv_jzk6llhf18tm3k0kyttxz/data/test"
IMG_SIZE = (224, 224)

@st.cache_resource(show_spinner=True)
def load_model(model_filename):
    model_path = os.path.join(MODEL_DIR, model_filename)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:  # if PNG with alpha channel
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    st.title("Fish Classification")

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    model_choice = st.selectbox("Select model", model_files)
    model = load_model(model_choice)

    uploaded_file = st.file_uploader("Upload a fish image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_array = preprocess_image(image)
        preds = model.predict(input_array)
        class_idx = np.argmax(preds, axis=1)[0]
        
        # You need your class names list here
        class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
        st.write(f"Predicted class: {class_names[class_idx]}")

if __name__ == "__main__":
    main()
