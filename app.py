import tensorflow as tf
import os
import numpy as np
from PIL import Image
import streamlit as st

def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

def preprocess_image(image: Image.Image, target_size=(224,224)):
    image = image.resize(target_size)
    image = np.array(image)
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[..., :3]
    image = image / 255.0  # normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

def main():
    st.title("Fish Classification")

    model_files = os.listdir("models_converted")
    model_choice = st.selectbox("Choose your model", model_files)
    if model_choice:
        model_path = os.path.join("models_converted", model_choice)
        model = load_model(model_path)
        st.write(f"Loaded model: {model_choice}")

        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            input_arr = preprocess_image(image, target_size=(224,224))  # or your model's expected size
            preds = model.predict(input_arr)
            st.write(f"Prediction output: {preds}")

if __name__ == "__main__":
    main()
