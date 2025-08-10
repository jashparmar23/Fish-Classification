import streamlit as st
import tensorflow as tf
import os

# Path to converted models folder
MODELS_DIR = "models_converted"

def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    # Keras 3 supports loading from .keras files directly
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def main():
    st.title("Fish Classification")

    # List available models in models_converted folder
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
    model_choice = st.selectbox("Choose model", model_files)

    if model_choice:
        model = load_model(model_choice)
        st.success(f"Loaded model: {model_choice}")

        # Here you can add your inference logic (e.g., image upload and prediction)

if __name__ == "__main__":
    main()
