import streamlit as st
import tensorflow as tf
import os

MODELS_DIR = "models_converted"

def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def main():
    st.title("Fish Classification")

    model_names = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    model_choice = st.selectbox("Choose Model", model_names)

    if model_choice:
        try:
            model = load_model(model_choice)
            st.success(f"Loaded model: {model_choice}")
            st.text(model.summary())
        except Exception as e:
            st.error(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
