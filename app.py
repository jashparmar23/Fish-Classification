import streamlit as st
import os
import tensorflow as tf

def load_model(model_filename):
    model_path = os.path.join("models_converted", model_filename)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def main():
    st.title("Fish Classification")

    # Get list of models
    try:
        model_files = os.listdir("models_converted")
        model_files = [f for f in model_files if f.endswith(".keras")]
    except FileNotFoundError:
        st.error("Models folder not found!")
        return

    if not model_files:
        st.warning("No models found in 'models_converted' folder.")
        return

    # Dropdown to select model
    model_choice = st.selectbox("Choose your model:", model_files)

    if model_choice:
        st.write(f"Loading model: models_converted/{model_choice}")
        model = load_model(model_choice)
        st.success("Model loaded successfully!")

        # Add your prediction code here (e.g., upload image, preprocess, predict...)

if __name__ == "__main__":
    main()
