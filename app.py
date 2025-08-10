import os
import streamlit as st

MODELS_DIR = "models_converted"

def main():
    st.title("Fish Classification")

    if not os.path.exists(MODELS_DIR):
        st.error(f"Models directory '{MODELS_DIR}' does not exist!")
        return

    model_names = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    st.write("Found models:", model_names)  # Debug print the models list

    if not model_names:
        st.warning("No models found in the models_converted folder.")
        return

    model_choice = st.selectbox("Choose Model", model_names)

    st.write(f"You selected: {model_choice}")  # Debug: show current selection

    # ... rest of your loading and prediction code

if __name__ == "__main__":
    main()
