import streamlit as st
import os

def main():
    st.title("Fish Classification")

    # Show current directory and contents for debugging
    st.write("Current working directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())
    if os.path.exists("models_converted"):
        st.write("Files in models_converted folder:", os.listdir("models_converted"))
    else:
        st.error("models_converted folder not found!")

    # Now you can continue with loading models from models_converted/
    # For example:
    models = os.listdir("models_converted") if os.path.exists("models_converted") else []
    model_choice = st.selectbox("Choose your model", models)

    if model_choice:
        model_path = os.path.join("models_converted", model_choice)
        st.write(f"Loading model: {model_path}")
        # Add your model loading logic here

if __name__ == "__main__":
    main()
