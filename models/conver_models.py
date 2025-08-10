import tensorflow as tf
import os

OLD_MODEL_DIR = "."  # current folder, i.e., models
NEW_MODEL_DIR = "../models_converted"  # save converted models in Fish\models_converted


os.makedirs(NEW_MODEL_DIR, exist_ok=True)

def convert_h5_to_keras(old_model_path, new_model_path):
    print(f"Loading model from: {old_model_path}")
    model = tf.keras.models.load_model(old_model_path, compile=False)
    print(f"Saving model to: {new_model_path}")
    model.save(new_model_path, save_format="keras")
    print("Conversion done!\n")

for filename in os.listdir(OLD_MODEL_DIR):
    if filename.endswith(".h5"):
        old_path = os.path.join(OLD_MODEL_DIR, filename)
        # Replace .h5 with .keras
        new_filename = filename.replace(".h5", ".keras")
        new_path = os.path.join(NEW_MODEL_DIR, new_filename)
        convert_h5_to_keras(old_path, new_path)
