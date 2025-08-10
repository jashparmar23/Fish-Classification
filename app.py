import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models"  # Use your actual path

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model expects {len(model.inputs)} inputs")

image_arr = preprocess_image("path_to_your_image.jpg")

if len(model.inputs) == 1:
    preds = model.predict(image_arr)
else:
    preds = model.predict([image_arr]*len(model.inputs))

print(preds)
