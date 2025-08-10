import tensorflow as tf
from PIL import Image
import numpy as np

<<<<<<< HEAD
model_path = "models/EfficientNetB0_fish.h5"
model = tf.keras.models.load_model(model_path, compile=False)
model.summary()
=======
# -----------------------
# CONFIG
# -----------------------
MODEL_DIR = r"C:\Users\jashp\Downloads\Fish\models"
DATASET_PATH = r"C:\Users\jashp\Downloads\Fish\images.cv_jzk6llhf18tm3k0kyttxz\data\test"
IMG_SIZE = (224, 224)
>>>>>>> 1487631 (Save local changes before syncing with remote)

image = Image.open("path_to_a_sample_image.jpg")
image = image.resize((224, 224))
img_array = np.array(image)
if img_array.shape[-1] == 4:
    img_array = img_array[..., :3]
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
print(f"Input shape: {img_array.shape}")

preds = model.predict(img_array)
print(preds)
