import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict(model, img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction), np.max(prediction)