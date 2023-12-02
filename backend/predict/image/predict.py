import os
import numpy as np
import tensorflow as tf
from keras.utils.image_utils import img_to_array, load_img
from PIL import Image

from backend.utils import CLASS_MAPPINGS


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale by 1/255
    return img_array


def predict_with_model(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    return model.predict(img_array)[0]


# Specify the path to the input image.
input_img_directory = os.path.join("..", "..", "..", "static", "img", "input")
img_path = os.path.join(input_img_directory, "image.jpg")  # Change 'image.jpg' to your image filename
model_directory = "../../release/img/"
# Initialize an empty dictionary to store the results.
results = {}

for coin, mappings in CLASS_MAPPINGS.items():
    model_path = os.path.join(model_directory, f"model_{coin}.h5")
    model = tf.keras.models.load_model(model_path)
    class_probabilities = predict_with_model(model, img_path)

    # Assuming the first class in mappings is your positive class
    positive_class_label = list(mappings.keys())[0]
    negative_class_label = list(mappings.keys())[1]  # Assuming only two classes

    probability_of_A = class_probabilities[0]
    probability_of_B = 1 - probability_of_A

    coin_results = [
        {'label': positive_class_label, 'percent': f"{probability_of_A * 100:.1f}%"},
        {'label': negative_class_label, 'percent': f"{probability_of_B * 100:.1f}%"}
    ]

    results[coin] = coin_results
print(results)
