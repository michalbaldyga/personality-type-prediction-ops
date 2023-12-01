import os

import numpy as np
import tensorflow as tf
from keras.utils.image_utils import img_to_array, load_img
from PIL import Image

from backend.utils import CLASS_MAPPINGS


def load_and_preprocess_image(img_path, target_size=(224, 224), save_preprocessed=True,
                              save_path="preprocessed_image.jpg"):
    """Loads an image from a specified path and preprocesses it for model prediction.

    :param img_path: str, the path to the image file.
    :param target_size: tuple, the target size to which the image is resized. Default is (224, 224).
    :param save_preprocessed: bool, whether to save the preprocessed image. Default is False.
    :param save_path: str, the path where the preprocessed image will be saved. Default is 'preprocessed_image.jpg'.
    :return: np.ndarray, the preprocessed image array.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale by 1/255

    if save_preprocessed:
        # Convert array back to image
        processed_img = Image.fromarray((img_array[0] * 255).astype("uint8"))
        processed_img.save(save_path)

    return img_array


def predict_with_model(model, img_path):
    """Predicts class probabilities for an image using a given model.

    :param model: tf.keras.models.Model, the trained TensorFlow/Keras model used for prediction.
    :param img_path: str, the path to the image file for which the prediction is to be made.
    :return: np.ndarray, an array containing the probabilities of the image belonging to class 0 and class 1.
    """
    img_array = load_and_preprocess_image(img_path, save_preprocessed=True, save_path="preprocessed_image.jpg")
    probability_of_class_1 = model.predict(img_array)[0, 0]
    probability_of_class_0 = 1 - probability_of_class_1
    return np.array([probability_of_class_0, probability_of_class_1])


# Specify the path to the input image.
INPUT_IMG_DIRECTORY = os.path.join("..", "..", "..", "static", "img", "input")
img = os.path.join(INPUT_IMG_DIRECTORY, "image.jpg")

for coin in CLASS_MAPPINGS:
    model = tf.keras.models.load_model(f"model_{coin}.h5")
    class_probabilities = predict_with_model(model, img)

    # Interpret probabilities based on mapping
    label_0, label_1 = list(CLASS_MAPPINGS[coin].keys())
    probability_label_0 = class_probabilities[0] * 100
    probability_label_1 = class_probabilities[1] * 100

    print(f"{coin}: {label_0}: {probability_label_0:.2f}%, {label_1}: {probability_label_1:.2f}%")
