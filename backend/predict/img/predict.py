import os

import numpy as np
import tensorflow as tf
from keras.utils import img_to_array, load_img

from backend.utils import CLASS_MAPPINGS

# Specify the path to the input image.
input_img_directory = os.path.join("..", "..", "static", "img", "input")
img_path = os.path.join(input_img_directory, "image.jpg")  # Change 'image.jpg' to your image filename
model_directory = "../../backend/release/img/"


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load an image from a given path and preprocess it for model prediction.

    :param img_path: str, the path to the image file.
    :param target_size: tuple, the desired target size of the image as (width, height).
    :return: np.ndarray, the preprocessed image array suitable for model input.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array


def predict_with_model(model, img_path):
    """Predict the class probabilities of an image using a given model.

    :param model: TensorFlow/Keras model, the pre-trained model for prediction.
    :param img_path: str, the path to the image file.
    :return: list, the predicted class probabilities.
    """
    img_array = load_and_preprocess_image(img_path)
    return model.predict(img_array)[0]


def predict(image_url):
    """Predict the classification for a given image using multiple trained models.

    This function loads a series of trained models, each corresponding to a different coin type.
    For each model, it predicts the probability that the image belongs to one of two classes
    (e.g., heads or tails) defined in `CLASS_MAPPINGS`. The function then converts these
    probabilities into a user-friendly percentage format and organizes them into a structured
    dictionary, which is finally transformed into a desired output format.

    :param image_url: str, the URL of the image to be classified. This image is passed to
                      each model for prediction.
    :return: A dictionary where keys are coin types and values are lists of dictionaries,
             each containing the label (class name) and its corresponding probability in
             percentage format.
    """
    # Initialize an empty dictionary to store the results.
    results = {}

    for coin, mappings in CLASS_MAPPINGS.items():
        model_path = os.path.join(model_directory, f"model_{coin}.h5")
        model = tf.keras.models.load_model(model_path)
        class_probabilities = predict_with_model(model, image_url)

        # Assuming the first class in mappings is your positive class
        positive_class_label = list(mappings.keys())[0]
        negative_class_label = list(mappings.keys())[1]  # Assuming only two classes

        probability_of_a = class_probabilities[0]
        probability_of_b = 1 - probability_of_a

        coin_results = [
            {"label": positive_class_label, "percent": f"{probability_of_a * 100:.1f}"},
            {"label": negative_class_label, "percent": f"{probability_of_b * 100:.1f}"},
        ]

        results[coin] = coin_results

    final_results = __convert_results_to_proper_format(results)

    print(final_results)
    return final_results


def __convert_results_to_proper_format(input_results):
    output_results = []

    for coin in input_results:
        values = input_results[coin]

        if coin == "Sexual Modality_Sensory":
            percent_mas = values[0]["percent"]
            percent_fem = values[1]["percent"]
            output_results.append({"label": "Mas_S", "percent": percent_mas})
            output_results.append({"label": "Fem_S", "percent": percent_fem})
        elif coin == "Sexual Modality_Extraverted Decider":
            percent_mas = values[0]["percent"]
            percent_fem = values[1]["percent"]
            output_results.append({"label": "Mas_De", "percent": percent_mas})
            output_results.append({"label": "Fem_De", "percent": percent_fem})
        else:
            output_results.extend(values)

    return output_results
