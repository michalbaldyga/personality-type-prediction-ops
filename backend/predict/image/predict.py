import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image


# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale by 1/255
    return img_array


# Function to predict the class and its confidence
def predict_with_model(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = int(prediction[0, 0] > 0.5)
    confidence = prediction[0, 0]
    return predicted_class, confidence


# Columns for different model prediction
coins_columns = [
    "Human Needs_Observer", "Human Needs_Decider", "Human Needs_Preferences",
    "Letter_Observer", "Letter_Decider", "Animal_Energy Animal",
    "Animal_Info Animal", "Animal_Dominant Animal",
    "Animal_Introverted vs Extraverted", "Sexual Modality_Sensory",
    "Sexual Modality_Extraverted Decider",
]

# Path to your new image
INPUT_IMG_DIRECTORY = os.path.join("..", "..", "..", "static", "img", "input")
img = os.path.join(INPUT_IMG_DIRECTORY, "image.jpg")
# Load models and make predictions on the new image
for coin in coins_columns:
    model = tf.keras.models.load_model(f"model_{coin}.h5")
    predicted_class, confidence = predict_with_model(model, img)
    print(f"{coin}: Class {predicted_class} with {confidence * 100:.2f}% confidence")
