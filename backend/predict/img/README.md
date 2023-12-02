# Coin Classification System

## Overview
This Python script is designed to classify images of coins using pre-trained models. It processes an input image, applies a neural network model for each coin type, and outputs the probability of each class within that coin type. 

## Dependencies
- Python
- TensorFlow
- Keras
- NumPy
- Pillow

## Setup
1. Ensure you have the required dependencies installed.
2. Place your input images in the designated directory.

## Usage
- Update the `img_path` variable in the script with the filename of your image.
- Run the script to get classification results for each coin type in the image.

## How It Works
1. **Image Preprocessing**: The script loads an image, resizes it, and normalizes the pixel values.
2. **Model Prediction**: It loads a pre-trained Keras model for each coin type and predicts the class probabilities.
3. **Output**: The script outputs a dictionary with the predicted probabilities for each class within each coin type.

## Key Functions
- `load_and_preprocess_image`: Loads and preprocesses the image.
- `predict_with_model`: Predicts class probabilities using a given model and image.

## Customization
- Modify `CLASS_MAPPINGS` to adjust the classes and indices according to your model.
- Change the target size in `load_and_preprocess_image` based on your model's input requirements.

## Notes
- Models named as `model_{coin}.h5` are expected for each coin type.
- The script assumes a specific directory structure for storing images and models. Adjust these paths as needed.

## Disclaimer
- This script is designed for educational purposes and may require modifications for production use.
