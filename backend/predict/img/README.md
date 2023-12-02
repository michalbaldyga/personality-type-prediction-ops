# Personality Type Prediction Using Images

This project includes a script for predicting personality types based on images using pre-trained deep learning models. The predictions are made using models trained on the ResNet50 architecture and TensorFlow/Keras.

## Overview

The script loads a pre-trained model, preprocesses an input image, and predicts the class probabilities of that image for each classification category (referred to as 'coins').

## Prerequisites

- Python 3.8 or later
- TensorFlow 2.x
- Keras
- NumPy

Ensure you have the above libraries installed in your Python environment before running the script.


## Prediction Process

The prediction process is executed by running the script which performs the following steps:

1. **Image Loading and Preprocessing**: Loads an input image, resizes it to the target size, and normalizes the pixel values.

2. **Model Prediction**: Uses the pre-trained model to predict class probabilities for the input image.

3. **Results Interpretation**: Interprets the predicted probabilities to determine the personality type for each coin category.

## Usage

To run the prediction script, navigate to the directory containing `predict.py` and execute:

```bash
python predict.py

