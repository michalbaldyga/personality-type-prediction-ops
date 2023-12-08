# Personality Type Prediction Model Training

This project involves training deep learning models to predict personality types based on images. The models are trained using the ResNet50 architecture and TensorFlow/Keras.

## Overview

The script sets up TensorFlow's mixed precision policy for efficient training on GPU, prepares the data for training using image augmentation, balances the dataset using under sampling, and trains a model for each classification category using KFold cross-validation.

## Prerequisites

- Python 3.8 or later
- TensorFlow 2.x
- Keras
- imbalanced-learn
- NumPy
- Pandas
- Sci-kit Learn

Ensure you have the above libraries installed in your Python environment before running the script.


## Training Process

The training process is initiated by the `main()` function which performs the following steps:

1. **Set up the mixed precision policy**: Optimizes GPU performance by utilizing both 32-bit and 16-bit floating-point types during training.

2. **GPU configuration**: Configures TensorFlow to manage GPU memory growth.

3. **Load data**: Reads the processed CSV file containing image data and mappings.

4. **Class label mapping**: Maps class labels to numeric values for binary classification.

5. **Image data augmentation**: Uses `ImageDataGenerator` to augment images during training, which helps prevent overfitting.

6. **KFold cross-validation**: Splits the data into training and validation sets using 5-fold cross-validation to ensure that every data point is used for both training and validation.

7. **Model training**: Trains a model for each classification category (referred to as 'coins').

8. **Model saving**: Saves the best-performing model based on validation accuracy.

9. **Results recording**: Stores the accuracy of each model in a CSV file for later analysis.

## Usage

To run the training script, navigate to the directory containing `train.py` and execute:

```bash
python train.py

