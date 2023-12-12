# Personality Type Prediction Using Images

This project includes a script for predicting personality types based on text using pre-trained deep learning models. The predictions are made using models trained on the `Transformer` architecture.

## Overview

The script loads a pre-trained model, preprocesses an input text, and predicts the class probabilities of that text for each classification category (referred to as 'coins').

## Prerequisites

- python 3.10
- transformers

Ensure you have the above libraries installed in your Python environment before running the script.


## Prediction Process

1. **Load model**: The process initiates by loading the specific pre-trained text classification model for each personality type category ('coin') from the model directory.

2. **Preprocess data**: The input text is segmented into batches of 512 tokens to accommodate the model's processing limits.

3. **Prediction**: The model then predicts sentiments for each batch, aggregating and normalizing these predictions across each 'coin' to produce final sentiment scores and percentages.

