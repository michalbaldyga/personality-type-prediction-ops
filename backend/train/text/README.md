# Text-Based Personality Type Prediction Model Training

This project involves developing and training text-based models to predict personality types. The models utilize the Transformer architecture with a focus on sequence classification.

## Overview

The script processes textual data, balances the dataset, and trains a model for each personality type classification using various hyperparameters.

## Prerequisites

- python 3.10
- torch
- transformers
- evaluate
- datasets
- pandas
- numpy
- tqdm

Ensure these libraries are installed before running the script.

## Training Process

1. **Load dataset**: The process begins by loading CSV files containing the text data and personality type labels. 

2. **Preprocess data**: In this step, the raw text data undergoes several preprocessing activities. It includes tokenizing the text, balancing the dataset for each personality type ('coin'), and converting text into a format suitable for the model.

3. **Train model**: This phase involves the core activity of training the models. For each personality type, a model is trained with various combinations of hyperparameters.

4. **Evaluate model**: After training, each model is evaluated to measure its performance. 

## Usage

To execute the training script, navigate to the directory containing `train.py` and run:

```bash
python train.py