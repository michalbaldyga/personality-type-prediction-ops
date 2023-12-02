# Model Storage Directory

This directory is dedicated to storing the best-found models for the Personality Type Prediction project. Models are categorized based on the type of data they process: images and text.
The models can be found on[ **google drive (ref link)]()** and after downloading should be pasted inside the proper subdirectories.

## Directory Structure

The models are organized into two separate subdirectories:

- `img/`: Contains models trained for image-based personality type prediction.
- `text/`: Houses models trained for text-based personality type prediction.

## Accessing Models

Models within these directories have been saved after extensive training and validation. They represent the best-performing models in their respective categories (images and text).


### Image Models

Located under `personality-type-prediction-ops/backend/release/img/`, these models are specifically trained to analyze and predict personality types based on image data.

### Text Models

Located under `personality-type-prediction-ops/backend/release/text/`, these models are tailored for processing and predicting personality types based on textual data.

## Usage

To use these models, refer to the specific documentation or script within the project that handles model loading and inference. 
The models are used inside the `personality-type-prediction-ops/backend/predict/` directory.

## Note

Please do not modify or delete the contents of these directories unless you are updating the models with newer, better-performing versions.
