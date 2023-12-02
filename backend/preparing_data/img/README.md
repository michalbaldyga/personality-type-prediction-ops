# Preparing Data - Image Processing

This directory, contains scripts for advanced image processing as part of the larger project. These scripts focus on image resizing, orientation correction, facial landmark detection, cropping to focus on faces, and enhancing image quality.

## Scripts Overview
### 1. Image Processing and Facial Landmark Detection (`preprocess_image.py`)

This script handles several image processing tasks:
- Resizes images while maintaining aspect ratio.
- Corrects image orientation based on detected facial landmarks.
- Crops images to focus primarily on the face.
- Optionally, annotates images with facial landmarks.

#### Key Features:
- Employs OpenCV and MediaPipe for image processing and landmark detection.
- Accepts `.jpg` image formats.
- Categorizes processed images based on face detection into appropriate directories.
- 
### 2. Image Quality Enhancement (`improve_quality.py`)

This script enhances the quality of images using the GFPGAN model. It processes images from the `corrected_and_cropped` directory, enhances their quality, and saves them in the `improved_quality` directory.

#### Key Features:
- Utilizes GFPGAN for image quality enhancement.
- Targets images in `.jpg` format.
- Outputs enhanced images to a designated directory.


## Installation

Ensure the following dependencies are installed:
- OpenCV
- MediaPipe
- GFPGAN
- NumPy

Install these using pip:
```bash
pip install opencv-python mediapipe gfpgan numpy
