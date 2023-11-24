import os

import cv2
import numpy as np
from gfpgan import GFPGANer

from backend import utils

# Directory for storing different categories of images.
STATIC_IMG_DIR = os.path.join("..", "..", "..", "static", "img")
CORRECTED_DIRECTORY = os.path.join(STATIC_IMG_DIR, "corrected_and_cropped")
IMPROVED_QUALITY_DIRECTORY = os.path.join(STATIC_IMG_DIR, "improved_quality")


def improve_quality(_image: np.ndarray, _filename: str, directory: str) -> None:
    """Enhance the quality of an image using the GFPGAN model and save it to the specified directory.

    :param _image: np.ndarray, the image to be enhanced.
    :param _filename: str, the filename to save the enhanced image as.
    :param directory: str, the directory path where the enhanced image will be saved.
    """
    restorer = GFPGANer(model_path="experiments/pretrained_models/GFPGANv1.3.pth", upscale=1, arch="clean",
                        channel_multiplier=2, bg_upsampler=None)
    _, _, output = restorer.enhance(_image, has_aligned=False, only_center_face=False, paste_back=True)
    utils.save_image(output, _filename, directory)


for filename in os.listdir(CORRECTED_DIRECTORY):
    if filename.lower().endswith(".jpg"):
        file_path = os.path.join(CORRECTED_DIRECTORY, filename)
        image = cv2.imread(file_path)
        if image is not None:
            improve_quality(image, filename, IMPROVED_QUALITY_DIRECTORY)
        else:
            print(f"Failed to process image {file_path}")
