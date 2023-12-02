import os

import cv2
import numpy as np
import pandas as pd
import unidecode
from PIL import Image

# Base directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Directory for storing CSV files.
STATIC_CSV_DIR = os.path.join(BASE_DIR, "static", "csv")
RECORDS_CSV = os.path.join(STATIC_CSV_DIR, "records.csv")
RECORDS_CLEANED_CSV = os.path.join(STATIC_CSV_DIR, "records_cleaned.csv")
RECORDS_CLEANED_PROCESSED_CSV = os.path.join(STATIC_CSV_DIR, "records_cleaned_processed.csv")

MODEL_IMG_ACCURACY_DIR = os.path.join(STATIC_CSV_DIR, "model_img_accuracies.csv")

# Updated mapping of classes for each binary coin
CLASS_MAPPINGS = {
    "Human Needs_Observer": {"Oe": 0, "Oi": 1},
    "Human Needs_Decider": {"De": 0, "Di": 1},
    "Human Needs_Preferences": {"DD": 0, "OO": 1},
    "Letter_Observer": {"S": 0, "N": 1},
    "Letter_Decider": {"F": 0, "T": 1},
    "Animal_Energy Animal": {"Sleep": 0, "Play": 1},
    "Animal_Info Animal": {"Consume": 0, "Blast": 1},
    "Animal_Dominant Animal": {"Energy": 0, "Info": 1},
    "Animal_Introverted vs Extraverted": {"Extro": 0, "Intro": 1},
    "Sexual Modality_Sensory": {"M": 0, "F": 1},
    "Sexual Modality_Extraverted Decider": {"M": 0, "F": 1},
}
# Columns for different model training
COINS_COLUMNS = [
    "Human Needs_Observer", "Human Needs_Decider", "Human Needs_Preferences",
    "Letter_Observer", "Letter_Decider", "Animal_Energy Animal",
    "Animal_Info Animal", "Animal_Dominant Animal",
    "Animal_Introverted vs Extraverted", "Sexual Modality_Sensory",
    "Sexual Modality_Extraverted Decider",
]


def image_exists(row: pd.Series, directory: str) -> bool:
    """Check if an image file exists in a given directory based on a DataFrame row.

    :param row: pd.Series, a row from a DataFrame containing image names.
    :param directory: str, the directory path where the image files are located.
    :return: bool, True if the image file exists, False otherwise.
    """
    image_path = os.path.join(directory, row["name"] + ".jpg")
    return os.path.isfile(image_path)


def save_image(image: np.ndarray, filename: str, directory: str) -> None:
    """Save an image to a specified directory with a given filename.

    :param image: np.ndarray, the image to be saved.
    :param filename: str, the name of the file to save the image as.
    :param directory: str, the directory path where the image will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    cv2.imwrite(file_path, image)


def rename_to_ascii_img(directory: str) -> None:
    """Rename all image files in a directory to ASCII, replacing non-ASCII characters.

    :param directory: str, the directory path containing the image files.
    """
    ascii_limit = 128  # Maximum ordinal value for ASCII characters

    for filename in os.listdir(directory):
        if not all(ord(char) < ascii_limit for char in filename):
            ascii_filename = unidecode.unidecode(filename).replace(" ", "_")
            original_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, ascii_filename)
            os.rename(original_path, new_path)
            print(f"Renamed '{filename}' to '{ascii_filename}'")


def rename_to_ascii_csv(directory: str, delimiter: str) -> None:
    """Convert 'name' column entries in a CSV file to ASCII and save the changes.

    :param directory: str, the directory path of the CSV file.
    :param delimiter: str, delimiter used in CSV file.
    """
    df = pd.read_csv(directory, delimiter=delimiter)
    df["name"] = df["name"].apply(lambda name: unidecode.unidecode(name).replace(" ", "_"))
    df.to_csv(directory, index=False, sep=delimiter)


def find_image_extremes(directory: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the smallest and largest images in a directory.

    :param directory: str, the directory path containing the image files.
    :return: tuple[tuple[int, int], tuple[int, int]], the dimensions of the smallest and largest images.
    """
    min_dim = None
    max_dim = None
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            with Image.open(os.path.join(directory, filename)) as img:
                width, height = img.size
                if min_dim is None or (width * height < min_dim[0] * min_dim[1]):
                    min_dim = (width, height)
                if max_dim is None or (width * height > max_dim[0] * max_dim[1]):
                    max_dim = (width, height)
    return min_dim, max_dim


def replace_space_with_underscore_image(directory: str) -> None:
    """Rename image files by replacing spaces with underscores.

    :param directory: str, the directory containing the images.
    """
    for filename in os.listdir(directory):
        if " " in filename:
            new_filename = filename.replace(" ", "_")
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_filename}'")


def replace_space_with_underscore_csv(csv_file_path: str, delimiter: str) -> None:
    """Replace spaces with underscores in the 'name' column of a CSV file.

    :param csv_file_path: str, the path to the CSV file.
    :param delimiter: str, delimiter used in CSV file.
    """
    try:
        df = pd.read_csv(csv_file_path, delimiter=delimiter)
        if "name" in df.columns:
            df["name"] = df["name"].str.replace(" ", "_")
            df.to_csv(csv_file_path, index=False)
            print(f"CSV file '{csv_file_path}' has been modified and overwritten.")
        else:
            print("Column 'name' not found in the CSV file.")

    except Exception as e:
        print(f"An error occurred: {e}")
