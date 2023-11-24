import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


# THE CODE BELOW IS THE OLD CODE
def histogram_analysis(image1, image2):
    # Convert images to grayscale for histogram comparison
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms for each grayscale image
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)

    # Compare the histograms
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def find_best_sharpening_parameters(image):
    best_density = calculate_edge_density(image)
    best_params = (3, 1.5, -0.5)  # Default parameters

    for kernel in range(3, 8, 2):  # Kernel sizes: 3, 5, 7
        for alpha in np.arange(1.0, 2.0, 0.1):  # Alpha values from 1.0 to 2.0
            for beta in np.arange(-1.0, 0.0, 0.1):  # Beta values from -1.0 to 0.0
                sharpened = sharpen_image(image, kernel, alpha, beta)
                density = calculate_edge_density(sharpened)

                if density > best_density:
                    best_density = density
                    best_params = (kernel, alpha, beta)

    return best_params


def calculate_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges) / (edges.shape[0] * edges.shape[1])


def sharpen_image(image, gaussian_kernel, alpha, beta):
    blurred = cv2.GaussianBlur(image, (gaussian_kernel, gaussian_kernel), 0)
    return cv2.addWeighted(image, alpha, blurred, beta, 0)


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def normalize_landmarks(landmarks_array, left_eye_index=133, right_eye_index=362):
    nose_tip_index = 4
    reference_point = landmarks_array[nose_tip_index]

    translated_landmarks = landmarks_array - reference_point
    eye_distance = np.linalg.norm(landmarks_array[left_eye_index] - landmarks_array[right_eye_index])
    return translated_landmarks / eye_distance


def process_and_normalize_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None  # If the image is not found or cannot be opened

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None  # If no face is detected

    landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))

    landmarks_array = np.array(landmarks)
    return normalize_landmarks(landmarks_array)


# Load OPS data from CSV
ops_data = pd.read_csv("..\\..\\..\\static\\csv\\records_update_cleaned_processed.csv")
image_directory = "..\\..\\..\\static\\improved_quality_img"

# Columns (coins) to include from the OPS data
coins_columns = [
    "Human Needs_Observer", "Human Needs_Decider", "Human Needs_Preferences",
    "Letter_Observer", "Letter_Decider", "Animal_Energy Animal",
    "Animal_Info Animal", "Animal_Dominant Animal",
    "Animal_Introverted vs Extraverted", "Sexual Modality_Sensory",
    "Sexual Modality_Extraverted Decider",
]

landmarks_data = []

for index, row in ops_data.iterrows():
    image_name = row["name"]
    image_path = os.path.join(image_directory, image_name + ".jpg")  # Adjust file extension if needed
    landmarks = process_and_normalize_image(image_path)

    if landmarks is not None:
        # Append coins and landmarks to the list
        landmarks_data.append([image_name] + [row[col] for col in coins_columns] + list(landmarks.flatten()))

# Column names for the new DataFrame
new_columns = ["name"] + coins_columns + [f"landmark_{i}" for i in
                                          range(len(landmarks_data[0]) - len(coins_columns) - 1)]

# Create a DataFrame from the landmarks data
landmarks_df = pd.DataFrame(landmarks_data, columns=new_columns)

# Saving landmarks data to a new CSV file
landmarks_df.to_csv("landmarks_with_coins.csv", index=False)
