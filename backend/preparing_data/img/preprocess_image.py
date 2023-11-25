import os
from math import atan2, degrees

import cv2
import mediapipe as mp
import numpy as np

from backend import utils

# Directory for storing different categories of images.
STATIC_IMG_DIR = os.path.join("..", "..", "..", "static", "img")
ORIGINAL_DIRECTORY = os.path.join(STATIC_IMG_DIR, "original")
RESIZED_DIRECTORY = os.path.join(STATIC_IMG_DIR, "resized")
CORRECTED_DIRECTORY = os.path.join(STATIC_IMG_DIR, "corrected_and_cropped")
NO_FACE_DIRECTORY = os.path.join(STATIC_IMG_DIR, "no_face")

# Initialize MediaPipe FaceMesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)


def resize_image(_image: np.ndarray, _filename: str, directory: str, size: int = 256) -> None:
    """Resize an image to a square format, maintaining aspect ratio, and save it.

    Args:
        _image: Image to be resized.
        _filename: Filename for the resized image.
        directory: Directory to save the resized image.
        size: Target size for the longest dimension of the image.

    This function resizes an image, adds padding to make it square, and saves it to the specified directory.
    """
    h, w = _image.shape[:2]
    if h > w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size

    __resized_image = cv2.resize(_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    __squared_image = cv2.copyMakeBorder(__resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                         value=color)
    utils.save_image(__squared_image, _filename, directory)


def crop_to_face(_corrected_image: np.ndarray, face_landmarks, w: int, h: int) -> np.ndarray | None:
    """Crop an image to the region containing the face based on facial landmarks.

    Args:
        _corrected_image: Image to crop.
        face_landmarks: Facial landmarks for determining the crop area.
        w: Width of the image.
        h: Height of the image.

    Returns:
        Cropped image centered around the face or None if no landmarks are detected.
    """
    landmark_points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.int32)

    if landmark_points.size > 0:
        bounding_box = cv2.boundingRect(landmark_points)
        x, y, bbox_w, bbox_h = bounding_box
        __cropped_image = _corrected_image[y:y + bbox_h, x:x + bbox_w]
        return cv2.resize(__cropped_image, (224, 224))
    return None


def correct_orientation_and_crop(_image: np.ndarray, _filename: str, corrected_directory: str,
                                 no_face_directory: str) -> None:
    """Correct the orientation of an image and crop it to the face.

    Args:
        _image: Image to be processed.
        _filename: Filename for the processed image.
        corrected_directory: Directory to save the image if a face is detected.
        no_face_directory: Directory to save the image if no face is detected.

    This function corrects the orientation of the image based on facial landmarks, crops it to the face, and saves it.
    """
    results = face_mesh.process(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye = face_landmarks.landmark[130]
            right_eye = face_landmarks.landmark[359]

            eye_angle = atan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x)
            angle = degrees(eye_angle)
            forty_five = 45
            if angle < -forty_five or angle > forty_five:
                angle += 180

            h, w = _image.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

            __corrected_image = cv2.warpAffine(_image, m, (w, h))

            __cropped_image = crop_to_face(__corrected_image, face_landmarks, w, h)
            utils.save_image(__cropped_image, _filename, corrected_directory)
            return
    else:
        utils.save_image(_image, _filename, no_face_directory)


def draw_landmarks_on_images(directory: str) -> None:
    """Draw facial landmarks on images in a specified directory.

    Args:
        directory: Directory containing images to process.

    This function processes each image in the directory, adds facial landmarks using MediaPipe's FaceMesh, and saves the modified image.
    """
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    for _filename in os.listdir(directory):
        if _filename.lower().endswith((".png", ".jpg", ".jpeg")):
            _file_path = os.path.join(directory, filename)
            _image = cv2.imread(file_path)
            if _image is None:
                continue
            results = face_mesh.process(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

            # Save the modified image
            cv2.imwrite(os.path.join(directory, "landmarked_" + _filename), _image)
            print(f"Processed and saved landmarks on {_filename}")


for filename in os.listdir(ORIGINAL_DIRECTORY):
    if filename.lower().endswith(".jpg"):
        file_path = os.path.join(ORIGINAL_DIRECTORY, filename)

        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            continue

        if image is not None:
            resize_image(image, filename, RESIZED_DIRECTORY)

            resized_image_path = os.path.join(RESIZED_DIRECTORY, filename)
            resized_image = cv2.imread(resized_image_path)

            if resized_image is None:
                print(f"Failed to load resized image: {resized_image_path}")
                continue

            correct_orientation_and_crop(resized_image, filename, CORRECTED_DIRECTORY, NO_FACE_DIRECTORY)

            corrected_image_path = os.path.join(CORRECTED_DIRECTORY, filename)
            corrected_image = cv2.imread(corrected_image_path)

            if corrected_image is None:
                print(f"Failed to load corrected image: {corrected_image_path}")
                continue

face_mesh.close()
