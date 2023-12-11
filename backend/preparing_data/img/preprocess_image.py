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
INPUT = os.path.join(STATIC_IMG_DIR, "input")

# Initialize MediaPipe FaceMesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)


def resize_image(_image: np.ndarray, _filename: str, directory: str, size: int = 256) -> None:
    """Resize an image to a square format, maintaining aspect ratio, and save it.

    :param _image: np.ndarray, image to be resized.
    :param _filename: str, filename for the resized image.
    :param directory: str, directory to save the resized image.
    :param size: int, target size for the longest dimension of the image.
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

    :param _corrected_image: np.ndarray, image to crop.
    :param face_landmarks: Landmarks, facial landmarks for determining the crop area.
    :param w: int, width of the image.
    :param h: int, height of the image.
    :return: Union[np.ndarray, None], cropped image centered around the face or None if no landmarks are detected.
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

    :param _image: np.ndarray, image to be processed.
    :param _filename: str, filename for the processed image.
    :param corrected_directory: str, directory to save the image if a face is detected.
    :param no_face_directory: str, directory to save the image if no face is detected.
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


def draw_major_features_with_red_eyes_on_images(directory: str) -> None:
    """Draw major facial landmarks on images in a specified directory, with specific landmarks for eyes marked in red.

    :param directory: str, directory containing images to process.
    """
    # Define drawing specifications for general landmarks
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    # Define drawing specifications for the eyes in red
    eye_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))

    # Define the major features to draw (excluding eyes)
    major_features = [mp_face_mesh.FACEMESH_LIPS,
                      mp_face_mesh.FACEMESH_NOSE]

    for _filename in os.listdir(directory):
        if _filename.lower().endswith((".png", ".jpg", ".jpeg")):
            _file_path = os.path.join(directory, _filename)
            _image = cv2.imread(_file_path)
            if _image is None:
                continue

            results = face_mesh.process(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw major features
                    for feature in major_features:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image=_image,
                            landmark_list=face_landmarks,
                            connections=feature,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)

                    # Draw the specific landmarks for the left and right eyes in red
                    left_eye_landmark = face_landmarks.landmark[130]
                    cv2.circle(_image,
                               (int(left_eye_landmark.x * _image.shape[1]),
                                int(left_eye_landmark.y * _image.shape[0])),
                               eye_drawing_spec.circle_radius,
                               eye_drawing_spec.color,
                               eye_drawing_spec.thickness)

                    right_eye_landmark = face_landmarks.landmark[359]
                    cv2.circle(_image,
                               (int(right_eye_landmark.x * _image.shape[1]),
                                int(right_eye_landmark.y * _image.shape[0])),
                               eye_drawing_spec.circle_radius,
                               eye_drawing_spec.color,
                               eye_drawing_spec.thickness)

            # Save the modified image
            cv2.imwrite(os.path.join(directory, "major_features_red_eyes_" + _filename), _image)
            print(f"Processed and saved with major features and red eye landmarks on {_filename}")


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



