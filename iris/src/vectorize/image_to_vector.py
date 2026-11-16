##
## POC PROJECT, 2026
## I.R.I.S
## File description:
## image to vector
##

import face_recognition
import numpy as np

def image_to_vector(face_image: np.ndarray) -> list[np.ndarray]:

    h, w, _ = face_image.shape

    face = [(0, w, h, 0)]
    encodages = face_recognition.face_encodings(face_image, known_face_locations=face)
    if len(encodages) == 0:
        return []
    return encodages

