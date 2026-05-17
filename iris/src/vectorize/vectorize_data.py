##
## POC PROJECT, 2026
## I.R.I.S
## File description:
## vectorize
##

import cv2
from pathlib import Path
from .image_to_vector import image_to_vector

def vectorize_data(image_paths: list[Path]) -> dict:

    data_base = {}

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Cannot read image : {path.name}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vector = image_to_vector(image_rgb)
        if len(vector) > 0:
            data_base[path.name] = vector[0]
            print(f"{path.name} vectorized : ({len(vector[0])})")
        else:
            print(f"No found faces in {path.name}")
    return data_base