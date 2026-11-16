from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

@dataclass(frozen=True, slots=True)
class Face:
    """One detected face in an image.

    Coordinates are in pixels, in OpenCV's top-left origin convention:
    (x1, y1) is the top-left corner of the bounding box, (x2, y2) the bottom-right.
    `crop` is an independent copy of the pixels inside the box so it survives
    even after the source frame is overwritten by the next webcam read.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    crop: np.ndarray

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class FaceDetector:
    """Thin wrapper around an Ultralytics YOLO face model.

    Hides the YOLO output format (a torch tensor tree) and returns plain
    `Face` dataclasses so the rest of the code never has to know about
    ultralytics or torch.
    """

    def __init__(
            self,
            model_path: Path,
            imgsz: int = 320,
            device: str = "cpu",
            conf: float = 0.5,
    ) -> None:
        """
        Args:
            model_path: path to the .pt YOLO weights file.
            imgsz: side length the model resizes inputs to before inference.
                Smaller = faster but worse on small/distant faces. 320 is a
                cheap default that still works well at webcam range.
            device: "cpu" or "cuda". The PoC targets a Raspberry Pi, so CPU.
            conf: minimum confidence to keep a detection. Anything below this
                is silently dropped by YOLO before we ever see it.
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.device = device
        self.conf = conf

    def detect(self, image: np.ndarray) -> list[Face]:
        """Run the model on one BGR image and return all faces above `self.conf`.

        Returns an empty list if no face passes the threshold. Never raises.
        """
        results = self.model(
            image,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,  # ultralytics prints per-call inference stats by default; mute it
            conf=self.conf,
        )
        faces: list[Face] = []
        # results is a list (one entry per input image). We always pass a single
        # image, so we only ever care about results[0].
        for box in results[0].boxes:
            # box.xyxy is a (1, 4) tensor of floats; convert to a python int 4-tuple
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            faces.append(
                Face(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
                    # .copy() detaches the crop from the source frame buffer —
                    # without it, the next cap.read() would mutate our pixels.
                    crop=image[y1:y2, x1:x2].copy(),
                )
            )
        return faces