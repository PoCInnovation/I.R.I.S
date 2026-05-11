from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

@dataclass(frozen=True, slots=True)
class Face:
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
    def __init__(
            self,
            model_path: Path,
            imgsz: int = 320,
            device: str = "cpu",
            conf: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.device = device
        self.conf = conf
    
    def detect(self, image: np.ndarray) -> list[Face]:
        results = self.model(
            image,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            conf=self.conf,
        )
        faces: list[Face] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            faces.append(
                Face(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
                    crop=image[y1:y2, x1:x2].copy(),
                )
            )
        return faces