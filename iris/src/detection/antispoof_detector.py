"""Anti-spoofing detector using the Silent-Face-Anti-Spoofing model from Mininglamp Vision Technology Co., Ltd.
@misc{minivisionai2020silentantispoofing,
  title  = {Silent-Face-Anti-Spoofing},
  author = {Mininglamp Vision Technology Co., Ltd.},
  year   = {2020},
  howpublished = {\\url{https://github.com/minivision-ai/Silent-Face-Anti-Spoofing}},
}
"""




from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from iris.src.detection.face_detector import Face

@dataclass(frozen=True, slots=True)
class SpoofVerdict:
    """One anti-spoofing verdict for a detected face.
    'score' is the model's liveness probability, between 0 and 1: higher means more likely real.
    probs keeps the full output of the model, which may be useful for debugging or future improvements.
    """

    label: str
    score: float
    probs: tuple[float, float, float]

class AntiSpoofDetector:
    """ Thin wrapper around an ONNX anti-spoofing model. Takes a cropped face image and returns a `SpoofVerdict`.
    
    Model trained on face crops that include ~2.7x the bbox area as context, feeding a tight crop tanks accuracy.
    This class expands the bbox to  square of side 2.7x the max dimension of the original bbox, centered on the original bbox, and clips to the image borders.
    """

    INPUT_SIZE = 80
    CROP_SCALE = 2.7

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        """
        Args:
            model_path: path to the .onnx anti-spoofing model file.
            threshold: minimum confidence to consider a face a spoof. The model outputs a score between 0 and 1, but the optimal threshold may not be 0.5. Tune this on your dataset if you can.
        """

        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def expanded_crop(self,frame: np.ndarray, face: Face) -> np.ndarray:
        """
        Return a square crop of `frame` covering ~2.7x the face bbox area.
        """
        h, w = frame.shape[:2]
        cx = (face.x1 + face.x2) / 2
        cy = (face.y1 + face.y2) / 2
        side = max(face.width, face.height) * self.CROP_SCALE
        half = side / 2
        # Clamp to frame edges; faces near borders just get a smaller crop.
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))

        return frame[y1:y2, x1:x2]
    
    def predict(self, frame: np.ndarray, face: Face) -> SpoofVerdict:
        """
        Run anti-spoofing inference on the given face crop and return a `SpoofVerdict`.
        """
        crop = self.expanded_crop(frame, face)
        resized = cv2.resize(crop, (self.INPUT_SIZE, self.INPUT_SIZE))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        logits = self.session.run(None, {self.input_name: blob[None]})[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = tuple(exp / exp.sum())
        p_live, p_print, p_replay = float(probs[0]), float(probs[1]), float(probs[2])
        score = p_live  # Liveness probability: higher = more likely real.
        label = "real" if score >= self.threshold else "spoof"
        return SpoofVerdict(label=label, score=score, probs=probs)