from pathlib import Path
import cv2
from .face_detector import FaceDetector
from .antispoof_detector import AntiSpoofDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACE_MODEL = PROJECT_ROOT / "models" / "yolov11n-face.pt"
SPOOF_MODEL = PROJECT_ROOT / "models" / "MiniFASNetV2_2.7_80x80.onnx"
INPUT_IMAGE = PROJECT_ROOT.parent.parent / "IO.jpg"


def main() -> None:
    """One-shot anti-spoof sanity check on a static image.

    Runs YOLO to find faces, then MiniFASNetV2 on each. Prints the
    liveness score and per-class probabilities. No file output —
    this is purely a "do the numbers look sane?" check before
    wiring anti-spoof into the live demo.
    """
    detector = FaceDetector(model_path=FACE_MODEL)
    spoof = AntiSpoofDetector(model_path=SPOOF_MODEL)

    image = cv2.imread(str(INPUT_IMAGE))
    faces = detector.detect(image)
    print(f"Detected {len(faces)} face(s)")

    for i, face in enumerate(faces):
        verdict = spoof.predict(image, face)
        p_live, p_print, p_replay = verdict.probs
        print(
            f" face{i}: {verdict.label}  score={verdict.score:.3f}  "
            f"(live={p_live:.3f} print={p_print:.3f} replay={p_replay:.3f})"
        )


if __name__ == "__main__":
    main()