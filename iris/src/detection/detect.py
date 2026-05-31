from pathlib import Path
import cv2
from .face_detector import FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov11n-face.pt"
INPUT_IMAGE = PROJECT_ROOT.parent.parent / "test.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output"

def main() -> None:
    """One-shot detection on a static image.

    Reads INPUT_IMAGE, runs the detector, then writes:
      - one cropped JPEG per face (faceN.jpg)
      - one annotated copy of the full image with boxes drawn (annotated.jpg)
    Used as a sanity check for the model before wiring it into the live demo.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(model_path=MODEL_PATH)
    image = cv2.imread(str(INPUT_IMAGE))
    faces = detector.detect(image)

    print(f"Detected {len(faces)} faces(s)")
    for i, face in enumerate(faces):
        print(f" face{i}: {face.width}x{face.height} conf={face.confidence:.2f}")

        # Save the clean crop first, before we draw on `image` below.
        crop_path = OUTPUT_DIR / f"face{i}.jpg"
        cv2.imwrite(str(crop_path), face.crop)

        cv2.rectangle(image, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"face {face.confidence:.2f}",
            (face.x1, face.y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    annotated_path = OUTPUT_DIR / "annotated.jpg"
    cv2.imwrite(str(annotated_path), image)
    print(f"Wrote anon.{annotated_path}")

if __name__ == "__main__":
    main()