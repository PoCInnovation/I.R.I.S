import time
from pathlib import Path

import cv2

from .face_detector import FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov11n-face.pt"


def main() -> None:
    print(">> loading model...")
    detector = FaceDetector(model_path=MODEL_PATH)
    print(">> opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam (index 0)")
    fps_smoothed = 0.0
    frame_n = 0
    try:
        while True:
            frame_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam")
                break

            faces = detector.detect(frame)
            frame_n += 1
            if frame_n % 10 == 0:
                print(f">> frame {frame_n}  faces={len(faces)}  fps={fps_smoothed:.1f}")

            for face in faces:
                cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"{face.confidence:.2f}",
                            (face.x1, face.y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            elapsed = time.perf_counter() - frame_start
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps
            cv2.putText(frame, f"FPS: {fps_smoothed:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("I.R.I.S", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
