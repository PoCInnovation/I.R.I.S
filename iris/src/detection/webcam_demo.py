import time
from pathlib import Path

import cv2
from .face_detector import FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov11n-face.pt"
CAPTURES_DIR = PROJECT_ROOT.parent / "captures"


def main() -> None:
    """Live webcam demo with corner-bracket overlay and on-demand face capture.

    Controls:
        s  - save each detected face crop to CAPTURES_DIR
        q  - quit
    """
    print(">> loading model...")
    detector = FaceDetector(model_path=MODEL_PATH, imgsz=192)
    print(">> opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam (index 0)")
    # Keep at most 1 frame queued in the driver — read() then always returns the
    # newest frame instead of one from N processing-cycles ago.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Exponential moving average of FPS — raw per-frame FPS jumps around a lot,
    # smoothing makes the on-screen number readable.
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

            # Snapshot the clean crops BEFORE drawing anything on `frame`, so
            # saved images don't have HUD overlays baked into them.
            crops = [frame[f.y1:f.y2, f.x1:f.x2].copy() for f in faces]

            # Corner-bracket HUD: 8 short lines per face — one horizontal arm
            # and one vertical arm at each of the 4 corners, all extending
            # inward from the corner point.
            for face in faces:
                w = face.x2 - face.x1
                h = face.y2 - face.y1
                L = min(w, h) // 4  # arm length, scales with face size
                color = (0, 0, 240)  # BGR — red
                thick = 2
                # top-left
                cv2.line(frame, (face.x1, face.y1), (face.x1 + L, face.y1), color, thick)
                cv2.line(frame, (face.x1, face.y1), (face.x1, face.y1 + L), color, thick)
                # top-right
                cv2.line(frame, (face.x2, face.y1), (face.x2 - L, face.y1), color, thick)
                cv2.line(frame, (face.x2, face.y1), (face.x2, face.y1 + L), color, thick)
                # bottom-left
                cv2.line(frame, (face.x1, face.y2), (face.x1 + L, face.y2), color, thick)
                cv2.line(frame, (face.x1, face.y2), (face.x1, face.y2 - L), color, thick)
                # bottom-right
                cv2.line(frame, (face.x2, face.y2), (face.x2 - L, face.y2), color, thick)
                cv2.line(frame, (face.x2, face.y2), (face.x2, face.y2 - L), color, thick)
                cv2.putText(frame, f"{face.confidence:.2f}", (face.x1, face.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elapsed = time.perf_counter() - frame_start
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            # 90% old value + 10% new = smooth but still responsive
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps
            cv2.putText(frame, f"FPS: {fps_smoothed:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 0), 2)
            cv2.imshow("I.R.I.S", frame)

            # waitKey both pumps the GUI event loop (mandatory for imshow) and
            # returns the pressed key. Call it once per frame and branch on the
            # result — calling it twice would eat half the keypresses.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = int(time.time())
                for i, crop in enumerate(crops):
                    if crop.size == 0:  # zero-area bbox at a frame edge
                        continue
                    path = CAPTURES_DIR / f"face_{ts}_{i}.jpg"
                    cv2.imwrite(str(path), crop)
                    print(f">> saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
