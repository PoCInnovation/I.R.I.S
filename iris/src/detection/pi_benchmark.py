"""Headless FPS benchmark for the Raspberry Pi.

Runs face detection on N webcam frames, prints per-frame timing and a summary.
No GUI — so the same script works over SSH on a headless Pi. Run with:

    uv run python -m iris.src.detection.pi_benchmark

If the USB webcam isn't on /dev/video0, override with CAMERA_INDEX env var.
For a Pi Camera Module (CSI), this script won't work — picamera2 has a
different API and we'll switch to it once we know we need to.
"""
import os
import time
from pathlib import Path

import cv2

from .face_detector import FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov11n-face.pt"
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
N_FRAMES = 100


def main() -> None:
    print(f">> loading model from {MODEL_PATH}")
    detector = FaceDetector(model_path=MODEL_PATH, imgsz=192)

    print(f">> opening camera index {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warm-up: first inference is always slow (graph compile, allocator init).
    # Drop it from the measurement.
    print(">> warm-up frame...")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but read() failed")
    detector.detect(frame)

    print(f">> benchmarking {N_FRAMES} frames...")
    times = []
    face_counts = []
    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {i}: read failed")
            continue
        t0 = time.perf_counter()
        faces = detector.detect(frame)
        dt = time.perf_counter() - t0
        times.append(dt)
        face_counts.append(len(faces))

    cap.release()

    n = len(times)
    avg = sum(times) / n
    fastest = min(times)
    slowest = max(times)
    avg_faces = sum(face_counts) / n
    print()
    print(f"=== results over {n} frames ===")
    print(f"  avg     : {avg*1000:6.1f} ms   ({1/avg:5.1f} FPS)")
    print(f"  fastest : {fastest*1000:6.1f} ms   ({1/fastest:5.1f} FPS)")
    print(f"  slowest : {slowest*1000:6.1f} ms   ({1/slowest:5.1f} FPS)")
    print(f"  avg faces detected per frame: {avg_faces:.2f}")


if __name__ == "__main__":
    main()
