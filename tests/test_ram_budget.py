"""RAM budget regression guard for the face detector.

The detector (YOLO weights + one inference) is the fixed memory cost the
Raspberry Pi 5 pays on every run, so we cap it. With the weights committed to
the repo this runs in CI; if the weights are ever removed it SKIPS rather than
failing spuriously.

Override the budget without editing code:

    IRIS_DETECTOR_RAM_BUDGET_MB=800 pytest tests/test_ram_budget.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import psutil
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "iris" / "src" / "models" / "yolov11n-face.pt"

# Budget for the detector's RSS delta (model load + warm-up inference).
# yolov11n is a nano model; ~600 MB leaves generous headroom on a 4 GB Pi 5
# while still catching a regression that pulls in a much heavier backend.
DETECTOR_RAM_BUDGET_MB = int(os.environ.get("IRIS_DETECTOR_RAM_BUDGET_MB", "600"))


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"YOLO weights not present at {MODEL_PATH}; skipping RAM guard",
)
def test_detector_ram_budget() -> None:
    from iris.src.detection.face_detector import FaceDetector

    proc = psutil.Process()
    baseline = proc.memory_info().rss

    detector = FaceDetector(model_path=MODEL_PATH, imgsz=320)
    # A blank frame is enough to force graph/allocator init -- we're measuring
    # memory, not detection quality, so no real face is required.
    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    used_mb = (proc.memory_info().rss - baseline) / (1024 * 1024)
    assert used_mb < DETECTOR_RAM_BUDGET_MB, (
        f"detector RSS delta {used_mb:.0f} MB exceeds budget "
        f"{DETECTOR_RAM_BUDGET_MB} MB"
    )
