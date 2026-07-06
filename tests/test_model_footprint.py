"""Deterministic dependency-footprint guards for the Raspberry Pi target.

Unlike ``test_ram_budget.py`` (which measures live RSS and is therefore
sensitive to the host's torch build), these checks are static: they only
``stat`` a file and parse ``pyproject.toml``. They run in milliseconds, need
no model load and no installed packages, and give the same answer on x86 CI
as on the Pi. They complement -- not replace -- the RAM guard: the RAM test
catches runtime bloat, these catch it at commit time before it ever runs.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "iris" / "src" / "models" / "yolov11n-face.pt"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

# The committed yolov11n weights are ~5.3 MB. 12 MB leaves >2x headroom for a
# retrain/quantization tweak while still failing the instant someone swaps in
# a yolo small (~19 MB) or larger backbone -- the regression we care about.
MODEL_SIZE_CAP_MB = 12.0

# Heavy ML frameworks that must NOT appear as DIRECT runtime deps. torch and
# torchvision are expected transitively via ultralytics; pinning them directly
# would let a CUDA/heavier build slip in unnoticed. Names are PEP 503 normalized.
HEAVY_DIRECT_DEPS = {
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tensorflow-cpu",
    "tensorflow-gpu",
    "jax",
    "jaxlib",
    "paddlepaddle",
    "mxnet",
}


def _normalize(dist_name: str) -> str:
    """PEP 503 normalization: lowercase and collapse runs of [-_.] to a dash."""
    return re.sub(r"[-_.]+", "-", dist_name).lower()


def _requirement_name(spec: str) -> str:
    """Extract the distribution name from a PEP 508 requirement string.

    Splits off version markers, extras and environment markers, e.g.
    ``ultralytics>=8.4.48`` -> ``ultralytics``, ``foo[bar]==1.0`` -> ``foo``.
    """
    head = re.split(r"[<>=!~;\[\( ]", spec.strip(), maxsplit=1)[0]
    return _normalize(head)


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"YOLO weights not present at {MODEL_PATH}; skipping footprint guard",
)
def test_model_file_size_under_cap() -> None:
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < MODEL_SIZE_CAP_MB, (
        f"{MODEL_PATH.name} is {size_mb:.1f} MB, over the {MODEL_SIZE_CAP_MB} MB "
        f"cap. A nano model should stay well under this; a larger YOLO variant "
        f"(small/medium/large) would blow the Pi's memory and latency budget. "
        f"If this bump is intentional, raise MODEL_SIZE_CAP_MB deliberately."
    )


def test_no_heavy_direct_runtime_dependency() -> None:
    with PYPROJECT_PATH.open("rb") as fh:
        pyproject = tomllib.load(fh)

    deps = pyproject.get("project", {}).get("dependencies", [])
    direct = {_requirement_name(spec) for spec in deps}

    offenders = sorted(direct & HEAVY_DIRECT_DEPS)
    assert not offenders, (
        f"heavy ML framework(s) added as DIRECT runtime deps: {offenders}. "
        f"torch/torchvision must arrive transitively via ultralytics, not be "
        f"pinned directly (a direct pin can silently pull a CUDA/heavier build). "
        f"Remove them from [project].dependencies or, if truly required, update "
        f"HEAVY_DIRECT_DEPS in this test with a reasoned exception."
    )
