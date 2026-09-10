"""Directory constants for the Fashion-MNIST study: artifacts/, logs/, the pipeline
model package and the data root. Scripts import this instead of building paths
relative to themselves.
"""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts")
LOGS = os.path.join(HERE, "logs")
_REPO = os.path.dirname(os.path.dirname(HERE))
MODEL_DIR = os.path.join(_REPO, "pipeline", "model")
DATA_DIR = os.environ.get("THRML_DATA_DIR") or os.path.join(_REPO, "data")


def ensure_art():
    os.makedirs(ART, exist_ok=True)
    return ART
