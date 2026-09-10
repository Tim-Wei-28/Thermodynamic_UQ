"""Directory layout of the study.

lib/ reusable modules, experiments/ training drivers, reporting/ table scripts,
artifacts/ trained parameters and result JSONs, logs/ run logs.
DATA_DIR honours $THRML_DATA_DIR, so the data can live outside the project folder.
"""
from __future__ import annotations
import os

# lib/paths.py -> studies/cifar10_resnet13/
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts")
LOGS = os.path.join(HERE, "logs")
# studies/cifar10_resnet13/ -> studies/ -> repo root
_REPO = os.path.dirname(os.path.dirname(HERE))
MODEL_DIR = os.path.join(_REPO, "pipeline", "model")
DATA_DIR = os.environ.get("THRML_DATA_DIR") or os.path.join(_REPO, "data")


def ensure_art():
    os.makedirs(ART, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    return ART
