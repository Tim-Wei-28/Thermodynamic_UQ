"""Data layer for the Fashion-MNIST study: splits, class-balanced subsets, LeNet input
shaping and the corruption probes. Reader, EMNIST-Letters loader and probes are reused
from pipeline/model/fashion_task.py.
"""
from __future__ import annotations
import os
import sys

import numpy as np

from paths import MODEL_DIR, DATA_DIR
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import fashion_task as _ft                                          # noqa: E402

_read_idx = _ft._read_idx
_letters_raw = _ft._letters_raw
blend_probe = _ft._blend
noise_probe = _ft._noise
rotate_probe = _ft._rotate
PROBES = {"letters": blend_probe, "noise": noise_probe, "rotate": rotate_probe}
FRACTIONS = _ft.FRACTIONS
CLASS_NAMES = _ft.CLASS_NAMES
SIDE = _ft.SIDE

FASHION_DIR = os.path.join(DATA_DIR, "fashion")
_IDX_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x":  "t10k-images-idx3-ubyte.gz",
    "test_y":  "t10k-labels-idx1-ubyte.gz",
}

N_VAL = 5000            # training images held back for temperature fitting
C = 10
_RAW = None


def _raw():
    """The four Fashion-MNIST arrays, all ten classes, labels unmapped."""
    global _RAW
    if _RAW is not None:
        return _RAW
    out = {}
    for key, fname in _IDX_FILES.items():
        p = os.path.join(FASHION_DIR, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Fashion-MNIST file {fname} missing at {p}.  Download the four IDX files "
                f"from https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
                f"master/data/fashion/ into pipeline/data/fashion/.")
        a = _read_idx(p)
        out[key] = a.reshape(a.shape[0], -1) if key.endswith("_x") else a
    _RAW = out
    return _RAW


def splits():
    """Train / val / test images in [0,1], flat (N,784).
    First 55 000 training images, last 5 000 as validation, official test set.
    No shuffling; the training file is already class-interleaved."""
    r = _raw()
    Xtr_all = r["train_x"].astype(np.float64) / 255.0
    Ytr_all = r["train_y"].astype(np.int32)
    n_tr = Xtr_all.shape[0] - N_VAL
    return ((Xtr_all[:n_tr], Ytr_all[:n_tr]),
            (Xtr_all[n_tr:], Ytr_all[n_tr:]),
            (r["test_x"].astype(np.float64) / 255.0, r["test_y"].astype(np.int32)))


def subset(X, Y, n_per, seed=0):
    """Class-balanced draw of n_per images per class."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(C):
        cand = np.flatnonzero(Y == c)
        if cand.size < n_per:
            raise ValueError(f"class {c} has only {cand.size} images, n_per={n_per} asked")
        idx.append(rng.choice(cand, n_per, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], Y[idx]


def to_lenet(X):
    """(N,784) in [0,1] -> (N,1,32,32) float32 LeNet-5 input. The pad to 32x32 gives
    the classic layer sizes and the 61.7 K parameter count of Liu et al. (2022)."""
    img = X.reshape(-1, 1, SIDE, SIDE).astype(np.float32)
    return np.pad(img, ((0, 0), (0, 0), (2, 2), (2, 2)))


def centre_fn(Xtr):
    """Subtract the training mean image. W0 is frozen, so an uncentred input would add
    a constant to every logit that nothing can remove."""
    mean = Xtr.mean(0)
    return lambda X: X - mean


def _ascii(x, w=28):
    """Render one flat image as text."""
    ramp = " .:-=+*#%@"
    a = x.reshape(w, w)
    return "\n".join("".join(ramp[min(9, int(v * 9.999))] for v in row) for row in a)


def selftest():
    """Shapes, ranges, class balance and one rendered image per source. The letter must
    come out upright, which checks that the EMNIST transpose was undone."""
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = splits()
    print(f"train {Xtr.shape} {Xtr.dtype}  range [{Xtr.min():.3f},{Xtr.max():.3f}]")
    print(f"val   {Xva.shape}   test {Xte.shape}")
    print("val class histogram :", np.bincount(Yva, minlength=C))
    print("test class histogram:", np.bincount(Yte, minlength=C))
    L = _letters_raw()
    print(f"letters {L.shape} {L.dtype}  range [{L.min()},{L.max()}]")
    print(f"\n--- clothing, label {Yte[0]} = {CLASS_NAMES[Yte[0]]}")
    print(_ascii(Xte[0]))
    print("\n--- letter (must read as an upright letter, not rotated/mirrored)")
    print(_ascii(L[0] / 255.0))
    for kind, fn in PROBES.items():
        Xb, Yb = fn(Xte, Yte, 0.5, 4, 0)
        print(f"\n--- {kind} at f=0.5, label {Yb[0]} = {CLASS_NAMES[Yb[0]]}  "
              f"(range [{Xb.min():.3f},{Xb.max():.3f}])")
        print(_ascii(Xb[0]))


if __name__ == "__main__":
    import os as _o, sys as _s
    _s.path[:0] = [_o.path.dirname(_o.path.abspath(__file__))]
    selftest()
