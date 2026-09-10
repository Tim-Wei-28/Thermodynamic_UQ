"""Data layer of the CIFAR-10 study: splits, class-balanced subsets, OOD probes.

The archive reader, the stratified split and the corruption probes come from
pipeline/model/cifar10_resnet_task.py, so the pipeline and this study see the same
images.  Split is 4500 train / 500 val per class, test is the official 10 000.
"""
from __future__ import annotations
import os
import sys

import numpy as np

# pipeline/model/ holds cifar10_resnet_task and resnet.
from paths import MODEL_DIR
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import cifar10_resnet_task as _ct                                          # noqa: E402

# ---- re-exported from the task module ------------------------------------------------
PROBES = _ct.PROBES                  # 'svhn' | 'noise' | 'rotate'
FRACTIONS = _ct.FRACTIONS            # 0.0 .. 0.9, the paper's Figure 6E ladder
SIDE, IN_CH, N_PIX = _ct.SIDE, _ct.IN_CH, _ct.N_PIX
C = _ct.N_CLASSES                    # 10 classes
MAX_TRAIN_PER = _ct.N_TRAIN_PER      # 4500
MAX_TEST_PER = 1000


def splits():
    """(Xtr, Ytr), (Xva, Yva), (Xte, Yte) as float64 images in [0,1], flat (N, 3072).

       The split seed is fixed and independent of the run seed.  configure(trunk='none')
       first: the pools do not depend on the trunk, and 'resnet45k' would train one as a
       side effect of fitting the feature scale."""
    _ct.configure(trunk="none", seed=0)
    return _ct._pool("train"), _ct._pool("val"), _ct._pool("test")


def subset(X, Y, n_per, seed=0):
    """Class-balanced draw of n_per images per class.

       The EBM trainer is full-batch, so the head trains on a budget it can carry while
       the trunk saw the whole split.  At n_per=4500 the two coincide."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(C):
        cand = np.flatnonzero(Y == c)
        if cand.size < n_per:
            raise ValueError(f"class {c} has only {cand.size} images, n_per={n_per} asked")
        idx.append(rng.choice(cand, n_per, replace=False))
    idx = np.sort(np.concatenate(idx))       # sorted: keeps the draw order reproducible
    return X[idx], Y[idx]


def probe_raw(kind, fraction, Xte, Yte, n=1000, seed=0):
    """(X, Y) for one OOD probe, as raw images with no front-end applied.

       The blend happens in raw [0,1] pixel space so that every row applies its own
       preprocessing afterwards and all rows are scored on the same pictures."""
    return PROBES[kind](Xte, Yte, float(fraction), int(n), int(seed))


def _ascii(x):
    ramp = " .:-=+*#%@"
    a = np.asarray(x).reshape(IN_CH, SIDE, SIDE).mean(0)
    return "\n".join("".join(ramp[min(9, int(v * 9.999))] for v in row) for row in a)


def selftest():
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = splits()
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    for nm, Y in (("train", Ytr), ("val", Yva), ("test", Yte)):
        h = np.bincount(Y, minlength=C)
        print(f"  {nm:5s} per-class {h.min()}..{h.max()}   classes {len(h)}")
    Xs, Ys = subset(Xtr, Ytr, 10, seed=0)
    print(f"subset(10) -> {Xs.shape}, {len(np.unique(Ys))} classes, "
          f"{np.bincount(Ys, minlength=C).min()} per class")
    for kind in PROBES:
        Xb, Yb = probe_raw(kind, 0.5, Xte, Yte, n=8, seed=50)
        print(f"\n--- {kind} at f=0.5  {Xb.shape}  range "
              f"[{Xb.min():.3f},{Xb.max():.3f}]  label {Yb[0]}")
        print(_ascii(Xb[0]))


if __name__ == "__main__":
    import os as _o, sys as _s
    _s.path[:0] = [_o.path.dirname(_o.path.abspath(__file__))]
    selftest()
