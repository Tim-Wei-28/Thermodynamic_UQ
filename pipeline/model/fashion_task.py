"""
Fashion-MNIST task.  Train = first 55 000 official training images, val = last
5 000, test = the official 10 000.  Input is either raw centred pixels (D=784) or
the 84 F6 features of a frozen LeNet-5 plus a constant column (D=85).
OOD probes: EMNIST-letter blend, uniform noise blend, rotation.
"""
from __future__ import annotations
import gzip
import os

import numpy as np
import jax
import jax.numpy as jnp

import lenet

# <repo>/data/ unless $THRML_DATA_DIR overrides it
_DATA = os.environ.get("THRML_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DATA_DIR = os.path.join(_DATA, "fashion")
EMNIST_DIR = os.path.join(_DATA, "emnist")
TRUNK_DIR = os.path.join(DATA_DIR, "trunks")

_IDX_FILES = {"train_x": "train-images-idx3-ubyte.gz",
              "train_y": "train-labels-idx1-ubyte.gz",
              "test_x": "t10k-images-idx3-ubyte.gz",
              "test_y": "t10k-labels-idx1-ubyte.gz"}
_LETTER_FILE = "emnist-letters-test-images-idx3-ubyte.gz"

CLASS_NAMES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
SIDE = 28
N_VAL = 5000                    # training images held back for temperature fitting

# derived constants, rebuilt by configure()
CLASSES = tuple(range(10))
TRUNK = "lenet55k"              # 'none' | 'lenet55k' | 'lenet10k'
TRUNK_EPOCHS = 20
PROBE = "letters"               # probe that feeds the headline columns
OOD_FRACTION = 0.5
SEED = 0                        # trunk and draw seed; set by the engine

C = 10                          # classes
D = 85                          # input dims the model sees
D_FEAT = 84                     # trunk output width (before the constant column)
K_DEFAULT = 10                  # gates when K_gates is blank
IMG_SHAPE = None                # figures: only meaningful without a trunk

_RAW = None
_LETTERS = None
_TRUNK_P = None                 # frozen LeNet parameters
_FEAT_SCALE = 1.0               # single scalar; W0 absorbs it exactly (see w0_matrix)
_MEAN_VEC = None                # trunk='none': the training mean image
_CACHE = {}


# ================================================================= 1. getting the bytes
def _read_idx(path):
    """Parse one IDX file (4 magic bytes, one big-endian int32 per dim, then the data)."""
    with gzip.open(path, "rb") as f:
        blob = f.read()
    ndim = blob[3]
    dims = np.frombuffer(blob[4:4 + 4 * ndim], dtype=">u4").astype(int)
    return np.frombuffer(blob[4 + 4 * ndim:], dtype=np.uint8).reshape(*dims)


def _raw():
    global _RAW
    if _RAW is not None:
        return _RAW
    out = {}
    for key, fname in _IDX_FILES.items():
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Fashion-MNIST file missing: {p}.  Download the four IDX files from "
                f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
                f"master/data/fashion/ into {DATA_DIR}.")
        a = _read_idx(p)
        out[key] = a.reshape(a.shape[0], -1) if key.endswith("_x") else a
    _RAW = out
    return _RAW


def _letters_raw():
    """EMNIST-Letters test images; EMNIST is stored column-major, hence the transpose."""
    global _LETTERS
    if _LETTERS is not None:
        return _LETTERS
    p = os.path.join(EMNIST_DIR, _LETTER_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"EMNIST-Letters missing: {p}.  Fetch "
            f"https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip and extract "
            f"gzip/{_LETTER_FILE} into {EMNIST_DIR}.")
    a = _read_idx(p).reshape(-1, SIDE, SIDE).transpose(0, 2, 1)
    _LETTERS = a.reshape(a.shape[0], -1).copy()
    return _LETTERS


# val is carved out of the training file; the official test set stays intact
_SPLITS = {"train": (0, 55000), "val": (55000, 60000)}


def _pool(group):
    """(images float64 in [0,1], labels) for one split, restricted to CLASSES."""
    r = _raw()
    if group == "test":
        X, Y = r["test_x"], r["test_y"]
    else:
        lo, hi = _SPLITS[group]
        X, Y = r["train_x"][lo:hi], r["train_y"][lo:hi]
    keep = np.isin(Y, np.asarray(CLASSES))
    X, Y = X[keep], Y[keep]
    remap = {c: i for i, c in enumerate(CLASSES)}            # label = position in CLASSES
    return X.astype(np.float64) / 255.0, np.array([remap[int(y)] for y in Y], np.int32)


# ================================================================= 2. shaping them
def _trunk():
    """Frozen LeNet-5, trained on demand and cached on disk keyed by (size, epochs, seed)."""
    global _TRUNK_P
    if TRUNK == "none":
        return None
    if _TRUNK_P is not None:
        return _TRUNK_P
    n_img = 55000 if TRUNK == "lenet55k" else 10000
    os.makedirs(TRUNK_DIR, exist_ok=True)
    path = os.path.join(TRUNK_DIR, f"{TRUNK}_e{TRUNK_EPOCHS}_s{SEED}.npz")
    if os.path.exists(path):
        z = np.load(path)
        _TRUNK_P = {k: jnp.asarray(z[k]) for k in z.files}
        return _TRUNK_P
    Xtr, Ytr = _pool("train")
    if n_img < Xtr.shape[0]:                                # class-balanced subset
        rng = np.random.default_rng(SEED)
        idx = np.sort(np.concatenate(
            [rng.choice(np.flatnonzero(Ytr == c), n_img // C, replace=False)
             for c in range(C)]))
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    print(f"  fashion: training the {TRUNK} trunk on {Xtr.shape[0]} images "
          f"({TRUNK_EPOCHS} epochs, seed {SEED}) -- cached afterwards")
    p, _ = lenet.train(Xtr, Ytr, epochs=TRUNK_EPOCHS, seed=SEED, verbose=False)
    jax.block_until_ready(p["out_w"])
    np.savez(path, **{k: np.asarray(v) for k, v in p.items()})
    _TRUNK_P = p
    return _TRUNK_P


def _features(X):
    """Raw images -> model input: centred pixels, or [F6 * scale, 1] with a trunk."""
    X = np.asarray(X, np.float64)
    if TRUNK == "none":
        return X - _MEAN_VEC
    F = lenet.features(_trunk(), X) * _FEAT_SCALE
    return np.concatenate([F, np.ones((F.shape[0], 1))], 1)


def _draw_idx(group, n_per):
    """Class-balanced index draw, shared by make_data and the feature-scale fit."""
    _, Y = _pool(group)
    rng = np.random.default_rng(SEED)
    idx = []
    for c in range(C):
        cand = np.flatnonzero(Y == c)
        if cand.size < n_per:
            raise ValueError(f"class {CLASSES[c]} has only {cand.size} images in "
                             f"{group!r}, but n_per={n_per} were requested")
        idx.append(rng.choice(cand, int(n_per), replace=False))
    return np.sort(np.concatenate(idx))


def configure(trunk=None, trunk_epochs=None, classes=None, probe=None,
              ood_fraction=None, seed=None, n_per=None):
    """Re-derive every module constant.  Validates before assigning."""
    global CLASSES, TRUNK, TRUNK_EPOCHS, PROBE, OOD_FRACTION, SEED
    global C, D, D_FEAT, K_DEFAULT, IMG_SHAPE, _MEAN_VEC, _TRUNK_P, _FEAT_SCALE

    _trunk_k = TRUNK if trunk is None else str(trunk)
    _ep = TRUNK_EPOCHS if trunk_epochs is None else int(trunk_epochs)
    if classes is None:
        _cls = CLASSES
    else:
        _cls = (tuple(range(10)) if str(classes).strip().lower() in ("all", "")
                else tuple(int(x) for x in str(classes).split(",") if str(x).strip()))
    _probe = PROBE if probe is None else str(probe)
    _frac = OOD_FRACTION if ood_fraction is None else float(ood_fraction)
    _seed = SEED if seed is None else int(seed)

    if _trunk_k not in ("none", "lenet55k", "lenet10k"):
        raise ValueError(f"fashion_trunk must be none | lenet55k | lenet10k "
                         f"(got {_trunk_k!r})")
    if _probe not in PROBES:
        raise ValueError(f"fashion_probe must be one of {sorted(PROBES)} (got {_probe!r})")
    if not 0.0 <= _frac <= 1.0:
        raise ValueError(f"fashion_ood_fraction must be in [0,1] (got {_frac})")
    if not _cls or len(set(_cls)) != len(_cls) or not all(0 <= c <= 9 for c in _cls):
        raise ValueError(f"fashion_classes must be 'all' or distinct 0-9 (got {_cls})")
    if _ep < 1:
        raise ValueError(f"fashion_trunk_epochs must be >= 1 (got {_ep})")

    changed = (_trunk_k, _ep, _cls, _seed) != (TRUNK, TRUNK_EPOCHS, CLASSES, SEED)
    CLASSES, TRUNK, TRUNK_EPOCHS, PROBE, OOD_FRACTION, SEED = \
        _cls, _trunk_k, _ep, _probe, _frac, _seed
    C = len(CLASSES)
    if changed:
        _TRUNK_P = None
    if TRUNK == "none":
        Xtr, _ = _pool("train")
        _MEAN_VEC = Xtr.mean(0)
        D_FEAT, D = SIDE * SIDE, SIDE * SIDE
        IMG_SHAPE = (SIDE, SIDE)
    else:
        Xtr, _ = _pool("train")
        # feature scale is fitted on the images the run actually trains on
        if n_per:
            F = lenet.features(_trunk(), Xtr[_draw_idx("train", int(n_per))])
        else:
            F = lenet.features(_trunk(), Xtr[:5000])
        _FEAT_SCALE = 1.0 / (float(F.std()) + 1e-12)
        D_FEAT = F.shape[1]
        D = D_FEAT + 1                        
        IMG_SHAPE = None
    K_DEFAULT = C


# ================================================================= 3. serving them
def make_data(n_per, key, group="train"):
    """Class-balanced draw of n_per images per class -> (X, Y).
       The draw uses numpy seeded by SEED; `key` is ignored.
       n_per=None takes the whole pool in file order."""
    X, Y = _pool(group)
    idx = np.arange(X.shape[0]) if n_per in (None, 0) else _draw_idx(group, n_per)
    return jnp.asarray(_features(X[idx])), jnp.asarray(Y[idx], dtype=jnp.int32)


def max_per_class(group):
    """Largest n_per a class-balanced draw from this pool can serve."""
    _, Y = _pool(group)
    return int(min(int((Y == c).sum()) for c in range(C)))


def w0_matrix(alpha, w0_scale, seed=0):
    """Frozen base map (C, D): random for trunk='none', otherwise
       (1-alpha) * [W_out/scale | b_out] + alpha * random."""
    rng = np.random.default_rng(seed)
    if TRUNK == "none":
        return w0_scale * rng.standard_normal((C, D))
    W_out, b_out = lenet.head(_trunk())                     # (10,84), (10,)
    W_out, b_out = W_out[:C], b_out[:C]
    trained = np.concatenate([W_out / _FEAT_SCALE, b_out[:, None]], 1)   # (C, D)
    return (1.0 - alpha) * trained + alpha * w0_scale * rng.standard_normal((C, D))


def acc_w0(W0, n_test=10000, seed=123):
    """Accuracy of the frozen base map on its own."""
    X, Y = make_data(max(1, n_test // C), jax.random.key(seed), group="test")
    return float((np.asarray(X) @ np.asarray(W0).T).argmax(1).__eq__(np.asarray(Y)).mean())


# ----------------------------------------------------------------- OOD probes
def _blend(Xte, Yte, f, n, seed):
    """x = (1-f)*clothing + f*letter, label kept from the clothing."""
    rng = np.random.default_rng(seed)
    L = _letters_raw().astype(np.float64) / 255.0
    ic = rng.choice(Xte.shape[0], n, replace=False)
    il = rng.choice(L.shape[0], n, replace=False)
    return (1.0 - f) * Xte[ic] + f * L[il], Yte[ic]


def _noise(Xte, Yte, f, n, seed):
    """Same blend with uniform pixel noise."""
    rng = np.random.default_rng(seed)
    ic = rng.choice(Xte.shape[0], n, replace=False)
    return (1.0 - f) * Xte[ic] + f * rng.random((n, Xte.shape[1])), Yte[ic]


def _rotate(Xte, Yte, f, n, seed):
    """Rotation by f*90 degrees, bilinear resampling in plain numpy."""
    rng = np.random.default_rng(seed)
    ic = rng.choice(Xte.shape[0], n, replace=False)
    img = Xte[ic].reshape(-1, SIDE, SIDE)
    th, c = np.deg2rad(float(f) * 90.0), (SIDE - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    sy = np.cos(th) * (yy - c) + np.sin(th) * (xx - c) + c
    sx = -np.sin(th) * (yy - c) + np.cos(th) * (xx - c) + c
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    fy, fx = sy - y0, sx - x0
    out = np.zeros_like(img)
    for dy in (0, 1):
        for dx in (0, 1):
            yi, xi = y0 + dy, x0 + dx
            ok = (yi >= 0) & (yi < SIDE) & (xi >= 0) & (xi < SIDE)
            w = ((1 - fy) if dy == 0 else fy) * ((1 - fx) if dx == 0 else fx)
            out += np.where(ok, w, 0.0) * img[:, np.clip(yi, 0, SIDE - 1),
                                              np.clip(xi, 0, SIDE - 1)]
    return np.clip(out, 0.0, 1.0).reshape(n, -1), Yte[ic]


PROBES = {"letters": _blend, "noise": _noise, "rotate": _rotate}
FRACTIONS = tuple(np.round(np.arange(0.0, 0.91, 0.1), 2))


def probe(kind, fraction, n=1000, seed=0):
    """(X, Y) for one OOD probe; blended in pixel space, then passed through the trunk."""
    Xte, Yte = _pool("test")
    Xb, Yb = PROBES[kind](Xte, Yte, float(fraction), int(n), int(seed))
    return jnp.asarray(_features(Xb)), jnp.asarray(Yb, dtype=jnp.int32)


def as_images(X, **_):
    """Model inputs back as images, or None when a trunk is in the way."""
    if TRUNK != "none":
        return None
    return (np.asarray(X) + _MEAN_VEC).reshape(-1, SIDE, SIDE)


def chance(**_):
    return 1.0 / C


# ================================================================= selftest
if __name__ == "__main__":
    import sys
    for trunk in ("none", "lenet55k"):
        configure(trunk=trunk, seed=0)
        print(f"\n=== trunk={TRUNK}  C={C}  D={D}  K_DEFAULT={K_DEFAULT}")
        X, Y = make_data(20, jax.random.key(0))
        print(f"  train  {X.shape}  labels {np.bincount(np.asarray(Y), minlength=C)}")
        for a in (0.0, 1.0):
            W0 = w0_matrix(a, 0.1, seed=0)
            print(f"  W0(alpha={a})  shape {W0.shape}  "
                  f"acc alone {acc_w0(W0, 1000):.4f}")
        for kind in PROBES:
            Xp, Yp = probe(kind, 0.5, n=50, seed=0)
            print(f"  probe {kind:8s} {Xp.shape}  "
                  f"range [{float(Xp.min()):.3f}, {float(Xp.max()):.3f}]")
    print("\nselftest ok")
