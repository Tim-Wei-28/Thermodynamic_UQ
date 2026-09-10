"""
CIFAR-10 task behind a frozen ResNet-13 trunk (thesis Chapter 5.2).  Stratified
split of the 50 000 training images into 4500 train / 500 val per class, fixed
seed; test = the official 10 000.  Input = 256 pooled features plus a constant
column (D=257), or centred pixels (D=3072) without a trunk.
OOD probes: SVHN blend, uniform noise blend, rotation.
"""
from __future__ import annotations
import hashlib
import os
import pickle
import sys
import tarfile
import urllib.error
import urllib.request

import numpy as np

import jax                                                          
import jax.numpy as jnp                                             
import resnet                                                       

# trunk head width, passed explicitly (rows may share one process)
N_CLASS_TRUNK = 10

# <repo>/data/ unless $THRML_DATA_DIR overrides it
_DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DATA_ROOT = os.environ.get("THRML_DATA_DIR") or _DEFAULT_ROOT
DATA_DIR = os.path.join(DATA_ROOT, "cifar")
SVHN_DIR = os.path.join(DATA_ROOT, "svhn")
TRUNK_DIR = os.path.join(DATA_DIR, "resnet_trunks")
FEAT_DIR = os.path.join(DATA_DIR, "resnet_features")

ARCHIVE = "cifar-10-python.tar.gz"
ARCHIVE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
SVHN_FILE = "test_32x32.mat"
SVHN_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
CACHE = os.path.join(DATA_DIR, "cifar_cache.npz")
SVHN_CACHE = os.path.join(SVHN_DIR, "svhn_test_cache.npz")
ALLOW_DOWNLOAD = True                # False -> raise instead of touching the network

SIDE, IN_CH = 32, 3
N_PIX = IN_CH * SIDE * SIDE          # 3072
N_CLASSES = 10
N_TRAIN_PER = 4500                   # per class, after the stratified val cut
N_VAL_PER = 500                      # per class

# module state, re-derived by configure()
CLASSES = tuple(range(N_CLASSES))
TRUNK = "resnet45k"                  # 'none' | 'resnet45k' | 'resnet10k'
TRUNK_EPOCHS = 100
TRUNK_SCHEDULE = "step"
PROBE = "svhn"
OOD_FRACTION = 0.5
SEED = 0

C = N_CLASSES
D = 257
D_FEAT = 256
K_DEFAULT = N_CLASSES                # K = C
IMG_SHAPE = None

_RAW = None
_SVHN = None
_TRUNK_P = None
_FEAT_SCALE = 1.0
_MEAN_VEC = None
_FEAT_CACHE = {}


# ================================================================= 1. getting the bytes
def _download(url, dst):
    """Stream `url` to `dst` atomically (pid-tagged temp file, timeout)."""
    if not ALLOW_DOWNLOAD:
        raise FileNotFoundError(
            f"{dst} is missing and ALLOW_DOWNLOAD is False.  Fetch {url} by hand, or point "
            f"$THRML_DATA_DIR at a directory that already has it.")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.part{os.getpid()}"
    print(f"  cifar10_resnet: downloading {os.path.basename(dst)} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except (urllib.error.URLError, OSError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"download of {url} failed: {e}") from e
    os.replace(tmp, dst)
    return dst


def _atomic_savez(dst, arrays):
    """np.savez into `dst` atomically.  An open handle is passed because savez appends
       .npz to a bare name that lacks it."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = f"{dst}.part{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dst


def _raw():
    """The four raw CIFAR-10 arrays as uint8, cached as one npz.  Rows are planar R|G|B."""
    global _RAW
    if _RAW is not None:
        return _RAW
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        _RAW = {k: z[k] for k in ("train_x", "train_y", "test_x", "test_y")}
        return _RAW
    tar = os.path.join(DATA_DIR, ARCHIVE)
    if not os.path.exists(tar):
        _download(ARCHIVE_URL, tar)
    tr_x, tr_y, te_x, te_y = [], [], None, None
    with tarfile.open(tar, "r:gz") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if not (name.startswith("data_batch_") or name == "test_batch"):
                continue
            d = pickle.load(tf.extractfile(m), encoding="bytes")
            x = np.asarray(d[b"data"], dtype=np.uint8)          # (10000, 3072)
            y = np.asarray(d[b"labels"], dtype=np.int64)
            if name == "test_batch":
                te_x, te_y = x, y
            else:
                tr_x.append(x); tr_y.append(y)
    if te_x is None or not tr_x:
        raise RuntimeError(f"{ARCHIVE} did not contain the expected batches")
    out = {"train_x": np.concatenate(tr_x), "train_y": np.concatenate(tr_y),
           "test_x": te_x, "test_y": te_y}
    _atomic_savez(CACHE, out)          # uncompressed on purpose
    _RAW = out
    return _RAW


def _svhn_raw():
    """SVHN test images as (N,3072) uint8 in CIFAR's plane order.  The .mat is
       (32,32,3,N), hence the transpose.  Labels are not used."""
    global _SVHN
    if _SVHN is not None:
        return _SVHN
    if os.path.exists(SVHN_CACHE):
        _SVHN = np.load(SVHN_CACHE)["x"]
        return _SVHN
    mat = os.path.join(SVHN_DIR, SVHN_FILE)
    if not os.path.exists(mat):
        _download(SVHN_URL, mat)
    from scipy.io import loadmat                       # only needed on the cache miss
    X = loadmat(mat)["X"]                              # (32,32,3,N) uint8
    X = np.transpose(X, (3, 2, 0, 1)).reshape(X.shape[3], -1).astype(np.uint8)
    _atomic_savez(SVHN_CACHE, {"x": X})
    _SVHN = X
    return _SVHN


# ----------------------------------------------------------------- OOD probes
def _blend(Xte, Yte, f, n, seed):
    """Thesis Eq. 5.1: x = (1-f)*cifar + f*svhn, label kept from the CIFAR image."""
    rng = np.random.default_rng(seed)
    S = _svhn_raw().astype(np.float64) / 255.0
    ic = rng.choice(Xte.shape[0], n, replace=False)
    isv = rng.choice(S.shape[0], n, replace=False)
    return (1.0 - f) * Xte[ic] + f * S[isv], Yte[ic]


def _noise(Xte, Yte, f, n, seed):
    """Same blend with uniform pixel noise."""
    rng = np.random.default_rng(seed)
    ic = rng.choice(Xte.shape[0], n, replace=False)
    return (1.0 - f) * Xte[ic] + f * rng.random((n, Xte.shape[1])), Yte[ic]


def _rotate(Xte, Yte, f, n, seed):
    """Rotation by f*90 degrees, bilinear resampling shared across the three planes."""
    rng = np.random.default_rng(seed)
    ic = rng.choice(Xte.shape[0], n, replace=False)
    img = Xte[ic].reshape(-1, IN_CH, SIDE, SIDE)
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
            out += np.where(ok, w, 0.0)[None, None] * img[:, :, np.clip(yi, 0, SIDE - 1),
                                                          np.clip(xi, 0, SIDE - 1)]
    return np.clip(out, 0.0, 1.0).reshape(n, -1), Yte[ic]


PROBES = {"svhn": _blend, "noise": _noise, "rotate": _rotate}
FRACTIONS = tuple(np.round(np.arange(0.0, 0.91, 0.1), 2))


# ================================================================= 2. shaping them
def _split_idx():
    """Stratified train/val indices into the training file.  Fixed seed, independent
       of the run seed: the split belongs to the dataset, not the run."""
    Y = _raw()["train_y"]
    rng = np.random.default_rng(12345)
    tr, va = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(Y == c)
        idx = idx[rng.permutation(idx.size)]
        va.append(idx[:N_VAL_PER])
        tr.append(idx[N_VAL_PER:N_VAL_PER + N_TRAIN_PER])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def _pool(group):
    """(images float64 in [0,1], labels) for one split, restricted to CLASSES."""
    r = _raw()
    if group == "test":
        X, Y = r["test_x"], r["test_y"]
    else:
        tr, va = _split_idx()
        sel = tr if group == "train" else va
        X, Y = r["train_x"][sel], r["train_y"][sel]
    keep = np.isin(Y, np.asarray(CLASSES))
    X, Y = X[keep], Y[keep]
    remap = {c: i for i, c in enumerate(CLASSES)}      # label = position in CLASSES
    return X.astype(np.float64) / 255.0, np.array([remap[int(y)] for y in Y], np.int32)


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


# ================================================================= 3. serving them
def _trunk_tag(trunk=None, epochs=None, sched=None, seed=None):
    """Filename stem shared by the parameter cache and the feature cache."""
    return (f"{TRUNK if trunk is None else trunk}"
            f"_e{TRUNK_EPOCHS if epochs is None else epochs}"
            f"_{TRUNK_SCHEDULE if sched is None else sched}"
            f"_s{SEED if seed is None else seed}")


def _trunk():
    """Frozen ResNet, trained on demand (expensive) and cached on disk."""
    global _TRUNK_P
    if TRUNK == "none":
        return None
    if _TRUNK_P is not None:
        return _TRUNK_P
    os.makedirs(TRUNK_DIR, exist_ok=True)
    path = os.path.join(TRUNK_DIR, _trunk_tag() + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        _TRUNK_P = {k: jnp.asarray(z[k]) for k in z.files}
        return _TRUNK_P
    Xtr, Ytr = _pool("train")
    n_img = 45000 if TRUNK == "resnet45k" else 10000
    if n_img < Xtr.shape[0]:                           # class-balanced subset
        rng = np.random.default_rng(SEED)
        idx = np.sort(np.concatenate(
            [rng.choice(np.flatnonzero(Ytr == c), n_img // C, replace=False)
             for c in range(C)]))
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    print(f"  cifar10_resnet: training the {TRUNK} trunk on {Xtr.shape[0]} images "
          f"({TRUNK_EPOCHS} epochs, {TRUNK_SCHEDULE}, seed {SEED}) -- cached afterwards",
          flush=True)
    p, _ = resnet.train(Xtr, Ytr, epochs=TRUNK_EPOCHS, seed=SEED, verbose=False,
                        schedule=TRUNK_SCHEDULE, n_class=N_CLASS_TRUNK)
    jax.block_until_ready(p["d_w"])
    if tuple(p["d_w"].shape) != (N_CLASS_TRUNK, resnet.N_FEAT):
        raise RuntimeError(f"trunk head is {tuple(p['d_w'].shape)}, expected "
                           f"{(N_CLASS_TRUNK, resnet.N_FEAT)}")
    _atomic_savez(path, {k: np.asarray(v) for k, v in p.items()})
    _TRUNK_P = p
    return _TRUNK_P


def _pool_features(group):
    """Raw pooled trunk activations for a whole split, cached on disk.  Cached before
       the scale and the constant column, since the scale depends on n_per."""
    _cls = hashlib.blake2s(repr(CLASSES).encode(), digest_size=3).hexdigest()
    key = f"{_trunk_tag()}_{group}_c{len(CLASSES)}-{_cls}"
    hit = _FEAT_CACHE.get(key)
    if hit is not None:
        return hit
    path = os.path.join(FEAT_DIR, f"{key}.npy")
    if os.path.exists(path):
        F = np.load(path)
        _FEAT_CACHE[key] = F
        return F
    X, _ = _pool(group)
    F = resnet.features(_trunk(), X)
    os.makedirs(FEAT_DIR, exist_ok=True)
    tmp = f"{path}.part{os.getpid()}"
    try:
        np.save(tmp, F)                    # np.save appends .npy
        os.replace(tmp + ".npy", path)
    except BaseException:
        for p in (tmp, tmp + ".npy"):
            if os.path.exists(p):
                os.remove(p)
        raise
    _FEAT_CACHE[key] = F
    return F


def _features(X):
    """Raw images -> model input: centred pixels, or [pooled * scale, 1] with a trunk."""
    X = np.asarray(X, np.float64)
    if TRUNK == "none":
        return X - _MEAN_VEC
    F = resnet.features(_trunk(), X) * _FEAT_SCALE
    return np.concatenate([F, np.ones((F.shape[0], 1))], 1)


def configure(trunk=None, trunk_epochs=None, classes=None, probe=None,
              ood_fraction=None, seed=None, n_per=None, schedule=None):
    """Re-derive every module constant.  Validates before assigning."""
    global CLASSES, TRUNK, TRUNK_EPOCHS, TRUNK_SCHEDULE, PROBE, OOD_FRACTION, SEED
    global C, D, D_FEAT, K_DEFAULT, IMG_SHAPE, _MEAN_VEC, _TRUNK_P, _FEAT_SCALE

    _trunk_k = TRUNK if trunk is None else str(trunk)
    _ep = TRUNK_EPOCHS if trunk_epochs is None else int(trunk_epochs)
    _sched = TRUNK_SCHEDULE if schedule is None else str(schedule)
    if classes is None:
        _cls = CLASSES
    else:
        _cls = (tuple(range(N_CLASSES)) if str(classes).strip().lower() in ("all", "")
                else tuple(int(x) for x in str(classes).split(",") if str(x).strip()))
    _probe = PROBE if probe is None else str(probe)
    _frac = OOD_FRACTION if ood_fraction is None else float(ood_fraction)
    _seed = SEED if seed is None else int(seed)

    if _sched not in resnet.SCHEDULES:
        raise ValueError(f"cifar10_trunk_schedule must be one of {resnet.SCHEDULES} "
                         f"(got {_sched!r})")
    if _trunk_k not in ("none", "resnet45k", "resnet10k"):
        raise ValueError(f"cifar10_trunk must be none | resnet45k | resnet10k "
                         f"(got {_trunk_k!r})")
    if _probe not in PROBES:
        raise ValueError(f"cifar10_probe must be one of {sorted(PROBES)} (got {_probe!r})")
    if not 0.0 <= _frac <= 1.0:
        raise ValueError(f"cifar10_ood_fraction must be in [0,1] (got {_frac})")
    if not _cls or len(set(_cls)) != len(_cls) or not all(0 <= c < N_CLASSES for c in _cls):
        raise ValueError(f"cifar10_classes must be 'all' or distinct 0-9 (got {_cls})")
    if _ep < 1:
        raise ValueError(f"cifar10_trunk_epochs must be >= 1 (got {_ep})")

    changed = ((_trunk_k, _ep, _sched, _cls, _seed)
               != (TRUNK, TRUNK_EPOCHS, TRUNK_SCHEDULE, CLASSES, SEED))
    CLASSES, TRUNK, TRUNK_EPOCHS, TRUNK_SCHEDULE, PROBE, OOD_FRACTION, SEED = \
        _cls, _trunk_k, _ep, _sched, _probe, _frac, _seed
    C = len(CLASSES)
    if changed:
        _TRUNK_P = None
        _FEAT_CACHE.clear()
    if TRUNK == "none":
        Xtr, _ = _pool("train")
        _MEAN_VEC = Xtr.mean(0)
        D_FEAT, D = N_PIX, N_PIX
        IMG_SHAPE = (IN_CH, SIDE, SIDE)
    else:
        # feature scale is fitted on the images the run actually trains on
        Fall = _pool_features("train")
        F = Fall[_draw_idx("train", int(n_per))] if n_per else Fall[:5000]
        _FEAT_SCALE = 1.0 / (float(F.std()) + 1e-12)
        D_FEAT = F.shape[1]
        D = D_FEAT + 1                       # constant column carries the bias
        IMG_SHAPE = None
    K_DEFAULT = C


def make_data(n_per, key, group="train"):
    """Class-balanced draw of n_per images per class -> (X, Y).  Max 4500 on 'train',
       500 on 'val', 1000 on 'test'.  `key` is ignored; the draw follows SEED.
       n_per=None takes the whole pool in file order."""
    X, Y = _pool(group)
    idx = np.arange(X.shape[0]) if n_per in (None, 0) else _draw_idx(group, n_per)
    if TRUNK == "none":
        return jnp.asarray(_features(X[idx])), jnp.asarray(Y[idx], dtype=jnp.int32)
    # index the cached pool features; exact since BatchNorm runs in eval mode
    F = _pool_features(group)[idx] * _FEAT_SCALE
    F = np.concatenate([F, np.ones((F.shape[0], 1))], 1)
    return jnp.asarray(F), jnp.asarray(Y[idx], dtype=jnp.int32)


def max_per_class(group):
    """Largest n_per a class-balanced draw from this pool can serve."""
    _, Y = _pool(group)
    return int(min(int((Y == c).sum()) for c in range(C)))


def w0_matrix(alpha, w0_scale, seed=0):
    """Frozen base map (C, D): random for trunk='none', otherwise
       (1-alpha) * [W_d/scale | b_d] + alpha * random."""
    rng = np.random.default_rng(seed)
    if TRUNK == "none":
        return w0_scale * rng.standard_normal((C, D))
    W_d, b_d = resnet.head(_trunk())                   # (10,256), (10,)
    W_d, b_d = W_d[:C], b_d[:C]
    trained = np.concatenate([W_d / _FEAT_SCALE, b_d[:, None]], 1)     # (C, D)
    return (1.0 - alpha) * trained + alpha * w0_scale * rng.standard_normal((C, D))


def acc_w0(W0, n_test=10000, seed=123):
    """Accuracy of the frozen base map on its own."""
    X, Y = make_data(max(1, min(n_test // C, max_per_class("test"))),
                     jax.random.key(seed), group="test")
    return float((np.asarray(X) @ np.asarray(W0).T).argmax(1).__eq__(np.asarray(Y)).mean())


def probe(kind, fraction, n=1000, seed=0):
    """(X, Y) for one OOD probe, passed through the configured front-end."""
    Xte, Yte = _pool("test")
    Xb, Yb = PROBES[kind](Xte, Yte, float(fraction), int(n), int(seed))
    return jnp.asarray(_features(Xb)), jnp.asarray(Yb, dtype=jnp.int32)


def chance(**_):
    """Majority-class rate on the test pool."""
    _, Y = _pool("test")
    return float(np.bincount(Y, minlength=C).max() / Y.size)


# ================================================================= selftest
def selftest():
    (Xtr, Ytr) = _pool("train")
    (Xva, Yva) = _pool("val")
    (Xte, Yte) = _pool("test")
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    for nm, Y, want in (("train", Ytr, N_TRAIN_PER), ("val", Yva, N_VAL_PER),
                        ("test", Yte, 1000)):
        h = np.bincount(Y, minlength=N_CLASSES)
        ok = h.min() == h.max() == want
        print(f"  {nm:5s} per class {h.min()}..{h.max()}  (want {want})  "
              f"{'OK' if ok else 'MISMATCH'}")
        assert ok, f"{nm} split is not {want} per class"
    assert Xtr.shape[0] == 45000, f"train is {Xtr.shape[0]}, not 45000"
    print(f"  -> N_train = {Xtr.shape[0]}")
    tr, va = _split_idx()
    assert not (set(tr.tolist()) & set(va.tolist())), "train and val overlap"
    print("  -> train / val indices are disjoint")
    print(f"  chance = {chance():.4f}")
    for kind in PROBES:
        Xb, Yb = PROBES[kind](Xte, Yte, 0.5, 8, 50)
        print(f"  probe {kind:7s} f=0.5 -> {Xb.shape} range "
              f"[{Xb.min():.3f},{Xb.max():.3f}]")


if __name__ == "__main__":
    import os as _o, sys as _s
    _s.path[:0] = [_o.path.dirname(_o.path.abspath(__file__))]
    selftest()
