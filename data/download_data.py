"""
Fetch the datasets the experiments use: fashion (Fashion-MNIST), emnist
(EMNIST-Letters test images, the OOD set), cifar (CIFAR-10), svhn (SVHN test set).
Files land in this folder, or in THRML_DATA_DIR, where the task modules read them.
Usage: python data/download_data.py [fashion emnist cifar svhn]   (default: all)
"""
from __future__ import annotations
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("THRML_DATA_DIR") or HERE

FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FASHION_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
EMNIST_ZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_MEMBER = "gzip/emnist-letters-test-images-idx3-ubyte.gz"


def _fetch(url, dst):
    """Stream `url` to `dst` with a progress line; skip if the file already exists."""
    if os.path.exists(dst):
        print(f"  have {os.path.basename(dst)}")
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    os.replace(tmp, dst)
    return dst


def fashion():
    for fn in FASHION_FILES:
        _fetch(FASHION_URL + fn, os.path.join(ROOT, "fashion", fn))


def emnist():
    dst = os.path.join(ROOT, "emnist", os.path.basename(EMNIST_MEMBER))
    if os.path.exists(dst):
        print(f"  have {os.path.basename(dst)}")
        return
    z = _fetch(EMNIST_ZIP_URL, os.path.join(ROOT, "emnist", "gzip.zip"))
    with zipfile.ZipFile(z) as zf, zf.open(EMNIST_MEMBER) as src, open(dst, "wb") as out:
        out.write(src.read())
    os.remove(z)                                   # only the one member is needed
    print(f"  extracted {os.path.basename(dst)}")


def cifar():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline", "model"))
    import cifar10_resnet_task as ct
    r = ct._raw()
    print(f"  cifar-10: train {r['train_x'].shape}, test {r['test_x'].shape}")


def svhn():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline", "model"))
    import cifar10_resnet_task as ct
    print(f"  svhn test: {ct._svhn_raw().shape}")


STEPS = {"fashion": fashion, "emnist": emnist, "cifar": cifar, "svhn": svhn}

if __name__ == "__main__":
    which = sys.argv[1:] or list(STEPS)
    unknown = [w for w in which if w not in STEPS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {list(STEPS)}")
    print(f"data root: {ROOT}")
    for w in which:
        print(f"[{w}]")
        STEPS[w]()
    print("done")
