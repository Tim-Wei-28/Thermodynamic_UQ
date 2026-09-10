"""Per-seed task preparation for the campaign runners: trunk (trained or loaded from
cache), training subset, feature closure, weak random W0, val/test data and the OOD
blend probe.  Tasks: fashion (LeNet-5, letters blend) and cifar10 (ResNet, SVHN
blend).  Call bind(task) before prep().
"""
from __future__ import annotations
import os as _os
import sys as _sys
from types import SimpleNamespace

_HERE = _os.path.dirname(_os.path.abspath(_os.path.abspath(__file__)))
_DS = _os.path.dirname(_os.path.dirname(_HERE))
_TASK = None

TASKS = ("fashion", "cifar10")


def bind(task: str):
    """Register the task and make its extra lib importable."""
    global _TASK
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; expected one of {TASKS}")
    if task == "cifar10":
        lib = _os.path.join(_DS, "cifar10_resnet13", "lib")
        if lib not in _sys.path:
            _sys.path.append(lib)                        # appended: the fashion lib stays primary
    _TASK = task
    return task


def task_name():
    return _TASK or "fashion"


def defaults(task: str):
    """Per-task defaults for the runners' argparse."""
    if task == "cifar10":
        return dict(n_per=4500, trunk_epochs=100, trunk="resnet45k", schedule="step")
    return dict(n_per=1000, trunk_epochs=20, trunk=None, schedule=None)


def _prep_fashion(seed, n_per, trunk_epochs, trunk_width=1.0, **_):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import fashion_data as fd
    import lenet
    import ebm_head
    from paths import ART as LIU_ART

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
    # a width-scaled trunk carries its width in the cache tag
    wtag = "" if trunk_width == 1.0 else f"w{int(round(trunk_width * 100)):03d}_"
    tag = f"{wtag}n55000_e{trunk_epochs}_s{seed}"
    path = _os.path.join(LIU_ART, f"lenet_{tag}.npz")
    trunk = None
    if _os.path.exists(path):
        try:
            z = np.load(path)
            trunk = {k: jnp.asarray(z[k]) for k in z.files}
        except Exception:
            trunk = None                                  # unreadable cache: retrain
    if trunk is None:
        print(f"  [{tag}] training trunk", flush=True)
        trunk, _ = lenet.train(Xtr, Ytr, epochs=trunk_epochs, seed=seed,
                               Xva=Xva[:2000], Yva=Yva[:2000], verbose=False,
                               width=trunk_width)
        jax.block_until_ready(trunk["out_w"])
        tmp = path + f".{_os.getpid()}.tmp.npz"
        np.savez(tmp, **{k: np.asarray(v) for k, v in trunk.items()})
        _os.replace(tmp, path)                            # atomic write, parallel jobs share the cache

    Xm, Ym = fd.subset(Xtr, Ytr, n_per, seed=seed)
    F = lenet.features(trunk, Xm)
    W0t, b0 = lenet.head(trunk)
    Xa, _, W0_trained = ebm_head.prepare_features(F, [], W0=W0t, b0=b0)
    scale = 1.0 / (F.std() + 1e-12)
    prep = lambda X: np.concatenate(                                      # noqa: E731
        [lenet.features(trunk, X) * scale, np.ones((len(X), 1))], 1)
    rng = np.random.default_rng(seed)
    W0 = ebm_head.HeadConfig().w0_scale * rng.standard_normal(W0_trained.shape)
    return SimpleNamespace(task="fashion", C=fd.C, Xa=Xa, Ym=Ym, W0=W0, prep=prep,
                           Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte,
                           probe=fd.PROBES["letters"], blend_name="letters")


def _prep_cifar10(seed, n_per, trunk_epochs, trunk="resnet45k", schedule="step",
                  trunk_pool=None, trunk_width=1.0, **_):
    import time
    import numpy as np
    import jax
    import resnet
    import cifar10_resnet_task as task
    import cifar10_data as cd
    import ebm_head

    splits = cd.splits()
    (Xtr_all, Ytr_all), (Xva, Yva), (Xte, Yte) = splits
    # trunk seed = head seed; a width-scaled trunk carries its width in the cache tag
    task.configure(trunk="none", seed=seed)               # builds the pools only
    wtag = "" if trunk_width == 1.0 else f"w{int(round(trunk_width * 100)):03d}_"
    path = _os.path.join(task.TRUNK_DIR,
                         wtag + task._trunk_tag(trunk, trunk_epochs, schedule, seed)
                         + ".npz")
    if _os.path.exists(path):
        z = np.load(path)
        det = {k: np.asarray(z[k]) for k in z.files}
    else:
        Xp, Yp = task._pool("train")
        if trunk_pool:
            Xp, Yp = Xp[:trunk_pool], Yp[:trunk_pool]
        print(f"  [trunk s{seed}{'' if not wtag else ' ' + wtag[:-1]}] training "
              f"({schedule}, {trunk_epochs} ep"
              + (f", pool {trunk_pool}" if trunk_pool else "") + ")", flush=True)
        t0 = time.time()
        det, _ = resnet.train(Xp, Yp, epochs=trunk_epochs, seed=seed, verbose=False,
                              schedule=schedule, n_class=cd.C, width=trunk_width)
        jax.block_until_ready(det["d_w"])
        task._atomic_savez(path, {k: np.asarray(v) for k, v in det.items()})
        print(f"  [trunk s{seed}] done in {time.time() - t0:.0f}s", flush=True)
        det = {k: np.asarray(v) for k, v in det.items()}

    if trunk_width == 1.0:
        task.configure(trunk=trunk, trunk_epochs=trunk_epochs, schedule=schedule,
                       seed=seed, n_per=n_per)
        Xa, Ya = task.make_data(n_per, None, group="train")
        Xa, Ya = np.asarray(Xa), np.asarray(Ya)
        scale = float(task._FEAT_SCALE)
        prep = (lambda Xr: np.concatenate(
            [resnet.features(det, Xr) * scale, np.ones((len(Xr), 1))], 1))
        W0 = np.asarray(task.w0_matrix(1.0, ebm_head.HeadConfig().w0_scale, seed=seed))
    else:
        # task.configure knows no width, so its subset draw, feature scale and W0
        # convention are replicated here; the indices address the task's own pool,
        # not cd.splits(), so both widths train on the same images
        Xp_, Yp_ = task._pool("train")
        idx = np.asarray(task._draw_idx("train", int(n_per)))
        Xm, Ya = np.asarray(Xp_)[idx], np.asarray(Yp_)[idx]
        F = resnet.features(det, Xm)
        scale = 1.0 / (float(F.std()) + 1e-12)
        prep = (lambda Xr: np.concatenate(
            [resnet.features(det, Xr) * scale, np.ones((len(Xr), 1))], 1))
        Xa = np.concatenate([F * scale, np.ones((len(F), 1))], 1)
        W0 = (ebm_head.HeadConfig().w0_scale
              * np.random.default_rng(seed).standard_normal((cd.C, F.shape[1] + 1)))
    # argument order as ood_sweep calls it: blend(Xte, Yte, fraction, n=, seed=)
    probe = (lambda Xt, Yt, f, n=1000, seed=0:
             cd.probe_raw("svhn", f, Xt, Yt, n=n, seed=seed))
    return SimpleNamespace(task="cifar10", C=cd.C, Xa=Xa, Ym=Ya, W0=W0, prep=prep,
                           Xva=np.asarray(Xva), Yva=np.asarray(Yva),
                           Xte=np.asarray(Xte), Yte=np.asarray(Yte),
                           probe=probe, blend_name="svhn")


def prep(seed, n_per, trunk_epochs, **kw):
    """Per-seed bundle: Xa/Ym, W0, prep(raw)->features, val/test data, blend probe."""
    if _TASK is None:
        raise SystemExit("task_prep.bind(task) must be called before prep()")
    fn = _prep_fashion if _TASK == "fashion" else _prep_cifar10
    return fn(seed, n_per, trunk_epochs, **kw)
