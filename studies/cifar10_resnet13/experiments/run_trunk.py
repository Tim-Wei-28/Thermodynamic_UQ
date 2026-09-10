"""Job 1: check geometry and the data layer, train the trunk, then time the EBM head.

Writes trunk_<trunk>_e<epochs>_<schedule>_s<seed>.json and nper_probe_s<seed>.json, and
caches trunk parameters plus pooled features for later runs.  Options: --seeds, --trunk,
--epochs, --schedules, --nper-ladder, --probe-epochs, --synthetic, --tag, --skip-*.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

# lib/ holds this study's own modules, pipeline/model/ the trainer, ResNet and task.
import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import ART, ensure_art                                    # noqa: E402

import jax                                                           # noqa: E402
# resnet.py defaults to the CIFAR-100 width, so n_class is passed explicitly wherever a
# network is built rather than switched on the module.
import resnet                                       # noqa: E402
N_CLASS_TRUNK = 10
import cifar10_resnet_task as task                                         # noqa: E402
import ebm_head                                                      # noqa: E402


# ================================================================= 1 + 2: the cheap checks
def check_geometry():
    w, b, bn, tot = resnet.n_params(n_class=N_CLASS_TRUNK)
    print("=" * 78)
    print("GEOMETRY")
    print(f"  conv/dense weights {w:>10,}")
    print(f"  biases             {b:>10,}   (only on 6/9/12 -- they have no BatchNorm)")
    print(f"  batchnorm          {bn:>10,}")
    # The target is the 10-way net, so the check is exact: the sibling's count minus its
    # 100-way dense head.  Liu et al. report no CIFAR-10 number to compare against.
    SIBLING = 1_251_652                       # the CIFAR-100 trunk, 100-way head
    want = SIBLING - (100 - N_CLASS_TRUNK) * (resnet.N_FEAT + 1)
    print(f"  TOTAL              {tot:>10,}   CIFAR-100 sibling {SIBLING:,}")
    print(f"  doubled (BNN)      {2 * w + b + bn:>10,}   (the VI-BNN row's count)")
    print(f"  replaced Dense-{N_CLASS_TRUNK:<3}{N_CLASS_TRUNK * resnet.N_FEAT + N_CLASS_TRUNK:>8,}"
          f"   = W0 at D={resnet.N_FEAT + 1}")
    ok = tot == want
    print(f"  -> {'MATCHES' if ok else 'DOES NOT MATCH'} the sibling minus its dense head "
          f"({SIBLING:,} - {SIBLING - want:,} = {want:,})")
    if ok:
        print(f"     i.e. the convolutional stack is identical; only 256->{N_CLASS_TRUNK} "
              f"differs")
    return ok


def check_data():
    print("=" * 78)
    print("DATA")
    print(f"  DATA_ROOT {task.DATA_ROOT}")
    r = task._raw()
    tr, va = task._split_idx()
    print(f"  raw train {r['train_x'].shape}  test {r['test_x'].shape}")
    print(f"  stratified split: train {tr.size}  val {va.size}  (no overlap: "
          f"{not set(tr.tolist()) & set(va.tolist())})")
    task.configure(trunk="none", seed=0)
    for g in ("train", "val", "test"):
        X, Y = task._pool(g)
        h = np.bincount(Y, minlength=task.C)
        print(f"  {g:5s} {str(X.shape):>14}  per-class {h.min()}..{h.max()}  "
              f"range [{X.min():.3f},{X.max():.3f}]")
    S = task._svhn_raw()
    print(f"  svhn  {str(S.shape):>14}  dtype {S.dtype}  range [{S.min()},{S.max()}]")
    # The SVHN .mat is stored (H, W, C, N); a wrong transpose does not crash, so the images
    # are rendered as text and stay visible in the log.
    ramp = " .:-=+*#%@"

    def ascii_(x):
        a = np.asarray(x).reshape(task.IN_CH, task.SIDE, task.SIDE).mean(0)
        return "\n".join("    " + "".join(ramp[min(9, int(v * 9.999))] for v in row)
                         for row in a)

    Xte, Yte = task._pool("test")
    print(f"  --- CIFAR test image, label {Yte[0]}:")
    print(ascii_(Xte[0]))
    print("  --- SVHN image (must read as a house-number crop):")
    print(ascii_(S[0] / 255.0))
    for kind in task.PROBES:
        Xb, _ = task.PROBES[kind](Xte, Yte, 0.5, 4, 0)
        print(f"  --- probe {kind!r} at f=0.5  range [{Xb.min():.3f},{Xb.max():.3f}]:")
        print(ascii_(Xb[0]))
    return True


# ================================================================= 3: the trunk
def train_trunk(seed, epochs, trunk, force=False, schedule="constant"):
    """Train (or load) one trunk and report what it scores on the official test set."""
    ensure_art()          # idempotent
    out = os.path.join(ART, f"trunk_{trunk}_e{epochs}_{schedule}_s{seed}.json")
    if os.path.exists(out) and not force:
        d = json.load(open(out, encoding="utf-8"))
        print(f"  [seed {seed}] cached: test_acc {d['test_acc']:.4f}")
        return d
    # trunk='none' skips fitting the feature scale, which would train a second trunk here.
    # The pools depend only on CLASSES, so the cheap variant is enough.
    task.configure(trunk="none", seed=seed)
    Xtr, Ytr = task._pool("train")
    Xva, Yva = task._pool("val")
    Xte, Yte = task._pool("test")
    n_img = 45000 if trunk == "resnet45k" else 10000
    if n_img < Xtr.shape[0]:
        rng = np.random.default_rng(seed)
        idx = np.sort(np.concatenate(
            [rng.choice(np.flatnonzero(Ytr == c), n_img // task.C, replace=False)
             for c in range(task.C)]))
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    print(f"  [seed {seed}] training {trunk} on {Xtr.shape[0]} images, {epochs} epochs, "
          f"lr schedule {schedule!r}", flush=True)
    t0 = time.time()
    P, hist = resnet.train(Xtr, Ytr, epochs=epochs, seed=seed, verbose=True,
                           schedule=schedule, n_class=N_CLASS_TRUNK)
    jax.block_until_ready(P["d_w"])
    if tuple(P["d_w"].shape) != (N_CLASS_TRUNK, resnet.N_FEAT):
        raise RuntimeError(f"trunk head is {tuple(P['d_w'].shape)}, expected "
                           f"{(N_CLASS_TRUNK, resnet.N_FEAT)} -- n_class did not take")
    secs = time.time() - t0
    acc_te = resnet.accuracy(P, Xte, Yte)
    acc_va = resnet.accuracy(P, Xva, Yva)
    # Cache the parameters where the task's _trunk() looks for them.  The tag is built by the
    # task and carries the schedule, so only one place composes the filename.
    os.makedirs(task.TRUNK_DIR, exist_ok=True)
    ppath = os.path.join(task.TRUNK_DIR,
                         task._trunk_tag(trunk, epochs, schedule, seed) + ".npz")
    task._atomic_savez(ppath, {k: np.asarray(v) for k, v in P.items()})
    d = dict(seed=seed, trunk=trunk, epochs=epochs, schedule=schedule,
             n_train=int(Xtr.shape[0]),
             test_acc=acc_te, val_acc=acc_va, secs=secs,
             n_params=resnet.n_params()[3], params_path=ppath,
             loss=[h["loss"] for h in hist], lr=[h["lr"] for h in hist])
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, default=float)
    print(f"  [seed {seed}] test_acc {acc_te:.4f}  val_acc {acc_va:.4f}  "
          f"({secs / 60:.1f} min)  -> {ppath}", flush=True)
    # Warm the feature cache while the trunk is loaded, so no head run pays for a full
    # ResNet pass over the pools.
    task.configure(trunk=trunk, trunk_epochs=epochs, schedule=schedule, seed=seed,
                   n_per=4500)
    for g in ("train", "val", "test"):
        F = task._pool_features(g)
        print(f"  [seed {seed}] feature cache warmed: {g:5s} {F.shape}", flush=True)
    print(f"  [seed {seed}] trained on {Xtr.shape[0]} images; no published CIFAR-10 "
          f"reference exists, so this is judged on its own", flush=True)
    return d


# ================================================================= 4: the n_per ladder
def probe_nper(seed, trunk, epochs, ladder, e_lo, e_hi, synthetic=False, tag=""):
    """Wall time of one EBM head at each training budget, written to nper_probe_*.json.

       Two epoch counts report the per-epoch cost as their difference, so the one-off JIT
       compile stays out of the slope; one epoch count reports plain total wall time.
       `synthetic=True` uses random features of the same shape: exact for cost, because the
       trainer is full-batch with no data-dependent control flow, and says nothing about
       quality."""
    print("=" * 78)
    print(f"n_per LADDER  (K={ebm_head.HeadConfig.K}, r={ebm_head.HeadConfig.rank}, "
          f"C={task.C}, family=tree, est=sfe-loo"
          + (", SYNTHETIC features -- cost only" if synthetic else "") + ")")
    single = e_hi is None
    if single:
        print(f"{'n_per':>7} {'N':>8} {'D':>5} {'epochs':>8} {'total_s':>10} "
              f"{'total_min':>10}")
    else:
        print(f"{'n_per':>7} {'N':>8} {'D':>5} {'t@%d' % e_lo:>8} {'t@%d' % e_hi:>8} "
              f"{'s/epoch':>9} {'compile':>9} {'2400ep':>10}")
    rows, spec, dim = [], None, None
    for n_per in ladder:
        if synthetic:
            # D = 256 pooled activations plus the constant column that carries the bias.
            D_syn = resnet.N_FEAT + 1
            rng = np.random.default_rng(seed)
            n = n_per * task.C
            X = np.concatenate([rng.standard_normal((n, D_syn - 1)),
                                np.ones((n, 1))], 1)
            Y = np.repeat(np.arange(task.C), n_per).astype(np.int32)
            W0 = 0.1 * rng.standard_normal((task.C, D_syn))
        else:
            task.configure(trunk=trunk, trunk_epochs=epochs, seed=seed, n_per=n_per)
            X, Y = task.make_data(n_per, jax.random.key(0), group="train")
            W0 = task.w0_matrix(1.0, 0.1, seed=seed)      # alpha=1: the live configuration
        ts = {}
        for ep in ((e_lo,) if single else (e_lo, e_hi)):
            cfg = ebm_head.HeadConfig(epochs=ep, seed=seed)
            t0 = time.time()
            try:
                p, s, _ = ebm_head.train_head(X, Y, task.C, W0=W0, cfg=cfg, verbose=False)
                jax.block_until_ready(p["B"])
            except Exception as e:
                print(f"{n_per:>7}  FAILED at {ep} epochs: {type(e).__name__}: "
                      f"{str(e)[:60]}", flush=True)
                ts = None
                break
            ts[ep] = time.time() - t0
            spec, dim = s, int(X.shape[1])
        if ts is None:
            rows.append(dict(n_per=n_per, failed=True))
            continue
        if single:
            print(f"{n_per:>7} {X.shape[0]:>8} {X.shape[1]:>5} {e_lo:>8} "
                  f"{ts[e_lo]:>10.1f} {ts[e_lo] / 60:>10.1f}", flush=True)
            rows.append(dict(n_per=n_per, n=int(X.shape[0]), d=int(X.shape[1]),
                             epochs=e_lo, total_s=ts[e_lo],
                             total_min=ts[e_lo] / 60.0))
            continue
        per = (ts[e_hi] - ts[e_lo]) / (e_hi - e_lo)
        comp = ts[e_lo] - per * e_lo
        print(f"{n_per:>7} {X.shape[0]:>8} {X.shape[1]:>5} {ts[e_lo]:>8.1f} "
              f"{ts[e_hi]:>8.1f} {per:>9.3f} {comp:>9.1f} {per * 2400 / 60:>9.1f}m",
              flush=True)
        # Keep the raw times as well: compile time varies between runs, so the derived slope
        # can be pure noise and only t_lo/t_hi show it.
        rows.append(dict(n_per=n_per, n=int(X.shape[0]), d=int(X.shape[1]),
                         e_lo=e_lo, e_hi=e_hi, t_lo=ts[e_lo], t_hi=ts[e_hi],
                         s_per_epoch=per, compile_s=comp, est_2400_min=per * 2400 / 60))
    # The parameter budget once: it does not depend on n_per.  `spec` comes from whichever
    # rung succeeded.
    parts = {}
    if spec is not None:
        parts = ebm_head.n_trained(spec, dim, task.C)
        _det = resnet.n_params(n_class=N_CLASS_TRUNK)[3]
        print(f"\n  head parameters: {parts['total']:,} trainable "
              f"({100 * parts['total'] / _det:.1f} % of the {_det:,}-parameter DNN)")
        print("  " + "  ".join(f"{k}={v:,}" for k, v in parts.items() if k != "total"))
    # The tag goes into the filename so two probes sharing a working directory cannot
    # overwrite each other.
    suffix = ("_synth" if synthetic else "") + (f"_{tag}" if tag else "")
    ensure_art()          # idempotent
    out = os.path.join(ART, f"nper_probe_s{seed}{suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(seed=seed, synthetic=synthetic, tag=tag, rows=rows, params=parts),
                  f, indent=1, default=float)
    print(f"\n  -> {os.path.basename(out)}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--trunk", default="resnet45k")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--schedules", default="constant",
                    help="comma list of lr schedules to train a trunk under "
                         f"({'|'.join(resnet.SCHEDULES)}).  Each is a DIFFERENT trunk and "
                         f"gets its own cache entry and its own result JSON.")
    ap.add_argument("--nper-ladder", default="1000,2500,4500")
    ap.add_argument("--probe-epochs", default="50,150",
                    help="TWO values -> two short runs and their difference (the compile "
                         "cancels only if it is stable; measured 2026-08-21 it is NOT, and "
                         "three of six rungs came out negative).  ONE value -> a single run "
                         "at that epoch count, reporting total wall time.  Prefer the "
                         "single form at the real epoch count: nothing is subtracted, so "
                         "nothing can cancel wrongly.")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-trunk", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the n_per ladder on random features of the RIGHT SHAPE "
                         "instead of real trunk features.  Exact for cost (the trainer is "
                         "full-batch with no data-dependent control flow), meaningless for "
                         "quality.  Lets this block run parallel to the trunk job.")
    ap.add_argument("--tag", default="",
                    help="appended to the probe's result filename.  REQUIRED whenever two "
                         "probes share a working directory (the L40S and A100 twins do): "
                         "without it both write the same file and one silently overwrites "
                         "the other.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    ensure_art()

    print(f"JAX devices: {jax.devices()}", flush=True)
    ok_geo = check_geometry()
    check_data()
    if not ok_geo:
        raise SystemExit("parameter count does not match -- stopping before training")

    seeds = [int(s) for s in args.seeds.split(",")]
    scheds = [s.strip() for s in args.schedules.split(",") if s.strip()]
    if not args.skip_trunk:
        print("=" * 78)
        print(f"TRUNK  ({args.trunk}, {args.epochs} epochs, seeds {seeds}, "
              f"schedules {scheds})")
        results = []
        for sc in scheds:
            for s in seeds:
                results.append(train_trunk(s, args.epochs, args.trunk, force=args.force,
                                           schedule=sc))
        if len(results) > 1:
            print("\n" + "=" * 78)
            # 'gap' is the distance to the best schedule measured here; there is no published
            # CIFAR-10 number to measure against.
            best = max(results, key=lambda r: r["test_acc"])
            print(f"TRUNK COMPARISON   (no published CIFAR-10 reference; "
                  f"'gap' is against the best row below)")
            print(f"{'schedule':>10} {'seed':>5} {'test':>8} {'val':>8} {'gap':>8} "
                  f"{'final loss':>11} {'min':>6}")
            for d in sorted(results, key=lambda r: -r["test_acc"]):
                print(f"{d.get('schedule', '?'):>10} {d['seed']:>5} "
                      f"{d['test_acc']:>8.4f} {d['val_acc']:>8.4f} "
                      f"{d['test_acc'] - best['test_acc']:>+8.4f} "
                      f"{d['loss'][-1]:>11.4f} {d['secs'] / 60:>6.1f}")
            print(f"\n  best: {best.get('schedule')!r} at {best['test_acc']:.4f}")

    if not args.skip_probe:
        _pe = [int(x) for x in args.probe_epochs.split(",")]
        e_lo, e_hi = (_pe[0], None) if len(_pe) == 1 else (_pe[0], _pe[1])
        probe_nper(seeds[0], args.trunk, args.epochs,
                   [int(x) for x in args.nper_ladder.split(",")], e_lo, e_hi,
                   synthetic=args.synthetic, tag=args.tag)
    print("\nrun_trunk done", flush=True)


if __name__ == "__main__":
    main()
