"""Multi-seed driver for five base-table rows: A4 at alpha=1 and 0, A1 on pixels, the
deterministic ResNet and MC dropout.  Writes seed_<seed>_n<n_per>.json per seed.

Head seeds (--seeds) and trunk seeds (--trunk-seeds) are separate lists; the two ResNet rows
depend on the trunk seed only and are emitted once per trunk.  Trunks and heads are cached.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
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
import cifar10_data as cd                                           # noqa: E402
import ebm_head                                                      # noqa: E402
import metrics_liu as mx                                             # noqa: E402


# ================================================================= cached training
def _trunk(trunk, epochs, sched, seed):
    """The frozen ResNet: loaded from the cache run_trunk.py fills, trained here if absent."""
    task.configure(trunk="none", seed=seed)          # pools only -- no trunk side effect
    path = os.path.join(task.TRUNK_DIR,
                        task._trunk_tag(trunk, epochs, sched, seed) + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: np.asarray(z[k]) for k in z.files}
    Xtr, Ytr = task._pool("train")
    print(f"  [trunk s{seed}] training ({sched}, {epochs} ep) -- not in cache", flush=True)
    t0 = time.time()
    P, _ = resnet.train(Xtr, Ytr, epochs=epochs, seed=seed, verbose=False,
                        schedule=sched, n_class=N_CLASS_TRUNK)
    jax.block_until_ready(P["d_w"])
    task._atomic_savez(path, {k: np.asarray(v) for k, v in P.items()})
    print(f"  [trunk s{seed}] done in {time.time() - t0:.0f}s", flush=True)
    return {k: np.asarray(v) for k, v in P.items()}


def _mc_resnet(epochs, sched, seed, p_drop):
    """A second ResNet, trained with dropout active.  Dropout has to be present during
       training for the MC estimate to approximate anything, so this row costs its own
       network."""
    path = os.path.join(task.TRUNK_DIR,
                        f"mcdrop{p_drop}_e{epochs}_{sched}_s{seed}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: np.asarray(z[k]) for k in z.files}
    task.configure(trunk="none", seed=seed)
    Xtr, Ytr = task._pool("train")
    print(f"  [mcdrop s{seed}] training (p={p_drop}, {sched}, {epochs} ep)", flush=True)
    t0 = time.time()
    P, _ = resnet.train(Xtr, Ytr, epochs=epochs, seed=seed, verbose=False,
                        schedule=sched, p_drop=p_drop, n_class=N_CLASS_TRUNK)
    jax.block_until_ready(P["d_w"])
    task._atomic_savez(path, {k: np.asarray(v) for k, v in P.items()})
    print(f"  [mcdrop s{seed}] done in {time.time() - t0:.0f}s", flush=True)
    return {k: np.asarray(v) for k, v in P.items()}


def _head(tag, X, Y, W0, cfg):
    """One EBM head, cached by tag.  Stored as a pickle because `field` is a (kind, dict)
       tuple rather than a flat array set."""
    path = os.path.join(ART, f"ebm_{tag}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)["params"]
    print(f"  [{tag}] training EBM head", flush=True)
    t0 = time.time()
    params, spec, _ = ebm_head.train_head(X, Y, cd.C, W0=W0, cfg=cfg, verbose=False)
    jax.block_until_ready(params["B"])     # JAX dispatches async; time the real work
    print(f"  [{tag}] done in {time.time() - t0:.0f}s", flush=True)
    params = {k: (np.asarray(v) if k != "field"
                  else (v[0], {kk: np.asarray(vv) for kk, vv in v[1].items()}))
              for k, v in params.items()}
    tmp = path + f".part{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump({"params": params, "n_pre": getattr(spec, "n_pre", cfg.K)}, f)
    os.replace(tmp, path)
    return params


def _to_jax(p):
    import jax.numpy as jnp
    return {k: ((v[0], {kk: jnp.asarray(vv) for kk, vv in v[1].items()})
                if k == "field" else jnp.asarray(v))
            for k, v in p.items()}


# ================================================================= evaluation
def evaluate(name, logit_fn, Xte, Yte, Xva, Yva, probe_of, n_ood, fractions, verbose=False):
    """One row: in-distribution metrics, the full SVHN ladder, and the 50 % point of the
       two control corruptions.

       T* is fitted once on the validation split and carried unchanged to every reported
       set, since at deployment the shifted data has no labels."""
    T = mx.fit_T(logit_fn(Xva), Yva)
    row = mx.summarise(name, logit_fn(Xte), Yte, T=T)
    row["ood"] = mx.ood_sweep(logit_fn, Xte, Yte, fractions, probe_of("svhn"),
                              n=n_ood, T=T, verbose=verbose)
    m50 = [o for o in row["ood"] if abs(o["fraction"] - 0.5) < 1e-9][0]
    row.update(acc_ood50=m50["acc"], ece_ood50=m50["ece"], nll_ood50=m50["nll"],
               brier_ood50=m50["brier"], h_epi_ood50=m50["h_epi"])
    for kind in ("noise", "rotate"):
        o = mx.ood_sweep(logit_fn, Xte, Yte, (0.5,), probe_of(kind),
                         n=n_ood, T=T, verbose=False)[0]
        row[f"ece_{kind}50"] = o["ece"]
        row[f"nll_{kind}50"] = o["nll"]
    print(f"    {name:22s} acc {row['acc']:.4f}  ECE {row['ece']:.4f}  "
          f"NLL {row['nll']:.4f} | f=.5 acc {row['acc_ood50']:.4f} "
          f"ECE {row['ece_ood50']:.4f}", flush=True)
    return row


def run_seed(seed, args, splits, first_for_trunk):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = splits
    out = os.path.join(ART, f"seed_{seed}_n{args.n_per}.json")
    if os.path.exists(out) and not args.force:
        print(f"[seed {seed}] cached", flush=True)
        return
    t_seed = args.trunk_seeds[seed % len(args.trunk_seeds)]
    print(f"\n{'=' * 72}\n[seed {seed}]  head seed {seed}, trunk seed {t_seed}", flush=True)
    t0 = time.time()

    # The blend hands out raw images; each row applies its own front-end inside its
    # logit_fn, so every row sees the same pictures.
    probe_of = (lambda kind:
                lambda f, Xt, Yt, n, sd: cd.probe_raw(kind, f, Xt, Yt, n=n, seed=sd))

    rows = []

    # ---- the two ResNets -------------------------------------------------------------
    det = _trunk(args.trunk, args.trunk_epochs, args.schedule, t_seed)
    if first_for_trunk:
        # These two rows are functions of the trunk seed alone, so they are emitted once per
        # trunk.  The MC network is trained inside the branch because it feeds only this row.
        mc = _mc_resnet(args.trunk_epochs, args.schedule, t_seed, args.mc_drop)
        rows.append(evaluate("A2 ResNet det",
                             lambda X: mx.logits_det(resnet.logits(det, X)),
                             Xte, Yte, Xva, Yva, probe_of, args.n_ood, cd.FRACTIONS))
        rows.append(evaluate("MC dropout ResNet",
                             lambda X: resnet.mc_dropout_logits(mc, X, p_drop=args.mc_drop,
                                                                n_samp=mx.N_SAMP),
                             Xte, Yte, Xva, Yva, probe_of, args.n_ood, cd.FRACTIONS))
        for r in rows:
            r["trunk_seed"] = t_seed

    # ---- the EBM heads on ResNet features (approach 4) --------------------------------
    # configure() fits the feature scale on the images this run trains on and warms the
    # on-disk feature cache.
    task.configure(trunk=args.trunk, trunk_epochs=args.trunk_epochs,
                   schedule=args.schedule, seed=t_seed, n_per=args.n_per)
    Xa, Ya = task.make_data(args.n_per, None, group="train")
    Xa, Ya = np.asarray(Xa), np.asarray(Ya)
    # Front-end captured here, because the A1 block below reconfigures the task.  The trailing
    # column of ones is the constant feature that carries the bias.
    scale = float(task._FEAT_SCALE)
    prep_a4 = (lambda Xr: np.concatenate(
        [resnet.features(det, Xr) * scale, np.ones((len(Xr), 1))], 1))
    for alpha in args.alphas:
        W0 = task.w0_matrix(alpha, args.w0_scale, seed=seed)
        cfg = ebm_head.HeadConfig(epochs=args.ebm_epochs, seed=seed, K=args.K,
                                  rank=args.rank, family=args.family)
        p = _head(f"a4_al{alpha:.2f}_{args.family}_n{args.n_per}_t{t_seed}_s{seed}",
                  Xa, Ya, W0, cfg)
        pj = _to_jax(p)
        r = evaluate(f"A4 alpha={alpha:.2f}",
                     lambda X, _p=pj: mx.sample_logits_ebm(_p, prep_a4(X)),
                     Xte, Yte, Xva, Yva, probe_of, args.n_ood, cd.FRACTIONS)
        r.update(alpha=alpha, trunk_seed=t_seed,
                 # what the frozen base map scores on its own
                 acc_w0=float((prep_a4(Xte) @ np.asarray(W0).T).argmax(1)
                              .__eq__(np.asarray(Yte)).mean()))
        rows.append(r)

    # ---- approach 1 on raw pixels -----------------------------------------------------
    if not args.skip_a1:
        # A feature control, not a competitive row: a linear map on 3072 raw pixels.
        task.configure(trunk="none", seed=t_seed)
        mean1 = np.asarray(task._MEAN_VEC, np.float64).copy()
        prep_a1 = lambda Xr: np.asarray(Xr, np.float64) - mean1        # noqa: E731
        X1, Y1 = cd.subset(Xtr, Ytr, args.n_per, seed=seed)
        cfg = ebm_head.HeadConfig(epochs=args.ebm_epochs, seed=seed, K=args.K,
                                  rank=args.rank, family=args.family)
        p = _head(f"a1_{args.family}_n{args.n_per}_s{seed}",
                  prep_a1(X1), Y1, None, cfg)
        pj = _to_jax(p)
        r = evaluate("A1 pixels + EBM",
                     lambda X, _p=pj: mx.sample_logits_ebm(_p, prep_a1(X)),
                     Xte, Yte, Xva, Yva, probe_of, args.n_ood, cd.FRACTIONS)
        r["trunk_seed"] = None            # this row uses no trunk at all
        rows.append(r)

    ensure_art()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(seed=seed, trunk_seed=t_seed, settings=vars(args), rows=rows),
                  f, indent=1, default=float)
    print(f"[seed {seed}] finished in {(time.time() - t0) / 60:.1f} min -> {out}",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--trunk-seeds", default="0",
                    help="SHORT list.  A trunk costs 2.7 min and a head 7.1, so trunk seeds "
                         "are the reduced axis.  Head seed s uses trunk s %% len(this).")
    ap.add_argument("--alphas", default="1.0,0.0",
                    help="1.0 = weak random frozen W0 (the live mixture), 0.0 = the trained "
                         "Dense-10 (the collapse control).  These two are the base table; "
                         "a denser ladder belongs to the alpha figure, not here.")
    ap.add_argument("--trunk", default="resnet45k")
    ap.add_argument("--trunk-epochs", type=int, default=100)
    ap.add_argument("--schedule", default="step",
                    help="'step' is inherited from the CIFAR-100 study, where it was MEASURED to be "
                         "worth 4.2 accuracy points over 'constant' (0.6752 vs 0.6337). "
                         "Those are 100-class numbers and say nothing about the size of "
                         "the effect here; the schedule is held fixed so the two chapters "
                         "differ in the class count and not in the trunk recipe.  "
                         "indistinguishable at one seed; constant is 4 points worse.")
    ap.add_argument("--n-per", type=int, default=4500)
    ap.add_argument("--ebm-epochs", type=int, default=2400)
    ap.add_argument("--K", type=int, default=ebm_head.HeadConfig.K,
                    help="K = C = 10, the Fashion baseline (see lib/ebm_head.py)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--family", default="tree")
    ap.add_argument("--w0-scale", type=float, default=0.1)
    ap.add_argument("--mc-drop", type=float, default=0.5)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=10000,
                    help="test images used for the in-distribution columns.  The default is "
                         "the FULL official test set, which is what the comparison to their "
                         "Table 3 rests on; lower it only for wiring tests.")
    ap.add_argument("--skip-a1", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    args.alphas = [float(a) for a in args.alphas.split(",")]
    args.trunk_seeds = [int(s) for s in args.trunk_seeds.split(",")]
    ensure_art()

    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"trunk={args.trunk} {args.trunk_epochs}ep schedule={args.schedule}  "
          f"K={args.K} rank={args.rank} family={args.family} n_per={args.n_per} "
          f"ebm_epochs={args.ebm_epochs}", flush=True)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = cd.splits()
    splits = ((Xtr, Ytr), (Xva, Yva), (Xte[:args.n_test], Yte[:args.n_test]))
    print(f"data: train {splits[0][0].shape}  val {splits[1][0].shape}  "
          f"test {splits[2][0].shape}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    # Head seed s uses trunk_seeds[s % L], so the smallest head seed for a trunk is the trunk
    # index itself and "first for this trunk" is exactly s < L.  This needs no shared state,
    # which matters when the seeds are split across parallel jobs.
    L = len(args.trunk_seeds)
    for s in seeds:
        run_seed(s, args, splits, first_for_trunk=(s < L))
    print("\nall seeds done", flush=True)


if __name__ == "__main__":
    main()
