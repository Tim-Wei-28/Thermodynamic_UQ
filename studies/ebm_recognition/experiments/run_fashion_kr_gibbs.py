"""K x r capacity grid for the ebm-gibbs family (S = 12 sweeps, chains 64/32/16 by K)
or its tree/mf twin, downstream metrics only, on fashion or cifar10.  Writes
artifacts/{stem}_krg_n{n_per}_s{seeds}_K{ks}.json; per-head .pkl files are the cache.
Options: --task, --seeds, --ks, --rs, --family, --trunk-width, --approach, --ebm-epochs.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_H,
                 _os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "fashion_lenet5", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]

import run_fashion_families as base     # noqa: E402
import ebm_recognition_gibbs as ebg     # noqa: E402
import jax                              # noqa: E402
import ebm_head                         # noqa: E402
import metrics_liu as mx                # noqa: E402
import task_prep as tp                  # noqa: E402

ART = base.ART
SWEEPS = 12                              # fixed across the grid


def chains_for(K):
    """Chain budget by K; the sufficient-statistics tensors grow as K^2."""
    return 64 if K <= 20 else (32 if K <= 30 else 16)


OOD_KEYS = ("acc", "ece", "nll", "brier", "ece_cal", "nll_cal", "brier_cal", "h_epi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fashion", choices=tp.TASKS,
                    help="fashion (E2 road, letters blend) | cifar10 (E4 road, SVHN blend)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ks", default="1,2,3,4,5,6,8,10,15,20,30,50")
    ap.add_argument("--rs", default="1,2,3,4,6,8,10,16,24,36")
    ap.add_argument("--n-per", type=int, default=None, help="per-task default: 1000 / 4500")
    ap.add_argument("--trunk-epochs", type=int, default=None,
                    help="per-task default: 20 (LeNet) / 100 (ResNet)")
    ap.add_argument("--trunk-pool", type=int, default=None,
                    help="cifar10 smoke only: slice the trunk's training pool")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--family", default="ebm-gibbs",
                    help="ebm-gibbs (default) | tree | mf -- 'tree' runs the chain-tree "
                         "TWIN of the grid under the identical protocol (no chains, no "
                         "sweeps; closed forms), for the difference figure 08")
    ap.add_argument("--trunk-width", type=float, default=1.0,
                    help="fashion only: width-scaled LeNet trunk (F6 pinned at 84, see "
                         "lenet.shapes); 0.25 = the 6.7k-parameter weak backbone.  Own "
                         "trunk cache tag, own JSON stem, own head-tag prefix")
    ap.add_argument("--approach", default="trunk", choices=("trunk", "a1"),
                    help="fashion only: a1 = NO trunk -- mean-centred raw pixels "
                         "(D=784), W0 drawn weak random inside train_head (the "
                         "run_fashion_a1 prep, byte-identical)")
    ap.add_argument("--ebm-epochs", type=int, default=None, help="override for smoke tests")
    args = ap.parse_args()
    if args.ebm_epochs:
        base.BASE["epochs"] = args.ebm_epochs
    if args.approach == "a1" and args.trunk_width != 1.0:
        raise SystemExit("--approach a1 has no trunk; --trunk-width is meaningless")
    if args.task == "cifar10" and args.trunk_width not in (1.0, 0.25):
        # only these widths have a file stem known downstream
        raise SystemExit("cifar10 supports --trunk-width 1.0 or 0.25")
    tp.bind(args.task)
    dflt = tp.defaults(args.task)
    n_per = args.n_per if args.n_per is not None else dflt["n_per"]
    trunk_epochs = (args.trunk_epochs if args.trunk_epochs is not None
                    else dflt["trunk_epochs"])
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]
    rs = [int(r) for r in args.rs.split(",")]
    fam = args.family
    is_gibbs = fam == "ebm-gibbs"
    famtag = f"ebm-g{SWEEPS}" if is_gibbs else fam
    # each task and backbone variant owns its file stem and cache-tag prefix
    vtag = ""
    if args.approach == "a1":
        vtag = "a1"
    elif args.trunk_width != 1.0:
        vtag = f"w{int(round(args.trunk_width * 100)):03d}"
    stem = ("fashion" if args.task == "fashion" else "cifar10") \
        + (f"_{vtag}" if vtag else "")
    tagpfx = (f"{vtag}krg" if args.task == "fashion" else f"{vtag}c10krg")
    os.makedirs(ART, exist_ok=True)
    # a non-default r set goes into the name too, so jobs split along r never share a file
    rpart = ("" if args.rs == ap.get_default("rs")
             else f"_r{args.rs.replace(',', '-')}")
    out_path = os.path.join(
        ART, f"{stem}_krg_n{n_per}_s{args.seeds.replace(',', '-')}"
             f"_K{args.ks.replace(',', '-')}" + rpart
             + ("" if is_gibbs else f"_{fam}") + ".json")

    print("JAX devices:", jax.devices(), flush=True)
    print(f"task={args.task}: {len(ks)} K x {len(rs)} r x {len(seeds)} seeds = "
          f"{len(ks) * len(rs) * len(seeds)} heads, family={fam}"
          + (f", S={SWEEPS}" if is_gibbs else ""), flush=True)

    rows = []
    t0 = time.time()

    def _dump():
        with open(out_path, "w") as f:
            json.dump(dict(base=base.BASE, sweeps=SWEEPS, settings=vars(args),
                           rows=rows), f, indent=1, default=float)

    for seed in seeds:
        if args.approach == "a1" and args.task == "fashion":
            # pixel prep: mean-centred on the training subset, W0 drawn inside train_head
            from types import SimpleNamespace
            import fashion_data as fd
            (Xtr_, Ytr_), (Xva_, Yva_), (Xte_, Yte_) = fd.splits()
            Xm, Ym_ = fd.subset(Xtr_, Ytr_, n_per, seed=seed)
            mu = np.asarray(Xm, np.float64).mean(0)
            b = SimpleNamespace(
                Xa=np.asarray(Xm, np.float64) - mu, Ym=Ym_, C=fd.C, W0=None,
                prep=(lambda X, _mu=mu: np.asarray(X, np.float64) - _mu),
                Xva=Xva_, Yva=Yva_, Xte=Xte_, Yte=Yte_,
                probe=fd.PROBES["letters"])
        elif args.approach == "a1":
            # cifar10 pixel prep: centred on the task's full-pool mean, not the subset mean
            from types import SimpleNamespace
            import cifar10_resnet_task as task
            import cifar10_data as cd
            task.configure(trunk="none", seed=seed)
            mean1 = np.asarray(task._MEAN_VEC, np.float64).copy()
            (Xtr_, Ytr_), (Xva_, Yva_), (Xte_, Yte_) = cd.splits()
            Xm, Ym_ = cd.subset(Xtr_, Ytr_, n_per, seed=seed)
            b = SimpleNamespace(
                Xa=np.asarray(Xm, np.float64) - mean1, Ym=Ym_, C=cd.C, W0=None,
                prep=(lambda X, _mu=mean1: np.asarray(X, np.float64) - _mu),
                Xva=Xva_, Yva=Yva_, Xte=Xte_, Yte=Yte_,
                probe=(lambda Xt, Yt, f, n=1000, seed=0:
                       cd.probe_raw("svhn", f, Xt, Yt, n=n, seed=seed)))
        else:
            b = tp.prep(seed, n_per, trunk_epochs,
                        trunk=dflt.get("trunk"), schedule=dflt.get("schedule"),
                        trunk_pool=args.trunk_pool, trunk_width=args.trunk_width)
        Xa, Ym, W0, prep = b.Xa, b.Ym, b.W0, b.prep
        Yva, Yte = b.Yva, b.Yte[:args.n_test]
        Xte = b.Xte[:args.n_test]
        Xva_p, Xte_p = prep(b.Xva), prep(Xte)

        for K in ks:
            if is_gibbs:
                ebg.SWEEPS, ebg.N_CHAINS = SWEEPS, chains_for(K)
            for r in rs:
                cfg = ebm_head.HeadConfig(**{**base.BASE,
                                             "family": fam, "rank": r},
                                          K=K, seed=seed)
                tag = f"{tagpfx}_{famtag}_K{K}r{r}_n{n_per}_e{cfg.epochs}_s{seed}"
                params, spec, _par, secs, cached = base._train(
                    tag, fam, Xa, Ym, b.C, W0, cfg)
                pj = base._to_jax(params)

                T = mx.fit_T(mx.sample_logits_ebm(pj, Xva_p), Yva)
                row = mx.summarise(fam, mx.sample_logits_ebm(pj, Xte_p),
                                   Yte, T=T)
                lf = lambda X, _p=pj: mx.sample_logits_ebm(_p, prep(X))   # noqa: E731
                o = mx.ood_sweep(lf, Xte, Yte, (0.5,), b.probe,
                                 n=args.n_ood, T=T, verbose=False)[0]
                for m in OOD_KEYS:
                    if o.get(m) is not None:
                        row[f"{m}_ood50"] = o[m]
                row.update(arm=fam, task=args.task, K=K, r=r, seed=seed,
                           sweeps=(SWEEPS if is_gibbs else 0),
                           n_chains=(chains_for(K) if is_gibbs else 0),
                           n_pre=getattr(spec, "n_pre", K), train_s=secs,
                           family=fam,
                           n_trained=ebm_head.n_trained(
                               params, type("S", (), {"n_pre": getattr(spec, "n_pre", K)}),
                               Xa.shape[1], b.C, cfg)["total"])
                rows.append(row)
                print(f"  s{seed} K={K:<3d}r={r:<3d} acc {row['acc']:.4f} "
                      f"ECE {row['ece']:.4f} NLL {row['nll']:.4f} "
                      f"Brier {row['brier']:.4f} | ood50 acc {row.get('acc_ood50', float('nan')):.4f}"
                      f"{'  (cached)' if cached else f'  {secs:.0f}s'}", flush=True)
                _dump()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
