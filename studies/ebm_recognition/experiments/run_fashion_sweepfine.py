"""Fine Gibbs sweep ladder S in {1..24} for four (K, r) cells, each with an
enum-trained ceiling head (recorded as sweeps=0) and the exact in-run gap probe.
Writes artifacts/{stem}_sweepfine_n{n_per}_s{seeds}.json.
Options: --task, --seeds, --cells, --sweeps, --ebm-epochs (smoke).
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
import fashion_data as fd               # noqa: E402
import lenet                            # noqa: E402
import ebm_head                         # noqa: E402
import metrics_liu as mx                # noqa: E402

ART = base.ART
CELLS = ((10, 1), (5, 1), (10, 2), (10, 8))          # (K, r)
SWEEPGRID = (1, 2, 3, 4, 5, 6, 8, 10, 16, 24)
N_CHAINS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fashion", choices=("fashion", "cifar10"),
                    help="fashion (byte-identical to the published study) | cifar10 "
                         "(the E4 road via task_prep: ResNet trunk, SVHN blend)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-per", type=int, default=None,
                    help="per-task default: 1000 (fashion) / 4500 (cifar10)")
    ap.add_argument("--lenet-epochs", type=int, default=20)
    ap.add_argument("--trunk-epochs", type=int, default=None,
                    help="cifar10 only: ResNet epochs (default 100)")
    ap.add_argument("--trunk-pool", type=int, default=None,
                    help="cifar10 smoke only: slice the trunk's training pool")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--n-probe", type=int, default=2000)
    ap.add_argument("--cells", default=None, help="e.g. '10:1,5:1' to restrict")
    ap.add_argument("--sweeps", default=None, help="restrict the S grid")
    ap.add_argument("--ebm-epochs", type=int, default=None, help="override for smoke tests")
    args = ap.parse_args()
    if args.ebm_epochs:
        base.BASE["epochs"] = args.ebm_epochs
    is_c10 = args.task == "cifar10"
    if args.n_per is None:
        args.n_per = 4500 if is_c10 else 1000
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = CELLS if not args.cells else tuple(
        tuple(int(x) for x in c.split(":")) for c in args.cells.split(","))
    sgrid = SWEEPGRID if not args.sweeps else tuple(
        int(s) for s in args.sweeps.split(","))
    os.makedirs(ART, exist_ok=True)
    stem = "cifar10" if is_c10 else "fashion"
    # cifar10 jobs split by cell, so a restricted cell set goes into the name
    cpart = ("" if not (is_c10 and args.cells)
             else "_c" + args.cells.replace(",", "-").replace(":", "x"))
    out_path = os.path.join(
        ART, f"{stem}_sweepfine_n{args.n_per}_s{args.seeds.replace(',', '-')}"
             f"{cpart}.json")
    tagpfx = "c10" if is_c10 else ""

    n = len(cells) * (len(sgrid) + 1) * len(seeds)    # +1: the enum ceiling head
    print("JAX devices:", jax.devices(), flush=True)
    print(f"{len(cells)} cells x ({len(sgrid)} sweeps + enum ceiling) x {len(seeds)} "
          f"seeds = {n} heads (cached ones skipped)", flush=True)

    if is_c10:
        import task_prep as tp
        tp.bind("cifar10")
    else:
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
        Xte, Yte = Xte[:args.n_test], Yte[:args.n_test]
    rows = []
    t0 = time.time()

    def _dump():
        with open(out_path, "w") as f:
            json.dump(dict(base=base.BASE, n_chains=N_CHAINS, settings=vars(args),
                           rows=rows), f, indent=1, default=float)

    for seed in seeds:
        if is_c10:
            bnd = tp.prep(seed, args.n_per,
                          args.trunk_epochs if args.trunk_epochs else 100,
                          trunk="resnet45k", schedule="step",
                          trunk_pool=args.trunk_pool)
            Xa, Ym, W0, C = bnd.Xa, bnd.Ym, bnd.W0, bnd.C
            Yva = np.asarray(bnd.Yva)
            Yte = np.asarray(bnd.Yte[:args.n_test])
            Xva_p = bnd.prep(bnd.Xva)
            Xte_p = bnd.prep(bnd.Xte[:args.n_test])
        else:
            trunk = base._trunk(seed, args.lenet_epochs, Xtr, Ytr,
                                Xva[:2000], Yva[:2000])
            Xm, Ym = fd.subset(Xtr, Ytr, args.n_per, seed=seed)
            F = lenet.features(trunk, Xm)
            W0t, b0 = lenet.head(trunk)
            Xa, _, W0_trained = ebm_head.prepare_features(F, [], W0=W0t, b0=b0)
            scale = 1.0 / (F.std() + 1e-12)
            prep = lambda X: np.concatenate(                              # noqa: E731
                [lenet.features(trunk, X) * scale, np.ones((len(X), 1))], 1)
            rng = np.random.default_rng(seed)
            W0 = ebm_head.HeadConfig().w0_scale * rng.standard_normal(W0_trained.shape)
            C = fd.C
            Xva_p, Xte_p = prep(Xva), prep(Xte)
        Xp, Yp = base._probe_subset(Xte_p, Yte, args.n_probe, seed)

        for (K, r) in cells:
            # sweeps=0 encodes the enum ceiling head of this cell
            for S in (0,) + sgrid:
                if S == 0:
                    fam, label = "ebm", "ebm"
                    tag = f"{tagpfx}sf_ebm_K{K}r{r}_n{args.n_per}_e2400_s{seed}"
                else:
                    fam, label = "ebm-gibbs", f"ebm-g{S}"
                    ebg.SWEEPS, ebg.N_CHAINS = S, N_CHAINS
                    # tag scheme of the coarse gibbs study, so its heads are reused from cache
                    tag = f"{tagpfx}trunk_{label}_K{K}r{r}_n{args.n_per}_e2400_s{seed}"
                if args.ebm_epochs:                    # smoke: separate cache namespace
                    tag = tag.replace("_e2400_", f"_e{args.ebm_epochs}_")
                cfg = ebm_head.HeadConfig(**{**base.BASE, "family": fam, "rank": r},
                                          K=K, seed=seed)
                params, spec, _par, secs, cached = base._train(
                    tag, label, Xa, Ym, C, W0, cfg)
                pj = base._to_jax(params)
                T = mx.fit_T(mx.sample_logits_ebm(pj, Xva_p), Yva)
                row = mx.summarise(label, mx.sample_logits_ebm(pj, Xte_p), Yte, T=T)
                row.update(base.exact_gap(pj, spec, fam, Xp, Yp, C, K))
                row.update(arm=label, approach="trunk", task=args.task,
                           K=K, r=r, seed=seed,
                           sweeps=S, family=fam,
                           n_chains=(0 if S == 0 else N_CHAINS),
                           n_pre=spec.n_pre, train_s=secs)
                rows.append(row)
                print(f"  s{seed} K={K:<3d}r={r} S={S:<3d} acc {row['acc']:.4f} "
                      f"ECE {row['ece']:.4f} Brier {row['brier']:.4f} | "
                      f"KL {row['kl_qp']:.4f}"
                      f"{'  (cached)' if cached else f'  {secs:.0f}s'}", flush=True)
                _dump()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
