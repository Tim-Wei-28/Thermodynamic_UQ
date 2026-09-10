"""Mechanism check on a task: part A trains mf, tree-chain and ebm at K=10, r=8 with
exact probes and the refit ladder; part B trains ebm-gibbs over S in {2,6,12,24} x
r in {1,8} with enum ceilings.  Writes artifacts/{task}_mech_n{n_per}_s{seeds}.json.
Options: --task, --seeds, --k, --r, --skip-gibbs, --skip-ladder, --ebm-epochs (smoke).
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
import gap_tools as gt                  # noqa: E402
import ebm_recognition as ebm           # noqa: E402
import ebm_recognition_gibbs as ebg     # noqa: E402
import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402
import ebm_head                         # noqa: E402
import field as fieldmod                # noqa: E402
import metrics_liu as mx                # noqa: E402
import task_prep as tp                  # noqa: E402

ART = base.ART
ARMS = [("mf",         dict(family="mf")),
        ("tree-chain", dict(family="tree", topology="chain")),
        ("ebm",        dict(family="ebm"))]
LADDER_FAMS = ("mf", "tree", "ebm")
SWEEPS_GRID = (2, 6, 12, 24)
GIBBS_RS = (1, 8)
N_CHAINS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cifar10", choices=tp.TASKS)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--r", type=int, default=8, help="operating-point rank (part A)")
    ap.add_argument("--n-per", type=int, default=None)
    ap.add_argument("--trunk-epochs", type=int, default=None)
    ap.add_argument("--trunk-pool", type=int, default=None, help="smoke only")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-probe", type=int, default=2000)
    ap.add_argument("--n-ladder", type=int, default=500)
    ap.add_argument("--ladder-steps", type=int, default=2500)
    ap.add_argument("--skip-gibbs", action="store_true")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--ebm-epochs", type=int, default=None, help="override for smoke tests")
    args = ap.parse_args()
    if args.ebm_epochs:
        base.BASE["epochs"] = args.ebm_epochs
    tp.bind(args.task)
    dflt = tp.defaults(args.task)
    n_per = args.n_per if args.n_per is not None else dflt["n_per"]
    trunk_epochs = (args.trunk_epochs if args.trunk_epochs is not None
                    else dflt["trunk_epochs"])
    K, r0 = args.k, args.r
    seeds = [int(s) for s in args.seeds.split(",")]
    pfx = "c10mech" if args.task == "cifar10" else "fashmech"
    os.makedirs(ART, exist_ok=True)
    out_path = os.path.join(
        ART, f"{args.task}_mech_n{n_per}_s{args.seeds.replace(',', '-')}.json")

    print("JAX devices:", jax.devices(), flush=True)
    rows, ladder = [], []
    t0 = time.time()

    def _dump():
        with open(out_path, "w") as f:
            json.dump(dict(base=base.BASE, K=K, r=r0, n_chains=N_CHAINS,
                           settings=vars(args), rows=rows, ladder=ladder),
                      f, indent=1, default=float)

    def run_head(bundle, label, fam, rr, seed, sweeps=None):
        """Train one head (cached by tag); measure clean, blend and exact probe."""
        cfg = ebm_head.HeadConfig(**{**base.BASE, "family": fam, "rank": rr},
                                  K=K, seed=seed)
        tag = f"{pfx}_{label}_K{K}r{rr}_n{n_per}_e{cfg.epochs}_s{seed}"
        if sweeps is not None:
            ebg.SWEEPS, ebg.N_CHAINS = sweeps, N_CHAINS
        params, spec, _par, secs, cached = base._train(
            tag, label, bundle.Xa, bundle.Ym, bundle.C, bundle.W0, cfg)
        pj = base._to_jax(params)
        T = mx.fit_T(mx.sample_logits_ebm(pj, bundle.Xva_p), bundle.Yva)
        row = mx.summarise(label, mx.sample_logits_ebm(pj, bundle.Xte_p),
                           bundle.Yte_c, T=T)
        lf = lambda X, _p=pj: mx.sample_logits_ebm(_p, bundle.prep(X))    # noqa: E731
        o = mx.ood_sweep(lf, bundle.Xte_c, bundle.Yte_c, (0.5,), bundle.probe,
                         n=args.n_ood, T=T, verbose=False)[0]
        for m in ("acc", "ece", "nll", "brier", "ece_cal", "nll_cal", "brier_cal",
                  "h_epi"):
            if o.get(m) is not None:
                row[f"{m}_ood50"] = o[m]
        row.update(base.exact_gap(pj, spec, fam, bundle.Xp, bundle.Yp, bundle.C, K))
        if fam.startswith("ebm"):
            jstat, _cm = base.jq_stats(pj, spec, bundle.Xp, bundle.Yp, bundle.C, K)
            row.update(jstat)
        row.update(arm=label, task=args.task, K=K, r=rr, seed=seed, family=fam,
                   sweeps=(0 if sweeps is None else sweeps),
                   n_chains=(0 if sweeps is None else N_CHAINS), train_s=secs)
        rows.append(row)
        print(f"  s{seed} {label:11s} r={rr} S={row['sweeps']:<3d} "
              f"acc {row['acc']:.4f} ECE {row['ece']:.4f} Brier {row['brier']:.4f} | "
              f"KL {row['kl_qp']:.4f} corr_post {row['corr_post']:.4f}"
              f"{'  (cached)' if cached else f'  {secs:.0f}s'}", flush=True)
        _dump()
        return pj, spec, fam

    for seed in seeds:
        b = tp.prep(seed, n_per, trunk_epochs,
                    trunk=dflt.get("trunk"), schedule=dflt.get("schedule"),
                    trunk_pool=args.trunk_pool)
        b.Xte_c, b.Yte_c = b.Xte[:args.n_test], b.Yte[:args.n_test]
        b.Xva_p, b.Xte_p = b.prep(b.Xva), b.prep(b.Xte_c)
        b.Xp, b.Yp = base._probe_subset(b.Xte_p, b.Yte_c, args.n_probe, seed)

        # part A: families at the operating point
        theta = {}
        for label, override in ARMS:
            fam = override["family"]
            pj, spec, _ = run_head(b, label, fam, r0, seed)
            theta[label] = (pj, spec)

        if not args.skip_ladder:
            Xl, Yl = base._probe_subset(b.Xte_p, b.Yte_c, args.n_ladder, seed + 1000)
            featl = jnp.concatenate([Xl, jax.nn.one_hot(Yl, b.C)], 1)
            for i_s, (src, (pj, spec)) in enumerate(theta.items()):
                h = fieldmod.apply(pj["field"][0], pj["field"][1], Xl)
                logp = gt.true_posterior_logp(pj["W0"], pj["B"], pj["A"], Xl, Yl,
                                              h, pj["J"], ebm.make_ebm_spec(K))
                for i_f, fam in enumerate(LADDER_FAMS):
                    for i_m, mode in enumerate(("amortized", "free")):
                        key = jax.random.fold_in(jax.random.key(31),
                                                 seed * 1000 + i_s * 100 + i_f * 10 + i_m)
                        met, _ = gt.fit_family(key, fam, K, featl, logp,
                                               amortized=(mode == "amortized"),
                                               steps=args.ladder_steps)
                        ladder.append(dict(seed=seed, src=src, fam=fam, mode=mode,
                                           **met))
                        print(f"  s{seed} ladder theta_{src:10s} {fam:4s} {mode:9s} "
                              f"KL {met['kl_mean']:.4f}", flush=True)
                        _dump()

        # part B: Gibbs saturation
        if not args.skip_gibbs:
            for rr in GIBBS_RS:
                if rr != r0:                       # the r0 ceiling is part A's ebm arm
                    run_head(b, "ebm", "ebm", rr, seed)
                for S in SWEEPS_GRID:
                    run_head(b, f"ebm-g{S}", "ebm-gibbs", rr, seed, sweeps=S)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
