"""Tree topology arms (chain, binary, star, Chow-Liu) next to mf and ebm at K=10,
r=8, each with the exact in-run probe, plus the 6 x 4 refit ladder on the frozen
thetas.  Writes artifacts/fashion_topo_n{n_per}.json (cifar10:
cifar10_topo_n{n_per}_s{seeds}.json).  Options: --task, --seeds, --k, --r, --n-ladder.
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
import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402
import fashion_data as fd               # noqa: E402
import lenet                            # noqa: E402
import ebm_head                         # noqa: E402
import field as fieldmod                # noqa: E402
import recognition as recog             # noqa: E402
import ebm_recognition as ebm           # noqa: E402

ART = base.ART

ARMS = [
    ("mf",          dict(family="mf")),
    ("tree-chain",  dict(family="tree", topology="chain")),
    ("tree-binary", dict(family="tree", topology="binary")),
    ("tree-star",   dict(family="tree", topology="star")),
    ("mst-warm-mf", dict(family="tree_max_span", topology="chain", refit="once",
                         struct_warmup_family="mf")),
    ("ebm",         dict(family="ebm")),
]
LADDER_FAMS = [("mf", "chain"), ("tree", "chain"), ("tree", "binary"),
               ("tree", "star"), ("tree-mst", None), ("ebm", "chain")]
LADDER_SRCS = ("mf", "tree-chain", "mst-warm-mf", "ebm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fashion", choices=("fashion", "cifar10"),
                    help="fashion (the E2 road, byte-identical to the published study) | "
                         "cifar10 (the E4 road via task_prep: ResNet trunk, SVHN blend)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--n-per", type=int, default=None,
                    help="per-task default: 1000 (fashion) / 4500 (cifar10)")
    ap.add_argument("--lenet-epochs", type=int, default=20)
    ap.add_argument("--trunk-epochs", type=int, default=None,
                    help="cifar10 only: ResNet epochs (default 100)")
    ap.add_argument("--trunk-pool", type=int, default=None,
                    help="cifar10 smoke only: slice the trunk's training pool")
    ap.add_argument("--n-probe", type=int, default=2000)
    ap.add_argument("--n-ladder", type=int, default=500)
    ap.add_argument("--ladder-steps", type=int, default=2500)
    ap.add_argument("--ebm-epochs", type=int, default=None, help="override for smoke tests")
    args = ap.parse_args()
    if args.ebm_epochs:
        base.BASE["epochs"] = args.ebm_epochs
    is_c10 = args.task == "cifar10"
    if args.n_per is None:
        args.n_per = 4500 if is_c10 else 1000
    K, r = args.k, args.r
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(ART, exist_ok=True)
    # the fashion name carries no seeds (existing readers depend on it); the cifar10 one does
    out_path = os.path.join(
        ART, (f"cifar10_topo_n{args.n_per}_s{args.seeds.replace(',', '-')}.json"
              if is_c10 else f"fashion_topo_n{args.n_per}.json"))
    tagpfx = "c10topo_" if is_c10 else ""

    print("JAX devices:", jax.devices(), flush=True)
    if not is_c10:
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
    else:
        import task_prep as tp
        tp.bind("cifar10")
    rows, ladder = [], []
    t0 = time.time()

    def _dump():
        with open(out_path, "w") as f:
            json.dump(dict(base=base.BASE, K=K, r=r, settings=vars(args),
                           rows=rows, ladder=ladder), f, indent=1, default=float)

    for seed in seeds:
        if is_c10:
            # same bundle as the K x r grid, so these arms pair with its cells
            bnd = tp.prep(seed, args.n_per,
                          args.trunk_epochs if args.trunk_epochs else 100,
                          trunk="resnet45k", schedule="step",
                          trunk_pool=args.trunk_pool)
            Xa, Ym, W0, C = bnd.Xa, bnd.Ym, bnd.W0, bnd.C
            Xte_p = bnd.prep(bnd.Xte[:10000])
            Yte_l = np.asarray(bnd.Yte[:10000])
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
            Xte_p = prep(Xte[:10000])
            Yte_l = Yte[:10000]
        Xp, Yp = base._probe_subset(Xte_p, Yte_l, args.n_probe, seed)
        Xl, Yl = base._probe_subset(Xte_p, Yte_l, args.n_ladder, seed + 1000)
        featl = jnp.concatenate([Xl, jax.nn.one_hot(Yl, C)], 1)

        theta = {}
        for label, override in ARMS:
            cfg = ebm_head.HeadConfig(**{**base.BASE, **override, "rank": r},
                                      K=K, seed=seed)
            tag = f"{tagpfx}{label}_K{K}_n{args.n_per}_e{cfg.epochs}_s{seed}"
            params, spec, parents, secs, cached = base._train(
                tag, label, Xa, Ym, C, W0, cfg)
            pj = base._to_jax(params)
            if label in LADDER_SRCS:
                theta[label] = (pj, spec, override["family"])
            row = dict(arm=label, K=K, r=r, seed=seed, family=override["family"],
                       task=args.task,
                       topology=override.get("topology"), train_s=secs)
            row.update(base.exact_gap(pj, spec, override["family"], Xp, Yp, C, K))
            if parents is not None:
                row["parents"] = parents
            rows.append(row)
            print(f"  s{seed} {label:12s} KL(q||p) {row['kl_qp']:.4f} "
                  f"corr_post {row['corr_post']:.4f} lost {row['corr_lost']:.4f}"
                  f"{'  (cached)' if cached else f'  {secs:.0f}s'}", flush=True)
            _dump()

        # the 6 x 4 ladder
        for i_s, (src, (pj, spec, family)) in enumerate(theta.items()):
            h = fieldmod.apply(pj["field"][0], pj["field"][1], Xl)
            logp = gt.true_posterior_logp(pj["W0"], pj["B"], pj["A"], Xl, Yl,
                                          h, pj["J"], ebm.make_ebm_spec(K))
            mst_sr = gt.mst_spec_from_logp(logp, K)          # topology from the target posterior
            for i_f, (fam, topo) in enumerate(LADDER_FAMS):
                for i_m, mode in enumerate(("amortized", "free")):
                    key = jax.random.fold_in(jax.random.key(23),
                                             seed * 1000 + i_s * 100 + i_f * 10 + i_m)
                    if fam == "tree-mst":
                        met, _ = gt.fit_family(key, "tree", K, featl, logp,
                                               amortized=(mode == "amortized"),
                                               spec_R=mst_sr, steps=args.ladder_steps)
                    else:
                        met, _ = gt.fit_family(key, fam, K, featl, logp,
                                               amortized=(mode == "amortized"),
                                               topology=topo, steps=args.ladder_steps)
                    lab = fam if topo in (None, "chain") else f"{fam}-{topo}"
                    ladder.append(dict(seed=seed, src=src, fam=lab, mode=mode, **met))
                    print(f"  s{seed} ladder theta_{src:11s} {lab:12s} {mode:9s} "
                          f"KL {met['kl_mean']:.4f}", flush=True)
                    _dump()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
