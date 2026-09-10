"""Gate ablations on the heads run_seeds.py already trained: prior, matched dropout, all-on.

Only the gate distribution changes; the classifier (W0, B, A) is held identical, and kappa
for the dropout gate is read off the trained prior so the two have the same sparsity.
Writes results_controls[_<tag>].json.  Options: --seeds, --alphas, --n-per, --trunk-seed.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time

import numpy as np

import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import ART, ensure_art                                    # noqa: E402

import jax                                                           # noqa: E402
import resnet                                                        # noqa: E402
import cifar10_resnet_task as task                                    # noqa: E402
import cifar10_data as cd                                             # noqa: E402
import metrics_liu as mx                                              # noqa: E402

N_CLASS_TRUNK = 10


def _load_ebm(tag):
    """A head trained by run_seeds.py.  The tags are its own and carry the seed."""
    path = os.path.join(ART, f"ebm_{tag}.pkl")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n  -- run_controls.py trains nothing; this head comes from "
            f"run_seeds.py.  Run that first, for every alpha and seed asked for here.")
    with open(path, "rb") as f:
        return pickle.load(f)["params"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.7,0.75,0.8,0.9,0.95,1.0",
                    help="the alpha ladder figure 04 is drawn on.  Must be a SUBSET of "
                         "what run_seeds.py was given, because the heads come from there.")
    ap.add_argument("--trunk", default="resnet45k")
    ap.add_argument("--trunk-epochs", type=int, default=100)
    ap.add_argument("--schedule", default="step")
    ap.add_argument("--n-per", type=int, default=4500)
    ap.add_argument("--trunk-seed", type=int, default=0)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--skip-a1", action="store_true",
                    help="skip the raw-pixel head (it has its own preprocessing and is not "
                         "on figure 04's alpha axis)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    alphas = [float(a) for a in args.alphas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    ensure_art()

    print(f"JAX devices: {jax.devices()}", flush=True)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = cd.splits()
    Xte, Yte = Xte[:args.n_test], Yte[:args.n_test]

    # The feature front end is rebuilt through configure(), as run_seeds does, so the feature
    # scale is the one fitted on the images the head trained on.
    t_seed = args.trunk_seed
    task.configure(trunk=args.trunk, trunk_epochs=args.trunk_epochs,
                   schedule=args.schedule, seed=t_seed, n_per=args.n_per)
    det = task._trunk()
    scale = float(task._FEAT_SCALE)
    prep_a4 = lambda X: np.concatenate(                              # noqa: E731
        [resnet.features(det, X) * scale, np.ones((len(X), 1))], 1)
    task.configure(trunk="none", seed=t_seed)
    mean1 = np.asarray(task._MEAN_VEC, np.float64).copy()
    prep_a1 = lambda X: np.asarray(X, np.float64) - mean1            # noqa: E731
    task.configure(trunk=args.trunk, trunk_epochs=args.trunk_epochs,
                   schedule=args.schedule, seed=t_seed, n_per=args.n_per)

    out = os.path.join(ART, "results_controls"
                            + (f"_{args.tag}" if args.tag else "") + ".json")
    rows = []
    if os.path.exists(out):
        rows = json.load(open(out, encoding="utf-8"))["rows"]
        print(f"resuming: {len(rows)} row(s) already present", flush=True)
    have = {(r["model"], r["gate"], r["seed"]) for r in rows}

    probe_of = (lambda kind:
                lambda f, Xt, Yt, n, sd: cd.probe_raw(kind, f, Xt, Yt, n=n, seed=sd))

    for seed in seeds:
        models = [(f"A4 a={a:.2f}",
                   f"a4_al{a:.2f}_tree_n{args.n_per}_t{t_seed}_s{seed}", prep_a4)
                  for a in alphas]
        if not args.skip_a1:
            models.append(("A1 pixels", f"a1_tree_n{args.n_per}_s{seed}", prep_a1))

        for name, tag, prep in models:
            if all((name, g, seed) in have for g in ("prior", "dropout", "allon")):
                print(f"[seed {seed}] {name}: cached", flush=True)
                continue
            params = _load_ebm(tag)
            # kappa from this head's own prior, so the dropout control is matched.
            kappa = mx.prior_gate_rate(params, prep(Xte[:1000]))
            print(f"\n[seed {seed}] {name}: prior gate rate = {kappa:.4f}"
                  f"  (dropout matched to it)", flush=True)
            for gate in ("prior", "dropout", "allon"):
                if (name, gate, seed) in have:
                    continue
                t0 = time.time()
                lf = (lambda g: (lambda X: mx.sample_logits_ebm(
                    params, prep(X), gate=g, kappa=kappa)))(gate)
                # T* on the whole validation pool, as run_seeds fits it.
                T = mx.fit_T(lf(Xva), Yva)
                r = mx.summarise(f"{name} / {gate}", lf(Xte), Yte, T=T)
                ood = mx.ood_sweep(lf, Xte, Yte, (0.5,), probe_of("svhn"),
                                   n=args.n_ood, T=T, verbose=False)[0]
                r.update(model=name, gate=gate, probe="svhn", seed=seed,
                         acc_ood50=ood["acc"], ece_ood50=ood["ece"],
                         nll_ood50=ood["nll"], brier_ood50=ood["brier"],
                         h_epi_ood50=ood["h_epi"],
                         gate_rate=kappa)
                rows.append(r)
                print(f"   {gate:8s} acc {r['acc']:.4f} ECE {r['ece']:.4f} "
                      f"NLL {r['nll']:.4f} | f=0.5: acc {ood['acc']:.4f} "
                      f"ECE {ood['ece']:.4f} NLL {ood['nll']:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dict(settings=vars(args), rows=rows), f, indent=1, default=float)

    print(f"\nall done -> {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
