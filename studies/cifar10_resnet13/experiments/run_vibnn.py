"""The variational baseline row: trains the mean-field ResNet and scores it through the
same metric path as every other row.

One row per (kl_weight, seed), cached and written to results_vibnn[_<tag>].json.  Options:
--seeds, --kl-weights, --epochs, --schedule, --sigma-max, --n-ood, --n-test, --tag.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import ART, ensure_art                                    # noqa: E402

import jax                                                           # noqa: E402
import cifar10_resnet_task as task                                         # noqa: E402
import cifar10_data as cd                                           # noqa: E402
import vi_bnn_resnet as vb                                           # noqa: E402
import metrics_liu as mx                                             # noqa: E402


def _net(klw, seed, args, Xtr, Ytr):
    """Train (or load) one variational network, cached by (kl_weight, seed)."""
    tag = f"vibnn_kl{klw}_e{args.epochs}_{args.schedule}_s{seed}"
    path = os.path.join(task.TRUNK_DIR, tag + ".npz")
    if os.path.exists(path) and not args.force:
        z = np.load(path)
        print(f"  [{tag}] cached", flush=True)
        return {k: np.asarray(z[k]) for k in z.files}
    print(f"  [{tag}] training on {Xtr.shape[0]} images, {args.epochs} epochs", flush=True)
    t0 = time.time()
    P, hist = vb.train(Xtr, Ytr, epochs=args.epochs, seed=seed, kl_weight=klw,
                       schedule=args.schedule, sigma_max=args.sigma_max, verbose=True)
    jax.block_until_ready(P["d_mu"])
    task._atomic_savez(path, {k: np.asarray(v) for k, v in P.items()})
    print(f"  [{tag}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return {k: np.asarray(v) for k, v in P.items()}


def evaluate(name, P, splits, args):
    """In-distribution metrics, the full SVHN ladder and the two control corruptions, on the
       protocol run_seeds uses for every other row."""
    (_Xtr, _Ytr), (Xva, Yva), (Xte, Yte) = splits
    lf = lambda X: vb.sample_logits(P, X, n_samp=mx.N_SAMP,          # noqa: E731
                                    sigma_max=args.sigma_max)
    probe_of = (lambda kind:
                lambda f, Xt, Yt, n, sd: cd.probe_raw(kind, f, Xt, Yt, n=n, seed=sd))
    T = mx.fit_T(lf(Xva), Yva)
    row = mx.summarise(name, lf(Xte), Yte, T=T)
    row["ood"] = mx.ood_sweep(lf, Xte, Yte, cd.FRACTIONS, probe_of("svhn"),
                              n=args.n_ood, T=T, verbose=False)
    m50 = [o for o in row["ood"] if abs(o["fraction"] - 0.5) < 1e-9][0]
    row.update(acc_ood50=m50["acc"], ece_ood50=m50["ece"], nll_ood50=m50["nll"],
               brier_ood50=m50["brier"], h_epi_ood50=m50["h_epi"])
    for kind in ("noise", "rotate"):
        o = mx.ood_sweep(lf, Xte, Yte, (0.5,), probe_of(kind),
                         n=args.n_ood, T=T, verbose=False)[0]
        row[f"ece_{kind}50"] = o["ece"]
        row[f"nll_{kind}50"] = o["nll"]
    print(f"    {name:22s} acc {row['acc']:.4f}  ECE {row['ece']:.4f}  "
          f"NLL {row['nll']:.4f} | f=.5 acc {row['acc_ood50']:.4f} "
          f"ECE {row['ece_ood50']:.4f}", flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--kl-weights", default=str(vb.KL_WEIGHT_PAPER),
                    help="the paper's calibration knob (their Figure 6D).  0.2 is the value "
                         "they quote for CIFAR-10 and the one the base table uses; a list "
                         "reproduces their trade-off curve.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--schedule", default="step")
    ap.add_argument("--sigma-max", type=float, default=None,
                    help="the paper caps sigma during training but gives NO value.  Default "
                         "is no cap, which is the stronger software baseline; set it only "
                         "to check what the constraint would cost.")
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--tag", default="",
                    help="appended to the result filename.  REQUIRED when the seeds are "
                         "split across parallel jobs: they share a working directory and "
                         "each block rewrites the whole file.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    args.kl_weights = [float(x) for x in args.kl_weights.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    ensure_art()

    print(f"JAX devices: {jax.devices()}", flush=True)
    mu, rho, det, tot = vb.n_params()
    # Iso-topology with the deterministic net, checked as a relation rather than against a
    # literal count: every kernel doubled into (mu, rho), biases and batch-norm deterministic.
    import resnet                                                     # noqa: E402
    w, b, bn, det_tot = resnet.n_params(n_class=vb.N_CLASS_TRUNK)
    want = 2 * w + b + bn
    print(f"VI-BNN parameters: {tot:,}   deterministic net: {det_tot:,}", flush=True)
    print(f"  mu {mu:,} + rho {rho:,} + deterministic {det:,}  "
          f"(biases and batch-norm are NOT variational)", flush=True)
    if tot != want:
        raise SystemExit(f"parameter count {tot:,} != 2*weights + biases + bn = {want:,} "
                         f"-- the variational net is not iso-topological with the trunk")
    print(f"  -> iso-topological with the 10-way ResNet: OK", flush=True)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = cd.splits()
    splits = ((Xtr, Ytr), (Xva, Yva), (Xte[:args.n_test], Yte[:args.n_test]))
    print(f"data: train {Xtr.shape}  val {Xva.shape}  test {splits[2][0].shape}",
          flush=True)
    print(f"schedule={args.schedule} epochs={args.epochs} kl={args.kl_weights} "
          f"sigma_max={args.sigma_max}\n", flush=True)

    out = os.path.join(ART, "results_vibnn" + (f"_{args.tag}" if args.tag else "") + ".json")
    rows = []
    if os.path.exists(out) and not args.force:
        rows = json.load(open(out, encoding="utf-8"))["rows"]
        print(f"resuming: {len(rows)} row(s) already present\n", flush=True)
    have = {(r["kl_weight"], r["seed"]) for r in rows}

    for klw in args.kl_weights:
        for s in seeds:
            if (klw, s) in have:
                print(f"[kl={klw} seed={s}] cached", flush=True)
                continue
            print(f"\n{'=' * 72}\n[kl={klw} seed={s}]", flush=True)
            # One failing seed must not cost the others; the seeds are independent.
            try:
                P = _net(klw, s, args, Xtr, Ytr)
                # The base table's row is the paper's kl weight; a swept weight gets its own
                # name so the two do not merge.
                nm = "VI-BNN" if klw == vb.KL_WEIGHT_PAPER else f"VI-BNN kl={klw}"
                r = evaluate(nm, P, splits, args)
                r.update(kl_weight=klw, seed=s, sigma=vb.sigma_stats(P, args.sigma_max))
                rows.append(r)
            except Exception as e:
                import traceback
                print(f"  !! seed {s} FAILED: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                traceback.print_exc()
                continue
            with open(out, "w", encoding="utf-8") as f:
                json.dump(dict(settings=vars(args), rows=rows), f, indent=1, default=float)
    print(f"\nall done -> {out}", flush=True)


if __name__ == "__main__":
    main()
