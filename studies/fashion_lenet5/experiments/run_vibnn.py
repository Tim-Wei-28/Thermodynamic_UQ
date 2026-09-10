"""Mean-field VI LeNet-5 (vi_bnn.py) trained at several KL weights and scored through
the shared metric path. The KL weight is swept because Liu et al. (2022) do not state
it for Fashion-MNIST. Writes artifacts/results_vibnn.json.
Options: --kl-weights, --seeds, --epochs, --lr, --n-ood, --n-test.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

# path bootstrap: lib/ and pipeline/model/ must be importable before `paths`
import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import HERE, ART                                          # noqa: E402

import fashion_data as fd
import lenet
import vi_bnn
import metrics_liu as mx



def _cached(tag, X, Y, epochs, lr, seed, kl_weight, Xva, Yva):
    path = os.path.join(ART, f"vibnn_{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: jnp.asarray(z[k]) for k in z.files}, True
    print(f"  [{tag}] training VI-BNN (kl_weight={kl_weight})", flush=True)
    t0 = time.time()
    p, _ = vi_bnn.train(X, Y, epochs=epochs, lr=lr, seed=seed, kl_weight=kl_weight,
                        Xva=Xva, Yva=Yva, verbose=False)
    jax.block_until_ready(p["out_mu"])
    print(f"  [{tag}] done in {time.time() - t0:.0f}s", flush=True)
    np.savez(path, **{k: np.asarray(v) for k, v in p.items()})
    return p, False


def evaluate(name, logit_fn, Xte, Yte, Xva, Yva, n_ood):
    """As run_seeds.evaluate: in-distribution metrics, the letter ladder, and the 50 %
    point of the noise and rotate probes, with T fitted in-distribution."""
    T = mx.fit_T(logit_fn(Xva), Yva)
    row = mx.summarise(name, logit_fn(Xte), Yte, T=T)
    row["ood"] = mx.ood_sweep(logit_fn, Xte, Yte, fd.FRACTIONS, fd.PROBES["letters"],
                              n=n_ood, T=T, verbose=False)
    _f50 = [o for o in row["ood"] if abs(o["fraction"] - 0.5) < 1e-9][0]
    for probe in ("letters", "noise", "rotate"):
        o = _f50 if probe == "letters" else mx.ood_sweep(
            logit_fn, Xte, Yte, (0.5,), fd.PROBES[probe], n=n_ood, T=T, verbose=False)[0]
        row[f"acc_{probe}"] = o["acc"]
        row[f"ece_{probe}"] = o["ece"]
        row[f"nll_{probe}"] = o["nll"]
        row[f"h_epi_{probe}"] = o["h_epi"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl-weights", default="0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()
    kls = [float(k) for k in args.kl_weights.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(ART, exist_ok=True)
    print("JAX devices:", jax.devices(), flush=True)
    mu, rho, b, tot = vi_bnn.n_params()
    print(f"VI-BNN parameters: {tot:,}  (paper: 123 176)  "
          f"{'MATCH' if tot == 123176 else 'MISMATCH'}", flush=True)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
    Xte, Yte = Xte[:args.n_test], Yte[:args.n_test]
    rows = []
    for seed in seeds:
        for kl in kls:
            tag = f"kl{kl}_e{args.epochs}_s{seed}"
            p, cached = _cached(tag, Xtr, Ytr, args.epochs, args.lr, seed, kl,
                                Xva[:2000], Yva[:2000])
            lf = lambda X, _p=p: vi_bnn.sample_logits(_p, X, n_samp=mx.N_SAMP)
            r = evaluate(f"VI-BNN kl={kl}", lf, Xte, Yte, Xva, Yva, args.n_ood)
            r.update(kl_weight=kl, seed=seed, n_params=tot,
                     sigma=vi_bnn.sigma_stats(p))
            rows.append(r)
            print(f"  kl={kl:<5} acc {r['acc']:.4f}  ECE {r['ece']:.4f}  "
                  f"NLL {r['nll']:.4f}  T* {r['T']:.2f} | f=.5 ECE {r['ece_letters']:.4f}"
                  f"  NLL {r['nll_letters']:.4f} | sigma {r['sigma']['mean']:.4f}"
                  f"{'  (cached)' if cached else ''}", flush=True)
            with open(os.path.join(ART, "results_vibnn.json"), "w") as f:
                json.dump(dict(settings=vars(args), rows=rows), f, indent=1, default=float)

    print("\n" + "=" * 96)
    print("KL-WEIGHT TRADE-OFF (paper Figure 6D) -- does the curve reach their point?")
    print("Liu et al. Fashion-MNIST BNN:  acc 0.9015   ECE 0.0156   ECE@50% 0.0918")
    print("=" * 96)
    hdr = (f"{'kl':>6} {'acc':>8} {'ECE':>8} {'NLL':>8} {'T*':>6} "
           f"{'ECE@.5':>8} {'NLL@.5':>8} {'ECEnoise':>9} {'ECErot':>8} {'sigma':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["kl_weight"]):
        print(f"{r['kl_weight']:>6} {r['acc']:>8.4f} {r['ece']:>8.4f} {r['nll']:>8.4f} "
              f"{r['T']:>6.2f} {r['ece_letters']:>8.4f} {r['nll_letters']:>8.4f} "
              f"{r['ece_noise']:>9.4f} {r['ece_rotate']:>8.4f} "
              f"{r['sigma']['mean']:>8.4f}")
    best = min(rows, key=lambda r: r["ece"])
    print(f"\nECE minimum at kl_weight={best['kl_weight']}: "
          f"acc {best['acc']:.4f}, ECE {best['ece']:.4f}  "
          f"(Liu: acc 0.9015, ECE 0.0156)")
    print(f"written: {os.path.join(ART, 'results_vibnn.json')}")


if __name__ == "__main__":
    main()
