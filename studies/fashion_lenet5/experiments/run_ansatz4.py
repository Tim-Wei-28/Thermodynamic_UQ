"""Approach 4: EBM head on frozen LeNet features with a degraded base map
W0(alpha) = (1 - alpha) * W0_trained + alpha * W0_random.
Writes artifacts/results_ansatz4.json.
Options: --alphas, --trunk, --n-per, --ebm-epochs, --family, --seed, --scan.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp

# path bootstrap: lib/ and pipeline/model/ must be importable before `paths`
import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import HERE, ART                                          # noqa: E402

import fashion_data as fd
import lenet
import ebm_head
import metrics_liu as mx
from mechanism import cached_train, report



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--trunk", type=int, default=55000,
                    help="which cached LeNet supplies the frozen features")
    ap.add_argument("--n-per", type=int, default=1000)
    ap.add_argument("--ebm-epochs", type=int, default=2400)
    ap.add_argument("--family", default="tree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--scan", action="store_true",
                    help="only tabulate the base map's own accuracy across alpha")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    cfg = ebm_head.HeadConfig(family=args.family, epochs=args.ebm_epochs, seed=args.seed)
    os.makedirs(ART, exist_ok=True)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
    Xm, Ym = fd.subset(Xtr, Ytr, args.n_per, seed=args.seed)
    tp = os.path.join(ART, f"lenet_n{args.trunk}_e20_s{args.seed}.npz")
    if not os.path.exists(tp):
        raise SystemExit(f"missing {tp} -- run run_compare.py first")
    z = np.load(tp)
    trunk = {k: jnp.asarray(z[k]) for k in z.files}
    print(f"trunk {args.trunk}   head trains on {Xm.shape[0]}   alphas {alphas}")

    F = lenet.features(trunk, Xm)
    W0t, b0 = lenet.head(trunk)
    Xa, _, W0_trained = ebm_head.prepare_features(F, [], W0=W0t, b0=b0)
    scale = 1.0 / (F.std() + 1e-12)
    prep = lambda X: np.concatenate(                                        # noqa: E731
        [lenet.features(trunk, X) * scale, np.ones((len(X), 1))], 1)
    Fte, Fva = prep(Xte), prep(Xva)
    # one random draw shared by all alphas, so the family is a line between fixed matrices
    rng = np.random.default_rng(args.seed)
    W0_random = cfg.w0_scale * rng.standard_normal(W0_trained.shape)

    # base-map accuracy is not linear in alpha; scan first to choose the alpha grid
    if args.scan:
        print(f"\n  {'alpha':>6}  {'acc of W0 alone':>15}  {'|W0|':>8}")
        for a in np.linspace(0, 1, 21):
            W = (1.0 - a) * W0_trained + a * W0_random
            acc = float(((Fte @ W.T).argmax(1) == np.asarray(Yte)).mean())
            print(f"  {a:6.2f}  {acc:15.4f}  {np.linalg.norm(W):8.3f}")
        return

    rows = []
    for alpha in alphas:
        W0 = (1.0 - alpha) * W0_trained + alpha * W0_random
        tag = f"a4_t{args.trunk}_al{alpha:.2f}_{args.family}"
        print(f"\n{'=' * 74}\nalpha = {alpha:.2f}")
        acc_w0 = float(((Fte @ W0.T).argmax(1) == np.asarray(Yte)).mean())
        print(f"  acc of W0 alone (frozen base): {acc_w0:.4f}")
        params, _ = cached_train(tag, Xa, Ym, fd.C, W0, cfg)
        logit_fn = lambda X: mx.sample_logits_ebm(params, prep(X))          # noqa: E731
        T = mx.fit_T(logit_fn(Xva), Yva)
        r = mx.summarise(f"alpha={alpha:.2f}", logit_fn(Xte), Yte, T=T)
        mech = report(f"alpha={alpha:.2f}", params, Fte, cfg)
        ood = mx.ood_sweep(logit_fn, Xte, Yte, fd.FRACTIONS, fd.blend_probe,
                           n=args.n_ood, T=T, verbose=False)
        m50 = [o for o in ood if abs(o["fraction"] - 0.5) < 1e-9]
        r.update(alpha=alpha, acc_w0=acc_w0, ood=ood,
                 ece_ood50=m50[0]["ece"] if m50 else None,
                 h_epi_ood50=m50[0]["h_epi"] if m50 else None,
                 adapter_ratio=mech["adapters_all_on"] / (mech["base"] + 1e-12),
                 gate_rate=mech["gate_rate"], draw_spread=mech["draw_spread"],
                 verdict=mech["verdict"])
        rows.append(r)
        print(f"  -> acc {r['acc']:.4f}  ECE {r['ece']:.4f}  "
              f"ECE@50 {r['ece_ood50']:.4f}  H_epi {r['h_epi']:.4f}")

    rows.sort(key=lambda r: r["alpha"])
    print("\n" + "=" * 78)
    print("APPROACH 4: degrading the frozen base map")
    print("acc_w0 = what the frozen W0 scores ALONE; adapter_ratio = how much the")
    print("adapters move the logits relative to it; h_epi = the mixture's disagreement.")
    print("=" * 78)
    print(mx.table(rows, cols=("name", "acc_w0", "acc", "ece", "ece_ood50",
                               "h_epi", "h_epi_ood50", "adapter_ratio", "gate_rate")))
    with open(os.path.join(ART, "results_ansatz4.json"), "w") as f:
        json.dump(dict(settings=vars(args), rows=rows), f, indent=1, default=float)
    print(f"\nwritten: {os.path.join(ART, 'results_ansatz4.json')}")


if __name__ == "__main__":
    main()
