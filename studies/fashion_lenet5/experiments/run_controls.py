"""Control experiments on heads trained by run_seeds.py: gate ablation
(prior / dropout / allon), the three corruption probes, and an MC-dropout LeNet.
Writes artifacts/results_controls.json.
Options: --seeds, --alphas, --n-per, --n-test, --n-ood, --mc-drop, --skip-a1, --skip-mc.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time

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



def _load_ebm(tag):
    """Load a head trained by run_seeds.py (tags carry the seed)."""
    path = os.path.join(ART, f"ebm_{tag}.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\n  -- run_controls.py trains nothing; this head "
                         f"comes from run_seeds.py.  Run that first (or copy its "
                         f"artifacts/ebm_*.pkl across) for every seed you ask for here.")
    with open(path, "rb") as f:
        return pickle.load(f)["params"]


def _trunk(n, seed=0):
    p = os.path.join(ART, f"lenet_n{n}_e20_s{seed}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run run_compare.py first")
    z = np.load(p)
    return {k: jnp.asarray(z[k]) for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=1000)
    ap.add_argument("--seeds", default="0")
    # heads for every alpha listed here must already exist; this driver trains no head
    ap.add_argument("--alphas", default="1.0,0.9")
    ap.add_argument("--skip-a1", action="store_true",
                    help="omit the pixel-path controls")
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=10000, help="test points per evaluation")
    ap.add_argument("--mc-drop", type=float, default=0.5)
    ap.add_argument("--lenet-epochs", type=int, default=20)
    ap.add_argument("--skip-mc", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",")]
    alphas = [float(a) for a in str(args.alphas).split(",")]
    os.makedirs(ART, exist_ok=True)

    (Xtr, Ytr), (Xva, Yva), (Xte0, Yte0) = fd.splits()
    Xte, Yte = Xte0[:args.n_test], Yte0[:args.n_test]
    print(f"controls: seeds {seeds}  test {len(Yte)}  ood {args.n_ood}/fraction")

    rows = []
    first = {}

    # gate ablation
    print("\n" + "=" * 78)
    print("RISK 2 -- does the LEARNED, INPUT-CONDITIONAL gating matter?")
    print("=" * 78)
    for seed in seeds:
        Xm, Ym = fd.subset(Xtr, Ytr, args.n_per, seed=seed)
        # rebuild the preprocessing maps exactly as the training run for this seed did
        trunk55 = _trunk(55000, seed)
        F = lenet.features(trunk55, Xm)
        scale = 1.0 / (F.std() + 1e-12)
        prep_a4 = (lambda t, s: (lambda X: np.concatenate(                # noqa: E731
            [lenet.features(t, X) * s, np.ones((len(X), 1))], 1)))(trunk55, scale)
        mean1 = Xm.mean(0)
        prep_a1 = (lambda m: (lambda X: np.asarray(X, np.float64) - m))(mean1)  # noqa: E731

        models = [(f"A4 a={a:.2f}",
                   _load_ebm(f"a4_al{a:.2f}_tree_n{args.n_per}_s{seed}"), prep_a4)
                  for a in alphas]
        if not args.skip_a1:
            models.append(("A1 pixels",
                           _load_ebm(f"a1_tree_n{args.n_per}_s{seed}"), prep_a1))
        if seed == seeds[0]:
            # the corruption block below uses the largest alpha of the first seed
            top = max(range(len(alphas)), key=lambda i: alphas[i])
            first = dict(trunk=trunk55, prep_a4=prep_a4, a4=models[top][1], seed=seed,
                         label=models[top][0])

        for name, params, prep in models:
            kappa = mx.prior_gate_rate(params, prep(Xte[:1000]))
            print(f"\n[seed {seed}] {name}: prior gate rate = {kappa:.4f}"
                  f"  (dropout control matched to it)")
            for gate in ("prior", "dropout", "allon"):
                t0 = time.time()
                lf = (lambda g: (lambda X: mx.sample_logits_ebm(
                    params, prep(X), gate=g, kappa=kappa)))(gate)
                T = mx.fit_T(lf(Xva), Yva)
                r = mx.summarise(f"{name} / {gate}", lf(Xte), Yte, T=T)
                ood = mx.ood_sweep(lf, Xte, Yte, (0.5,), fd.blend_probe,
                                   n=args.n_ood, T=T, verbose=False)[0]
                r.update(model=name, gate=gate, probe="letters", seed=seed,
                         acc_ood50=ood["acc"], ece_ood50=ood["ece"], nll_ood50=ood["nll"],
                         brier_ood50=ood["brier"],
                         h_epi_ood50=ood["h_epi"], gate_rate=kappa)
                rows.append(r)
                print(f"   {gate:8s} acc {r['acc']:.4f} ECE {r['ece']:.4f} "
                      f"NLL {r['nll']:.4f} | f=0.5: acc {ood['acc']:.4f} "
                      f"ECE {ood['ece']:.4f} NLL {ood['nll']:.4f} "
                      f"H_epi {ood['h_epi']:.4f}  ({time.time()-t0:.0f}s)")
        with open(os.path.join(ART, "results_controls.json"), "w") as f:
            json.dump(dict(settings=vars(args), rows=rows), f, indent=1, default=float)

    # full corruption ladders for two methods, first seed only
    print("\n" + "=" * 78)
    print("RISK 4 -- does the advantage survive a DIFFERENT corruption?  (first seed only)")
    print("=" * 78)
    prep_a4, a4p = first["prep_a4"], first["a4"]
    det55 = first["trunk"]
    det_lf = lambda X: mx.logits_det(lenet.logits(det55, X))              # noqa: E731
    T_det = mx.fit_T(det_lf(Xva[:2000]), Yva[:2000])
    a4_lf = lambda X: mx.sample_logits_ebm(a4p, prep_a4(X))               # noqa: E731
    T_a4 = mx.fit_T(a4_lf(Xva), Yva)
    for probe_name, probe in fd.PROBES.items():
        print(f"\n--- probe: {probe_name}")
        for nm, lf, T in (("LeNet-5 det", det_lf, T_det), (first["label"], a4_lf, T_a4)):
            sw = mx.ood_sweep(lf, Xte, Yte, fd.FRACTIONS, probe, n=args.n_ood,
                              T=T, verbose=False)
            for o in sw:
                rows.append(dict(o, model=nm, gate="prior", probe=probe_name,
                                 name=f"{nm} / {probe_name} / f={o['fraction']:.1f}"))
            m50 = [o for o in sw if abs(o["fraction"] - 0.5) < 1e-9][0]
            print(f"   {nm:12s} f=0.5: acc {m50['acc']:.4f} ECE {m50['ece']:.4f} "
                  f"NLL {m50['nll']:.4f} H_epi {m50['h_epi']:.4f}   "
                  f"| f=0.9 ECE {sw[-1]['ece']:.4f}")

    # MC-dropout LeNet baseline
    if not args.skip_mc:
        print("\n" + "=" * 78)
        print(f"RISK 1 -- MC dropout on LeNet-5 (p={args.mc_drop}), the cheap incumbent")
        print("=" * 78)
        s0 = first["seed"]
        path = os.path.join(ART, f"lenet_mcdrop{args.mc_drop}_n55000_s{s0}.npz")
        if os.path.exists(path):
            z = np.load(path)
            pmc = {k: jnp.asarray(z[k]) for k in z.files}
            print("  cached")
        else:
            print(f"  training LeNet-5 with dropout on 55 000 images")
            pmc, _ = lenet.train(Xtr, Ytr, epochs=args.lenet_epochs, seed=s0,
                                 Xva=Xva[:2000], Yva=Yva[:2000], p_drop=args.mc_drop)
            np.savez(path, **{k: np.asarray(v) for k, v in pmc.items()})
        mc_lf = lambda X: lenet.mc_dropout_logits(                        # noqa: E731
            pmc, X, p_drop=args.mc_drop, n_samp=mx.N_SAMP)
        T_mc = mx.fit_T(mc_lf(Xva), Yva)
        r = mx.summarise("MC dropout LeNet", mc_lf(Xte), Yte, T=T_mc)
        for probe_name, probe in fd.PROBES.items():
            sw = mx.ood_sweep(mc_lf, Xte, Yte, fd.FRACTIONS, probe, n=args.n_ood,
                              T=T_mc, verbose=False)
            for o in sw:
                rows.append(dict(o, model="MC dropout", gate="mc-dropout",
                                 probe=probe_name,
                                 name=f"MC dropout / {probe_name} / f={o['fraction']:.1f}"))
            m50 = [o for o in sw if abs(o["fraction"] - 0.5) < 1e-9][0]
            print(f"   {probe_name:8s} in-dist acc {r['acc']:.4f} ECE {r['ece']:.4f} "
                  f"NLL {r['nll']:.4f} | f=0.5: acc {m50['acc']:.4f} "
                  f"ECE {m50['ece']:.4f} NLL {m50['nll']:.4f} H_epi {m50['h_epi']:.4f}")
        rows.append(dict(r, model="MC dropout", gate="mc-dropout", probe="in-dist"))

    with open(os.path.join(ART, "results_controls.json"), "w") as f:
        json.dump(dict(settings=vars(args), rows=rows), f, indent=1, default=float)
    print(f"\nwritten: {os.path.join(ART, 'results_controls.json')}")


if __name__ == "__main__":
    main()
