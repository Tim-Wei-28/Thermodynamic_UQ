"""Multi-seed comparison on Fashion-MNIST: per seed a deterministic LeNet-5, an
MC-dropout LeNet-5, EBM heads on the frozen trunk for each alpha, and a pixel EBM head.
Every artefact is cached by (config, seed); writes artifacts/seed_<s>_n<n_per>.json.
Options: --seeds, --alphas, --n-per, --ebm-epochs, --lenet-epochs, --family, --force.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
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
import ebm_head
import metrics_liu as mx



def _lenet(tag, X, Y, Xva, Yva, epochs, seed, p_drop=0.0):
    path = os.path.join(ART, f"lenet_{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: jnp.asarray(z[k]) for k in z.files}
    print(f"  [{tag}] training LeNet-5 ({X.shape[0]} images, p_drop={p_drop})", flush=True)
    t0 = time.time()
    p, _ = lenet.train(X, Y, epochs=epochs, seed=seed, Xva=Xva, Yva=Yva,
                       p_drop=p_drop, verbose=False)
    jax.block_until_ready(p["out_w"])
    np.savez(path, **{k: np.asarray(v) for k, v in p.items()})
    print(f"  [{tag}] done in {time.time() - t0:.0f}s", flush=True)
    return p


def _ebm(tag, X, Y, C, W0, cfg):
    path = os.path.join(ART, f"ebm_{tag}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)["params"]
    print(f"  [{tag}] training EBM head", flush=True)
    t0 = time.time()
    params, spec, _ = ebm_head.train_head(X, Y, C, W0=W0, cfg=cfg, verbose=False)
    jax.block_until_ready(params["B"])
    print(f"  [{tag}] done in {time.time() - t0:.0f}s", flush=True)
    params = {k: (np.asarray(v) if k != "field"
                  else (v[0], {kk: np.asarray(vv) for kk, vv in v[1].items()}))
              for k, v in params.items()}
    with open(path, "wb") as f:
        pickle.dump({"params": params, "n_pre": getattr(spec, "n_pre", cfg.K)}, f)
    return params


def _to_jax(p):
    return {k: (v if k != "field" else (v[0], {kk: jnp.asarray(vv)
                                               for kk, vv in v[1].items()}))
            if k == "field" else jnp.asarray(v) for k, v in p.items()}


def evaluate(name, logit_fn, Xte, Yte, Xva, Yva, n_ood, fractions):
    """One result row: in-distribution metrics, the letter ladder, and the 50 % point
    of the noise and rotate probes."""
    T = mx.fit_T(logit_fn(Xva), Yva)
    row = mx.summarise(name, logit_fn(Xte), Yte, T=T)
    row["ood"] = mx.ood_sweep(logit_fn, Xte, Yte, fractions, fd.blend_probe,
                              n=n_ood, T=T, verbose=False)
    m50 = [o for o in row["ood"] if abs(o["fraction"] - 0.5) < 1e-9][0]
    row.update(ece_ood50=m50["ece"], nll_ood50=m50["nll"], acc_ood50=m50["acc"],
               brier_ood50=m50["brier"], h_epi_ood50=m50["h_epi"])
    for probe_name in ("noise", "rotate"):
        o = mx.ood_sweep(logit_fn, Xte, Yte, (0.5,), fd.PROBES[probe_name],
                         n=n_ood, T=T, verbose=False)[0]
        row[f"ece_{probe_name}50"] = o["ece"]
        row[f"nll_{probe_name}50"] = o["nll"]
    print(f"    {name:26s} acc {row['acc']:.4f}  ECE {row['ece']:.4f}  "
          f"NLL {row['nll']:.4f} | f=.5 ECE {row['ece_ood50']:.4f} "
          f"NLL {row['nll_ood50']:.4f}", flush=True)
    return row


def run_seed(seed, args, splits):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = splits
    out = os.path.join(ART, f"seed_{seed}_n{args.n_per}.json")
    if os.path.exists(out) and not args.force:
        print(f"[seed {seed}] cached", flush=True)
        return
    print(f"\n{'=' * 72}\n[seed {seed}]", flush=True)
    t0 = time.time()
    cfg_base = dict(family=args.family, epochs=args.ebm_epochs, seed=seed)
    Xm, Ym = fd.subset(Xtr, Ytr, args.n_per, seed=seed)

    det = _lenet(f"n55000_e{args.lenet_epochs}_s{seed}", Xtr, Ytr, Xva[:2000], Yva[:2000],
                 args.lenet_epochs, seed)
    mc = _lenet(f"mcdrop{args.mc_drop}_n55000_s{seed}", Xtr, Ytr, Xva[:2000], Yva[:2000],
                args.lenet_epochs, seed, p_drop=args.mc_drop)

    F = lenet.features(det, Xm)
    W0t, b0 = lenet.head(det)
    Xa, _, W0_trained = ebm_head.prepare_features(F, [], W0=W0t, b0=b0)
    scale = 1.0 / (F.std() + 1e-12)
    prep = lambda X: np.concatenate(                                     # noqa: E731
        [lenet.features(det, X) * scale, np.ones((len(X), 1))], 1)
    rng = np.random.default_rng(seed)
    W0_random = ebm_head.HeadConfig().w0_scale * rng.standard_normal(W0_trained.shape)

    rows = []
    rows.append(evaluate("A2 LeNet det",
                         lambda X: mx.logits_det(lenet.logits(det, X)),
                         Xte, Yte, Xva, Yva, args.n_ood, fd.FRACTIONS))
    rows.append(evaluate("MC dropout LeNet",
                         lambda X: lenet.mc_dropout_logits(mc, X, p_drop=args.mc_drop,
                                                           n_samp=mx.N_SAMP),
                         Xte, Yte, Xva, Yva, args.n_ood, fd.FRACTIONS))
    for alpha in args.alphas:
        cfg = ebm_head.HeadConfig(**cfg_base)
        W0 = (1.0 - alpha) * W0_trained + alpha * W0_random
        p = _ebm(f"a4_al{alpha:.2f}_{args.family}_n{args.n_per}_s{seed}",
                 Xa, Ym, fd.C, W0, cfg)
        pj = _to_jax(p)
        r = evaluate(f"A4 alpha={alpha:.2f}",
                     lambda X, _p=pj: mx.sample_logits_ebm(_p, prep(X)),
                     Xte, Yte, Xva, Yva, args.n_ood, fd.FRACTIONS)
        r["alpha"] = alpha
        r["acc_w0"] = float(((prep(Xte) @ W0.T).argmax(1) == np.asarray(Yte)).mean())
        rows.append(r)
    if not args.skip_a1:
        cfg = ebm_head.HeadConfig(**cfg_base)
        Xa1, _, _ = ebm_head.prepare_features(Xm, [], W0=None, centre=True)
        mean1 = Xm.mean(0)
        p = _ebm(f"a1_{args.family}_n{args.n_per}_s{seed}", Xa1, Ym, fd.C, None, cfg)
        pj = _to_jax(p)
        rows.append(evaluate("A1 pixels + EBM",
                             lambda X, _p=pj: mx.sample_logits_ebm(
                                 _p, np.asarray(X, np.float64) - mean1),
                             Xte, Yte, Xva, Yva, args.n_ood, fd.FRACTIONS))

    with open(out, "w") as f:
        json.dump(dict(seed=seed, settings=vars(args), rows=rows), f, indent=1,
                  default=float)
    print(f"[seed {seed}] finished in {time.time() - t0:.0f}s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--alphas", default="1.0,0.9,0.0")
    ap.add_argument("--n-per", type=int, default=1000)
    ap.add_argument("--ebm-epochs", type=int, default=2400)
    ap.add_argument("--lenet-epochs", type=int, default=20)
    ap.add_argument("--family", default="tree")
    ap.add_argument("--mc-drop", type=float, default=0.5)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--skip-a1", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.alphas = [float(a) for a in args.alphas.split(",")]
    os.makedirs(ART, exist_ok=True)

    print("JAX devices:", jax.devices(), flush=True)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
    splits = ((Xtr, Ytr), (Xva, Yva), (Xte[:args.n_test], Yte[:args.n_test]))
    for s in [int(x) for x in args.seeds.split(",")]:
        run_seed(s, args, splits)
    print("\nall seeds done", flush=True)


if __name__ == "__main__":
    main()
