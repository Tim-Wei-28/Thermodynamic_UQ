"""Recognition families mf, tree-chain, mst-warm-mf, ebm and ebm-rj on Fashion-MNIST
at K=10, r=8 with a weak random W0: downstream metrics, exact in-run gap on held-out
images, Jq statistics and the refit ladder on the frozen thetas.  Writes
artifacts/fashion_families_n{n_per}_s{seeds}.json; per-head .pkl files are the cache.
Options: --seeds, --arms, --k, --n-per, --no-ladder, --ebm-epochs (smoke).
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
                 _os.path.join(_H, "..", "..", "fashion_lenet5", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]

import ebm_recognition as ebm                                        # noqa: E402
import gap_tools as gt                                               # noqa: E402
import jax                                                           # noqa: E402
import jax.numpy as jnp                                              # noqa: E402
from paths import ART as LIU_ART                                     # noqa: E402
import fashion_data as fd                                            # noqa: E402
import lenet                                                         # noqa: E402
import ebm_head                                                      # noqa: E402
import family_probe as fprobe                                        # noqa: E402
import metrics_liu as mx                                             # noqa: E402
import field as fieldmod                                             # noqa: E402
import recognition as recog                                          # noqa: E402
import tree_recognition as tr                                        # noqa: E402
import trainer as trmod                                              # noqa: E402

ART = os.path.join(os.path.dirname(_H), "artifacts")

# operating point; the ebm family accepts only the sfe-loo estimator
BASE = dict(rank=8, epochs=2400, lr=0.1, T=8, beta_max=1.0, free_bits=0.02, wd=0.01,
            j_init=-1.5, field_n_hidden=24, gamma0=1.0, estimator="sfe-loo")

ARMS = [
    ("mf",          dict(family="mf")),
    ("tree-chain",  dict(family="tree", topology="chain")),
    ("mst-warm-mf", dict(family="tree_max_span", topology="chain", refit="once",
                         struct_warmup_family="mf")),
    ("ebm",         dict(family="ebm")),          # zero coupling init
    ("ebm-rj",      dict(family="ebm")),          # random coupling init (control)
]
LADDER_FAMS = ("mf", "tree", "ebm")
LADDER_SRCS = ("mf", "tree-chain", "ebm")         # whose theta the refits target


def _trunk(seed, epochs, Xtr, Ytr, Xva, Yva):
    """Frozen LeNet-5 for this seed, cached in the fashion_lenet5 artifacts dir."""
    tag = f"n55000_e{epochs}_s{seed}"
    path = os.path.join(LIU_ART, f"lenet_{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: jnp.asarray(z[k]) for k in z.files}
    print(f"  [{tag}] training trunk", flush=True)
    p, _ = lenet.train(Xtr, Ytr, epochs=epochs, seed=seed, Xva=Xva, Yva=Yva, verbose=False)
    jax.block_until_ready(p["out_w"])
    np.savez(path, **{k: np.asarray(v) for k, v in p.items()})
    return p


def _train(tag, label, X, Y, C, W0, cfg):
    """-> (params, spec, parents, secs, cached).  pkl cache in this study's artifacts."""
    path = os.path.join(ART, f"fam_{tag}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["params"], d["spec"], d.get("parents"), d["secs"], True
    log = [] if cfg.refit else None
    orig_setup = trmod._setup
    # zero coupling init (U rows and c entries) for every ebm label except the ebm-rj control
    if label.startswith("ebm") and label != "ebm-rj":
        def zeroJ_setup(key, tc_, spec):
            key, Xd, Yd, qi, lik, (B, A, U, c, fp, J, ch) = orig_setup(key, tc_, spec)
            U = U.at[tc_.K:, :].set(0.0)
            c = c.at[tc_.K:].set(0.0)
            return key, Xd, Yd, qi, lik, (B, A, U, c, fp, J, ch)
        trmod._setup = zeroJ_setup
    t0 = time.time()
    try:
        params, spec, _fwd = ebm_head.train_head(X, Y, C, W0=W0, cfg=cfg, verbose=False,
                                                 refit_log=log)
    finally:
        trmod._setup = orig_setup
    jax.block_until_ready(params["B"])
    secs = time.time() - t0
    parents = [int(p) for p in log[-1]["parents"]] if log else None
    params = {k: (np.asarray(v) if k != "field"
                  else (v[0], {kk: np.asarray(vv) for kk, vv in v[1].items()}))
              for k, v in params.items()}
    with open(path, "wb") as f:
        pickle.dump({"params": params, "spec": spec, "parents": parents, "secs": secs}, f)
    return params, spec, parents, secs, False


def _to_jax(p):
    return {k: (v[0], {kk: jnp.asarray(vv) for kk, vv in v[1].items()}) if k == "field"
            else jnp.asarray(v) for k, v in p.items()}


def _probe_subset(Xte_p, Yte, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xte_p), size=min(n, len(Xte_p)), replace=False)
    return (jnp.asarray(np.asarray(Xte_p)[idx]),
            jnp.asarray(np.asarray(Yte)[idx], dtype=jnp.int32))


def _tables(pj, spec, family, Xp, Yp, C, K):
    """(logq, logp) enumerated tables for a trained head on probe inputs."""
    feat = jnp.concatenate([Xp, jax.nn.one_hot(Yp, C)], 1)
    pre = feat @ pj["U"].T + pj["c"]
    logq = recog.REGISTRY[family]["enumerate_logq"](pre, spec)
    logq = logq - jax.scipy.special.logsumexp(logq, 1, keepdims=True)   # defensive
    h = fieldmod.apply(pj["field"][0], pj["field"][1], Xp)
    logp = gt.true_posterior_logp(pj["W0"], pj["B"], pj["A"], Xp, Yp, h, pj["J"],
                                  ebm.make_ebm_spec(K))
    return pre, logq, logp


def exact_gap(pj, spec, family, Xp, Yp, C, K):
    """Exact in-run gap and correlation structure on the probe set."""
    _, logq, logp = _tables(pj, spec, family, Xp, Yp, C, K)
    cfg = tr.all_configs(K)
    kl = (jnp.exp(logq) * (logq - logp)).sum(1)
    cpost_m, cpost_x = fprobe._corr_stats(jnp.exp(logp), cfg, K)
    cq_m, _ = fprobe._corr_stats(jnp.exp(logq), cfg, K)
    m = gt.dist_metrics(logq, logp, K)
    return dict(kl_qp=float(kl.mean()),
                kl_qp_se=float(kl.std(ddof=1) / np.sqrt(kl.shape[0])),
                corr_post=float(cpost_m.mean()), corr_post_max=float(cpost_x.mean()),
                corr_q=float(cq_m.mean()), corr_lost=float((cpost_m - cq_m).mean()),
                mu_err=m["mu_err"], cov_err=m["cov_err"], n_probe=int(Xp.shape[0]))


def jq_stats(pj, spec, Xp, Yp, C, K):
    """Summary scalars and class means of the ebm coupling matrices Jq(x,y)."""
    feat = jnp.concatenate([Xp, jax.nn.one_hot(Yp, C)], 1)
    pre = feat @ pj["U"].T + pj["c"]
    Jq = ebm.couplings(pre, spec)                                    # (N,K,K) symmetric
    iu, ju = jnp.asarray(spec.iu), jnp.asarray(spec.ju)
    pairs = Jq[:, iu, ju]                                            # (N,P)
    Jp = pj["J"][iu, ju]                                             # prior couplings (P,)
    mean_pairs = pairs.mean(0)
    align = float(np.corrcoef(np.asarray(mean_pairs), np.asarray(Jp))[0, 1])
    cls_mean = jnp.stack([Jq[Yp == c].mean(0) for c in range(C)])    # (C,K,K)
    return dict(jq_abs=float(jnp.abs(pairs).mean()),
                jq_frac_neg=float((pairs < 0).mean()),
                jq_prior_align=align), np.asarray(cls_mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--arms", default=None, help="comma list of arm labels")
    ap.add_argument("--n-per", type=int, default=1000, help="training images PER CLASS")
    ap.add_argument("--lenet-epochs", type=int, default=20)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--n-probe", type=int, default=2000)
    ap.add_argument("--n-ladder", type=int, default=500)
    ap.add_argument("--ladder-steps", type=int, default=2500)
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--ebm-epochs", type=int, default=None, help="override for smoke tests")
    args = ap.parse_args()
    if args.ebm_epochs:
        BASE["epochs"] = args.ebm_epochs
    K = args.k
    if K > ebm.MAX_K_ENUM or K > fprobe.K_MAX:
        raise SystemExit(f"K={K} exceeds the enumeration ceiling")
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = ARMS if not args.arms else [a for a in ARMS if a[0] in set(args.arms.split(","))]
    os.makedirs(ART, exist_ok=True)
    # seeds in the filename: concurrent per-seed jobs must not share a JSON
    out_path = os.path.join(
        ART, f"fashion_families_n{args.n_per}_s{args.seeds.replace(',', '-')}.json")

    print("JAX devices:", jax.devices(), flush=True)
    print(f"{len(arms)} arms x {len(seeds)} seeds = {len(arms) * len(seeds)} heads "
          f"at K={K}", flush=True)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
    Xte, Yte = Xte[:args.n_test], Yte[:args.n_test]
    rows, ladder_rows = [], []
    t_start = time.time()

    def _dump():
        with open(out_path, "w") as f:
            json.dump(dict(base=BASE, K=K, arms=[(a, o) for a, o in ARMS],
                           settings={k: v for k, v in vars(args).items()},
                           rows=rows, ladder=ladder_rows), f, indent=1, default=float)

    for seed in seeds:
        # trunk, subset and W0 derive from the seed before the arms branch (paired arms)
        trunk = _trunk(seed, args.lenet_epochs, Xtr, Ytr, Xva[:2000], Yva[:2000])
        Xm, Ym = fd.subset(Xtr, Ytr, args.n_per, seed=seed)
        F = lenet.features(trunk, Xm)
        W0t, b0 = lenet.head(trunk)
        Xa, _, W0_trained = ebm_head.prepare_features(F, [], W0=W0t, b0=b0)
        scale = 1.0 / (F.std() + 1e-12)
        prep = lambda X: np.concatenate(                                  # noqa: E731
            [lenet.features(trunk, X) * scale, np.ones((len(X), 1))], 1)
        rng = np.random.default_rng(seed)
        W0 = ebm_head.HeadConfig().w0_scale * rng.standard_normal(W0_trained.shape)
        Xva_p, Xte_p = prep(Xva), prep(Xte)
        Xp, Yp = _probe_subset(Xte_p, Yte, args.n_probe, seed)

        theta = {}                                       # ladder sources for this seed
        for label, override in arms:
            cfg = ebm_head.HeadConfig(**{**BASE, **override}, K=K, seed=seed)
            # epochs in the tag: smoke and full runs must not share a cache entry
            tag = f"{label}_K{K}_n{args.n_per}_e{cfg.epochs}_s{seed}"
            params, spec, parents, secs, cached = _train(tag, label, Xa, Ym, fd.C, W0, cfg)
            pj = _to_jax(params)
            if label in LADDER_SRCS:
                theta[label] = (pj, spec, override["family"])

            lf = lambda X, _p=pj: mx.sample_logits_ebm(_p, prep(X))       # noqa: E731
            T = mx.fit_T(mx.sample_logits_ebm(pj, Xva_p), Yva)
            r = mx.summarise(label, mx.sample_logits_ebm(pj, Xte_p), Yte, T=T)
            o = mx.ood_sweep(lf, Xte, Yte, (0.5,), fd.PROBES["letters"],
                             n=args.n_ood, T=T, verbose=False)[0]
            for m_ in ("acc", "ece", "nll", "h_epi"):
                r[f"{m_}_ood50"] = o[m_]

            r.update(exact_gap(pj, spec, override["family"], Xp, Yp, fd.C, K))
            if override["family"] == "ebm":
                jstat, cls_mean = jq_stats(pj, spec, Xp, Yp, fd.C, K)
                r.update(jstat)
                np.save(os.path.join(ART, f"jq_classmean_{tag}.npy"), cls_mean)
            if parents is not None:
                r["parents"] = parents
                r["chain_agree"] = fprobe.chain_agreement(parents, K)
            spec_stub = type("S", (), {"n_pre": getattr(spec, "n_pre", K)})
            r.update(arm=label, K=K, seed=seed, n_pre=getattr(spec, "n_pre", K),
                     train_s=secs, family=override["family"],
                     n_trained=ebm_head.n_trained(params, spec_stub, Xa.shape[1],
                                                  fd.C, cfg)["total"])
            rows.append(r)
            print(f"  s{seed} {label:12s} acc {r['acc']:.4f} ECE {r['ece']:.4f} "
                  f"NLL {r['nll']:.4f} | KL(q||p) {r['kl_qp']:.4f} "
                  f"corr_post {r['corr_post']:.4f} lost {r['corr_lost']:.4f}"
                  f"{'  (cached)' if cached else f'  {secs:.0f}s'}", flush=True)
            _dump()

        # ladder: refit every family to every frozen theta
        if not args.no_ladder:
            Xl, Yl = _probe_subset(Xte_p, Yte, args.n_ladder, seed + 1000)
            featl = jnp.concatenate([Xl, jax.nn.one_hot(Yl, fd.C)], 1)
            for i_s, (src, (pj, spec, family)) in enumerate(theta.items()):
                h = fieldmod.apply(pj["field"][0], pj["field"][1], Xl)
                logp = gt.true_posterior_logp(pj["W0"], pj["B"], pj["A"], Xl, Yl,
                                              h, pj["J"], ebm.make_ebm_spec(K))
                for i_f, fam in enumerate(LADDER_FAMS):
                    for i_m, mode in enumerate(("amortized", "free")):
                        key = jax.random.fold_in(jax.random.key(11),
                                                 seed * 1000 + i_s * 100 + i_f * 10 + i_m)
                        met, _ = gt.fit_family(key, fam, K, featl, logp,
                                               amortized=(mode == "amortized"),
                                               steps=args.ladder_steps)
                        ladder_rows.append(dict(seed=seed, src=src, fam=fam, mode=mode,
                                                **met))
                        print(f"  s{seed} ladder theta_{src:10s} {fam:4s} {mode:9s} "
                              f"KL {met['kl_mean']:.4f}  cov_err {met['cov_err']:.4f}",
                              flush=True)
                        _dump()

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
