"""Cached head training (artifacts/ebm_<tag>.pkl) and a mechanism readout that
separates two collapse modes: (a) adapters driven to zero, (b) gate distribution
frozen onto one pattern. Both give logits = W0 x and zero epistemic uncertainty.
"""
from __future__ import annotations
import os
import pickle
import time

import numpy as np

import sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from paths import ART                                             # noqa: E402
from paths import MODEL_DIR as _MODEL                             # noqa: E402
if _MODEL not in sys.path:
    sys.path.insert(0, _MODEL)

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import model as pmodel                                            # noqa: E402
import field as fieldmod                                          # noqa: E402
import ebm_head                                                   # noqa: E402

def cached_train(tag, X, Y, C, W0, cfg):
    """Train the head once and cache it under artifacts/ by tag."""
    path = os.path.join(ART, f"ebm_{tag}.pkl")
    if os.path.exists(path):
        print(f"[{tag}] cached")
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["params"], d["n_pre"]
    print(f"[{tag}] training ...")
    t0 = time.time()
    params, spec, _ = ebm_head.train_head(X, Y, C, W0=W0, cfg=cfg)
    jax.block_until_ready(params["B"])      # JAX dispatches asynchronously; time the work
    print(f"[{tag}] trained in {time.time() - t0:.0f}s (blocking)")
    params = {k: (np.asarray(v) if not isinstance(v, tuple) else
                  (v[0], {kk: np.asarray(vv) for kk, vv in v[1].items()}))
              for k, v in params.items()}
    with open(path, "wb") as f:
        pickle.dump({"params": params, "n_pre": getattr(spec, "n_pre", cfg.K)}, f)
    return params, getattr(spec, "n_pre", cfg.K)


def _to_jax(params):
    return {k: (v if isinstance(v, tuple) else jnp.asarray(v)) if k != "field"
            else (v[0], {kk: jnp.asarray(vv) for kk, vv in v[1].items()})
            for k, v in params.items()}


def report(tag, params, X, cfg, n=1000, n_samp=100, seed=3):
    """Adapter magnitudes, gate statistics and logit spread across draws; prints a verdict."""
    p = _to_jax(params)
    W0, B, A = p["W0"], p["B"], p["A"]
    kind, fp = p["field"]
    J = p["J"]
    K = J.shape[0]
    Xs = jnp.asarray(np.asarray(X[:n], np.float64))

    # gate prior: n_samp patterns per input
    def draw(key, x):
        h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
        z0 = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float64)
        return pmodel.gibbs(key, h, J, z0, 30, K)
    keys = jax.random.split(jax.random.key(seed), Xs.shape[0])
    Z = np.asarray(jax.jit(jax.vmap(draw))(keys, Xs))           # (n, n_samp, K)

    base = np.asarray(Xs @ W0.T)                                # (n, C)
    a = jnp.einsum("krd,nd->nkr", A, Xs)
    all_on = np.asarray(jnp.einsum("kcr,nkr->nc", B, a))        # adapter ceiling, all gates on
    zmean = jnp.asarray(Z.mean(1))                              # (n, K)
    realised = np.asarray(jnp.einsum("nk,kcr,nkr->nc", zmean, B, a))

    # logit spread across draws, the source of H_epi
    def spread(key, x):
        h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
        z0 = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float64)
        z = pmodel.gibbs(key, h, J, z0, 30, K)
        lg = pmodel.batch_logits(W0, B, A, jnp.broadcast_to(x, (n_samp, W0.shape[1])), z)
        return lg.std(0).mean()
    lg_std = float(np.mean(np.asarray(jax.jit(jax.vmap(spread))(keys, Xs))))

    rate = Z.mean((0, 1))
    per_input = Z.mean(1)                                       # (n, K)
    print(f"\n--- {tag} " + "-" * (66 - len(tag)))
    print(f"  ||W0 x||                        {np.abs(base).mean():.4f}")
    print(f"  ||adapters, ALL gates on||      {np.abs(all_on).mean():.4f}"
          f"   ({100 * np.abs(all_on).mean() / (np.abs(base).mean() + 1e-12):.2f} % of base)")
    print(f"  ||adapters, as realised||       {np.abs(realised).mean():.4f}")
    print(f"  |B| / |B_init|                  {float(jnp.linalg.norm(B)):.4f} / "
          f"{cfg.adapter_init * np.sqrt(B.size):.4f}")
    print(f"  |A| / |A_init|                  {float(jnp.linalg.norm(A)):.4f} / "
          f"{cfg.adapter_init * np.sqrt(A.size):.4f}")
    print(f"  mean gate rate                  {Z.mean():.4f}")
    print(f"  per-gate rates                  {np.round(rate, 3)}")
    print(f"  per-input rate spread (std)     {per_input.std(0).mean():.4f}"
          f"   (0 = the gate pattern ignores the input)")
    print(f"  within-input draw spread (std)  {Z.std(1).mean():.4f}"
          f"   (0 = the prior is a point mass)")
    print(f"  logit std across draws          {lg_std:.6f}   <- this is what makes H_epi")
    print(f"  |J| max / mean                  {float(jnp.abs(J).max()):.3f} / "
          f"{float(jnp.abs(J).mean()):.3f}")
    verdict = ("(a) ADAPTERS DIED" if np.abs(all_on).mean() < 0.02 * np.abs(base).mean()
               else "(b) GATES FROZE" if Z.std(1).mean() < 0.01
               else "neither -- the mixture is alive")
    print(f"  VERDICT                         {verdict}")
    return dict(tag=tag, base=float(np.abs(base).mean()),
                adapters_all_on=float(np.abs(all_on).mean()),
                gate_rate=float(Z.mean()), draw_spread=float(Z.std(1).mean()),
                logit_std=lg_std, verdict=verdict)


