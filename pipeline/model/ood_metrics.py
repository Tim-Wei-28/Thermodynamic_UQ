"""
Scores one (possibly shifted) image set: accuracy, calibration metrics and the
entropy decomposition of Liu et al.  The temperature is fitted once
in-distribution and carried over unchanged.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import model
import field as fieldmod
import calibration as cal

SWEEPS = 30          # Gibbs sweeps per gate draw
_CHUNK = 500         # inputs per vmapped batch, caps the (chunk, n_samp, K) allocation


def _sample_logits(params, X, n_samp, sweeps=SWEEPS, seed=2):
    """(N, n_samp, C) logits, one column per draw from the learned gate prior."""
    kind, fp = params["field"]
    J, W0, B, A = params["J"], params["W0"], params["B"], params["A"]
    K, D = J.shape[0], W0.shape[1]

    def one(key, x):
        h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
        z0 = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float64)
        z = model.gibbs(key, h, J, z0, sweeps, K)
        return model.batch_logits(W0, B, A, jnp.broadcast_to(x, (n_samp, D)), z)

    run = jax.jit(jax.vmap(one))
    X = jnp.asarray(np.asarray(X, np.float64))
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    out = [np.asarray(run(keys[i:i + _CHUNK], X[i:i + _CHUNK]))
           for i in range(0, X.shape[0], _CHUNK)]
    return np.concatenate(out)


def _entropies(L, T=1.0):
    """(H_total, H_aleatoric, H_epistemic) per input, in nats."""
    P = np.asarray(jax.nn.softmax(jnp.asarray(L) / T, -1))       # (N, S, C)
    ent = lambda q: -(q * np.log(q + 1e-12)).sum(-1)             # noqa: E731
    h_tot = ent(P.mean(1))                                       # H(E[p])
    h_ale = ent(P).mean(1)                                       # E[H(p)]
    return h_tot, h_ale, h_tot - h_ale


def fit_T(params, X, Y, n_samp=100, seed=2):
    """Temperature minimising NLL on a held-out in-distribution split."""
    return cal.fit_T(jnp.asarray(_sample_logits(params, X, n_samp, seed=seed)),
                     np.asarray(Y))


def score(params, X, Y, n_samp=100, T=1.0, seed=2, n_class=None):
    """All metrics for one image set.  `T` applies to the *_cal entries only;
       the raw entries stay at T=1."""
    L = _sample_logits(params, X, n_samp, seed=seed)
    Y = np.asarray(Y)
    pm_raw = np.asarray(cal.probs_at_T(jnp.asarray(L), 1.0))
    pm_cal = np.asarray(cal.probs_at_T(jnp.asarray(L), float(T))) if T != 1.0 else pm_raw
    h_tot, h_ale, h_epi = _entropies(L)
    pred = pm_raw.argmax(1)
    out = dict(
        acc=float((pred == Y).mean()),
        ece=cal.ece(jnp.asarray(pm_raw), Y), nll=cal.nll(jnp.asarray(pm_raw), Y),
        brier=cal.brier(pm_raw, Y),
        ece_cal=cal.ece(jnp.asarray(pm_cal), Y), nll_cal=cal.nll(jnp.asarray(pm_cal), Y),
        brier_cal=cal.brier(pm_cal, Y),
        h_total=float(h_tot.mean()), h_alea=float(h_ale.mean()),
        h_epi=float(h_epi.mean()),      # equals BALD
    )
    sel = cal.selective(jnp.asarray(L), Y)
    out.update({f"sel_at_{str(c).replace('.', 'p')}": v for c, v in sel.items()})
    if n_class:
        out["per_class"] = {c: float((pred[Y == c] == c).mean()) if (Y == c).any()
                            else float("nan") for c in range(n_class)}
    return out
