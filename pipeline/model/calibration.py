"""
Calibration and selective-prediction metrics (NLL, ECE, Brier, BALD, temperature
scaling).  The classifier is fixed; only the gate sampler differs between methods.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import model
import field as fieldmod

NS, SW = 200, 30                               # gate samples / Gibbs sweeps per point
T_GRID = np.linspace(0.3, 6.0, 60)


# ---------------------------------------------------------------- gate samplers
def prior_gate_fn(params):
    """Gate sampler from the learned prior p(z|x)."""
    kind, fp = params["field"]; J = params["J"]; K = J.shape[0]
    def fn(key, x, n_samp):
        h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
        z0 = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float64)
        return model.gibbs(key, h, J, z0, SW, K)
    return fn


def dropout_gate_fn(kappa, K):
    """Input-agnostic Bernoulli(kappa) gate sampler (x is ignored on purpose)."""
    def fn(key, x, n_samp):
        return (jax.random.uniform(key, (n_samp, K)) < kappa).astype(jnp.float64)
    return fn


def sample_logits(params, gate_fn, X, seed=0):
    """Per-sample classifier logits for each held-out x: (N, NS, C)."""
    W0, B, A = params["W0"], params["B"], params["A"]
    D = W0.shape[1]
    def one(key, x):
        z = gate_fn(key, x, NS)                          # (NS, K)
        return model.batch_logits(W0, B, A, jnp.broadcast_to(x, (NS, D)), z)  # (NS, C)
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    return jax.jit(jax.vmap(one))(keys, X)               # (N, NS, C)


# ---------------------------------------------------------------- metrics
def probs_at_T(L, T):
    # softmax per gate draw, then average: a mixture, not a rescaled classifier
    return jax.nn.softmax(L / T, -1).mean(1)             # (N, C)


def nll(pm, Y):
    return float(-jnp.mean(jnp.log(pm[jnp.arange(pm.shape[0]), Y] + 1e-12)))


def brier(pm, Y):
    """Multiclass Brier score, bounded in [0, 2]."""
    P = np.asarray(pm)
    oh = np.zeros_like(P)
    oh[np.arange(P.shape[0]), np.asarray(Y)] = 1.0
    return float(((P - oh) ** 2).sum(1).mean())


def ece(pm, Y, nbins=15):
    conf = np.asarray(pm.max(1)); pred = np.asarray(pm.argmax(1))
    corr = (pred == np.asarray(Y)).astype(float); N = len(Y); e = 0.0
    for b in range(nbins):
        lo, hi = b / nbins, (b + 1) / nbins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / N) * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def bald_of(L):
    """BALD from pre-sampled logits: H(mean p) - mean H(p)."""
    probs = jax.nn.softmax(L, -1)                        # (N, NS, C)
    pm = probs.mean(1)
    H = lambda q: -(q * jnp.log(q + 1e-12)).sum(-1)
    return H(pm) - H(probs).mean(1)                      # (N,)


def fit_T(L_va, Y_va):
    """Grid search for the temperature with the lowest validation NLL."""
    return float(min(T_GRID, key=lambda T: nll(probs_at_T(L_va, float(T)), Y_va)))


def selective(L, Y, coverages=(1.0, 0.75, 0.5, 0.25)):
    """Accuracy at each coverage after discarding the highest-BALD points first."""
    mi = np.asarray(bald_of(L))
    corr = (np.asarray(probs_at_T(L, 1.0).argmax(1)) == np.asarray(Y)).astype(float)
    order = np.argsort(mi)
    return {c: float(corr[order[:max(1, int(c * len(Y)))]].mean()) for c in coverages}


def mean_gate_rate(params, X, seed=1):
    """Mean fraction of gates on under the prior (dropout keep-rate matching)."""
    fn = prior_gate_fn(params)
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    z = jax.vmap(lambda k, x: fn(k, x, 64))(keys, X)     # (N, 64, K)
    return float(z.mean())


# ---------------------------------------------------------------- driver
def evaluate(name, params, gate_fn, Xte, Yte, Xva, Yva, out=None):
    """Fit T* on (Xva,Yva) by NLL, then report raw and calibrated metrics on (Xte,Yte).
       `out`: optional dict that receives the per-sample probabilities and BALD."""
    Lte = sample_logits(params, gate_fn, Xte, seed=2)
    Lva = sample_logits(params, gate_fn, Xva, seed=3)
    T = fit_T(Lva, Yva)
    pm1, pmT = probs_at_T(Lte, 1.0), probs_at_T(Lte, T)
    acc = float((np.asarray(pm1.argmax(1)) == np.asarray(Yte)).mean())
    if out is not None:
        out.update(pm_raw=np.asarray(pm1), pm_cal=np.asarray(pmT),
                   Y=np.asarray(Yte), bald=np.asarray(bald_of(Lte)), T=T)
    return dict(name=name, acc=acc, bald=float(bald_of(Lte).mean()),
                ece_raw=ece(pm1, Yte), ece_cal=ece(pmT, Yte),
                nll_raw=nll(pm1, Yte), nll_cal=nll(pmT, Yte),
                brier_raw=brier(pm1, Yte), brier_cal=brier(pmT, Yte), T=T,
                sel=selective(Lte, Yte))
