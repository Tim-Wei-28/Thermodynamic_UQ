"""Scoring for the comparison. Every approach is reduced to one logit tensor of shape
(N, S, C), a deterministic net being the S = 1 case, and all metrics are functions of
that tensor. Accuracy, ECE, NLL and Brier come from the pipeline calibration module; the
entropy decomposition follows Liu et al. (2022) Eqs. 8-9, in nats.
"""
from __future__ import annotations
import os
import sys

import numpy as np

# .../pipeline/studies/fashion_lenet5/ -> .../pipeline/
from paths import MODEL_DIR as _MODEL, HERE as _H
_PIPE = os.path.dirname(os.path.dirname(_H))          # .../pipeline
for _p in (_MODEL, _PIPE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                                    # noqa: E402
from jax import config as _jax_config                         # noqa: E402
_jax_config.update("jax_enable_x64", True)                    # float64, as in engine.py
import jax.numpy as jnp                                       # noqa: E402
import model                                                  # noqa: E402
import field as fieldmod                                      # noqa: E402
import calibration as cal                                     # noqa: E402
import ood_metrics as _ood                                    # noqa: E402

# Scoring is imported from pipeline/model/ood_metrics.py, not reimplemented, so the
# standalone baselines and the pipeline sweep are measured by the same code.  Only the
# extra gate modes below are local; the pipeline has no use for them.
N_SAMP = 100                 # gate samples per input
SWEEPS = _ood.SWEEPS         # Gibbs sweeps per sample
_CHUNK = _ood._CHUNK         # inputs per vmapped batch
_entropies = _ood._entropies


def sample_logits_ebm(params, X, n_samp=N_SAMP, sweeps=SWEEPS, seed=2,
                      gate="prior", kappa=None):
    """(N, n_samp, C) logits, one row per input, one column per gate draw.
       Like calibration.sample_logits but chunked over inputs to bound the allocation.

       Gate modes, all sharing the same adapters and classifier:
         'prior'    learned p(z|x), field plus Gibbs through J; input-conditional.
         'allon'    z = 1 always, so no mixture and no stochasticity.
         'dropout'  input-independent Bernoulli(kappa); stochastic but cannot route.
    """
    kind, fp = params["field"]
    J, W0, B, A = params["J"], params["W0"], params["B"], params["A"]
    K, D = J.shape[0], W0.shape[1]
    if gate == "dropout" and kappa is None:
        raise ValueError("gate='dropout' needs kappa (match it to the prior's gate rate)")

    # The prior path is delegated to the pipeline sampler instead of duplicated here.
    if gate == "prior":
        return _ood._sample_logits(params, X, n_samp, sweeps=sweeps, seed=seed)

    def one(key, x):
        if gate == "allon":
            z = jnp.ones((n_samp, K), jnp.float64)
        elif gate == "dropout":
            # x is deliberately ignored here.
            z = (jax.random.uniform(key, (n_samp, K)) < kappa).astype(jnp.float64)
        else:
            raise ValueError(f"unknown gate mode {gate!r}")
        return model.batch_logits(W0, B, A, jnp.broadcast_to(x, (n_samp, D)), z)

    run = jax.jit(jax.vmap(one))
    X = jnp.asarray(np.asarray(X, np.float64))
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    out = [np.asarray(run(keys[i:i + _CHUNK], X[i:i + _CHUNK]))
           for i in range(0, X.shape[0], _CHUNK)]
    return np.concatenate(out)


def prior_gate_rate(params, X, n_samp=32, sweeps=SWEEPS, seed=1):
    """Average fraction of gates the learned prior turns on; use it as the dropout kappa
       so the two gate modes have the same sparsity."""
    kind, fp = params["field"]
    J, K = params["J"], params["J"].shape[0]

    def one(key, x):
        h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
        z0 = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float64)
        return model.gibbs(key, h, J, z0, sweeps, K)
    X = jnp.asarray(np.asarray(X[:1000], np.float64))
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    return float(np.asarray(jax.jit(jax.vmap(one))(keys, X)).mean())


def logits_det(L):
    """Wrap a deterministic network's (N, C) logits as the S = 1 case."""
    return np.asarray(L, np.float64)[:, None, :]


# ----------------------------------------------------------------- the metrics
def decompose(L, T=1.0):
    """Entropy decomposition of a logit tensor: (H_total, H_aleatoric, H_epistemic),
       each (N,).  Thin wrapper over ood_metrics so there is one definition."""
    return _entropies(L, T)


def summarise(name, L, Y, T=1.0):
    """All scalars for one row of the comparison table.
       T is applied only to the *_cal columns; the headline ECE and NLL stay at T = 1."""
    L = np.asarray(L, np.float64)
    Y = np.asarray(Y)
    pm_raw = np.asarray(cal.probs_at_T(jnp.asarray(L), 1.0))
    pm_cal = np.asarray(cal.probs_at_T(jnp.asarray(L), T)) if T != 1.0 else pm_raw
    h_tot, h_ale, h_epi = decompose(L)
    return dict(
        name=name, n=len(Y), n_samples=L.shape[1],
        acc=float((pm_raw.argmax(1) == Y).mean()),
        ece=cal.ece(jnp.asarray(pm_raw), Y),          # the column comparable to the paper
        nll=cal.nll(jnp.asarray(pm_raw), Y),
        # Bounded proper score; unlike NLL it is not dominated by a few confident errors.
        brier=cal.brier(pm_raw, Y),
        ece_cal=cal.ece(jnp.asarray(pm_cal), Y), nll_cal=cal.nll(jnp.asarray(pm_cal), Y),
        brier_cal=cal.brier(pm_cal, Y),
        T=float(T),
        h_total=float(h_tot.mean()), h_alea=float(h_ale.mean()), h_epi=float(h_epi.mean()))


def fit_T(L_va, Y_va):
    """Temperature that minimises validation NLL, on the pipeline's default grid."""
    return cal.fit_T(jnp.asarray(np.asarray(L_va, np.float64)), np.asarray(Y_va))


def ood_sweep(logit_fn, Xte, Yte, fractions, blend, n=1000, seed=0, T=1.0, verbose=True):
    """Recompute every metric across the letter-fraction ladder.
       logit_fn(X) -> (N, S, C) is what differs between approaches;
       blend(Xte, Yte, f, n, seed) -> (X, Y) is fashion_data.blend_probe."""
    rows = []
    for f in fractions:
        # Seed depends only on the fraction, so all approaches see the same blended images.
        Xb, Yb = blend(Xte, Yte, float(f), n=n, seed=seed + int(round(f * 100)))
        r = summarise(f"f={f:.1f}", logit_fn(Xb), Yb, T=T)
        r["fraction"] = float(f)
        rows.append(r)
        if verbose:
            print(f"    f={f:.1f}  acc {r['acc']:.4f}  ECE {r['ece']:.4f}  "
                  f"H_tot {r['h_total']:.3f}  H_epi {r['h_epi']:.3f}")
    return rows


def table(rows, cols=("name", "acc", "ece", "nll", "ece_cal", "nll_cal",
                      "h_total", "h_alea", "h_epi")):
    """Render rows as a fixed-width text table."""
    w = {c: max(len(c), max((len(_fmt(r.get(c))) for r in rows), default=0)) for c in cols}
    out = ["  ".join(c.rjust(w[c]) for c in cols),
           "  ".join("-" * w[c] for c in cols)]
    for r in rows:
        out.append("  ".join(_fmt(r.get(c)).rjust(w[c]) for c in cols))
    return "\n".join(out)


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# Reference rows from Liu et al. (2022), Table 3.
PAPER = [
    dict(name="Liu DNN (sw)", acc=0.9009, ece=0.0328, ece_ood50=0.3435),
    dict(name="Liu BNN (sw)", acc=0.9015, ece=0.0156, ece_ood50=0.0918),
    dict(name="Liu BNN (spin)", acc=0.8970, ece=0.0135, ece_ood50=0.1066),
]
