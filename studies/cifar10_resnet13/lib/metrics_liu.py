"""Paper metrics for every approach, computed from a single logit tensor L of shape (N, S, C).

A deterministic network is the S = 1 case.  Headline ece/nll are untempered; the *_cal
columns use a fitted temperature.  Entropy decomposition follows Eqs. 8-9 of the paper.
"""
from __future__ import annotations
import os
import sys

import numpy as np

from paths import MODEL_DIR as _MODEL, HERE as _H
_PIPE = os.path.dirname(os.path.dirname(_H))          # .../pipeline
for _p in (_MODEL, _PIPE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                                    # noqa: E402
from jax import config as _jax_config                         # noqa: E402
_jax_config.update("jax_enable_x64", True)                    # float64, as in the engine
import jax.numpy as jnp                                       # noqa: E402
import calibration as cal                                     # noqa: E402
import ood_metrics as _ood                                    # noqa: E402

N_SAMP = 100                 # gate samples per input, following the paper
SWEEPS = _ood.SWEEPS         # Gibbs sweeps per sample
_CHUNK = _ood._CHUNK         # inputs per vmapped batch, to bound memory
_entropies = _ood._entropies  # Eq. 8/9 decomposition, one definition


def sample_logits_ebm(params, X, n_samp=N_SAMP, sweeps=SWEEPS, seed=2,
                      gate="prior", kappa=None):
    """(N, n_samp, C) logits: one row per input, one column per draw of the gates.

       `gate` swaps the gate distribution while the classifier is held identical:
       'prior' is the learned input-conditional p(z|x), 'allon' is z = 1, 'dropout' is an
       input-agnostic Bernoulli(kappa).  The prior path delegates to ood_metrics."""
    if gate == "prior":
        return _ood._sample_logits(params, X, n_samp, sweeps=sweeps, seed=seed)
    if gate == "dropout" and kappa is None:
        raise ValueError("gate='dropout' needs kappa (match it to the prior's gate rate)")

    import model                                                    # noqa: E402
    J, W0, B, A = params["J"], params["W0"], params["B"], params["A"]
    K, D = J.shape[0], W0.shape[1]

    def one(key, x):
        if gate == "allon":
            z = jnp.ones((n_samp, K), jnp.float64)
        elif gate == "dropout":
            # x is ignored on purpose: this gate cannot route.
            z = (jax.random.uniform(key, (n_samp, K)) < kappa).astype(jnp.float64)
        else:
            raise ValueError(f"unknown gate mode {gate!r}")
        return model.batch_logits(W0, B, A, jnp.broadcast_to(x, (n_samp, D)), z)

    run = jax.jit(jax.vmap(one))
    X = jnp.asarray(np.asarray(X, np.float64))
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    return np.concatenate([np.asarray(run(keys[i:i + _CHUNK], X[i:i + _CHUNK]))
                           for i in range(0, X.shape[0], _CHUNK)])


def prior_gate_rate(params, X, n_samp=32, sweeps=SWEEPS, seed=1):
    """Mean fraction of gates the learned prior turns on; used as the kappa that makes the
       dropout control equally sparse."""
    import model                                                    # noqa: E402
    import field as fieldmod                                        # noqa: E402
    kind, fp = params["field"]
    J = params["J"]
    K = J.shape[0]

    def one(key, x):
        h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
        z0 = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float64)
        return model.gibbs(key, h, J, z0, sweeps, K)

    X = jnp.asarray(np.asarray(X[:1000], np.float64))
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    z = np.concatenate([np.asarray(jax.jit(jax.vmap(one))(keys[i:i + _CHUNK],
                                                          X[i:i + _CHUNK]))
                        for i in range(0, X.shape[0], _CHUNK)])
    return float(z.mean())


def logits_det(L):
    """A deterministic network's (N, C) logits as the S = 1 case of the same interface."""
    return np.asarray(L, np.float64)[:, None, :]


def gate_stats(params, X, C, rank, n_samp=64, sweeps=SWEEPS, seed=1):
    """Gate occupancy of the learned prior, and whether it can span the C outputs.

       One expert's map B_k A_k has rank at most min(rank, C, D), so ceil(C / rank_ba)
       experts have to be on at the same time.  Drawn from the prior rather than
       enumerated, so the cost does not grow with K."""
    import model                                                # noqa: E402
    import field as fieldmod                                    # noqa: E402
    kind, fp = params["field"]
    J = params["J"]
    K = J.shape[0]
    X = jnp.asarray(np.asarray(X, np.float64))

    def one(key, x):
        h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
        z0 = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float64)
        return model.gibbs(key, h, J, z0, sweeps, K)            # (n_samp, K)

    run = jax.jit(jax.vmap(one))
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    z = np.concatenate([np.asarray(run(keys[i:i + _CHUNK], X[i:i + _CHUNK]))
                        for i in range(0, X.shape[0], _CHUNK)])   # (N, n_samp, K)
    on = z.sum(-1)                                                # (N, n_samp)
    rank_ba = min(rank, C, params["W0"].shape[1])
    needed = int(np.ceil(C / rank_ba))
    # `needed` can exceed K; then no gate pattern spans the output space and the setting is
    # infeasible before training, which is a different diagnosis from a prior that keeps
    # gates off.
    return dict(gate_rate=float(z.mean()), gates_on=float(on.mean()),
                gates_on_sd=float(on.std()), K=int(K),
                rank_ba=int(rank_ba), gates_needed=needed,
                feasible=bool(needed <= K),
                frac_needed=float(needed / K),
                p_enough=float((on >= needed).mean()))


# ----------------------------------------------------------------- the metrics
def decompose(L, T=1.0):
    """Paper Eqs. 8-9 on a logit tensor: (H_total, H_aleatoric, H_epistemic), each (N,)."""
    return _entropies(L, T)


def summarise(name, L, Y, T=1.0):
    """Every scalar of one row of the comparison table.  `T` applies only to *_cal."""
    L = np.asarray(L, np.float64)
    Y = np.asarray(Y)
    pm_raw = np.asarray(cal.probs_at_T(jnp.asarray(L), 1.0))
    pm_cal = np.asarray(cal.probs_at_T(jnp.asarray(L), T)) if T != 1.0 else pm_raw
    h_tot, h_ale, h_epi = decompose(L)
    return dict(
        name=name, n=len(Y), n_samples=L.shape[1],
        acc=float((pm_raw.argmax(1) == Y).mean()),
        ece=cal.ece(jnp.asarray(pm_raw), Y),          # the paper-comparable number
        nll=cal.nll(jnp.asarray(pm_raw), Y),
        # bounded proper score; unlike NLL it also charges for mass on the wrong classes
        brier=cal.brier(pm_raw, Y),
        ece_cal=cal.ece(jnp.asarray(pm_cal), Y), nll_cal=cal.nll(jnp.asarray(pm_cal), Y),
        brier_cal=cal.brier(pm_cal, Y),
        T=float(T),
        h_total=float(h_tot.mean()), h_alea=float(h_ale.mean()), h_epi=float(h_epi.mean()))


def fit_T(L_va, Y_va):
    """The temperature that minimises validation NLL (60-point grid, pipeline default)."""
    return cal.fit_T(jnp.asarray(np.asarray(L_va, np.float64)), np.asarray(Y_va))


def ood_sweep(logit_fn, Xte, Yte, fractions, probe, n=1000, seed=0, T=1.0, verbose=True):
    """Recompute every metric across the SVHN-fraction ladder.

       `logit_fn(X_raw) -> (N, S, C)` applies its own front-end, which is why `probe` hands
       out raw images.  `probe` is cifar10_data.probe_raw."""
    rows = []
    for f in fractions:
        # One fixed seed per fraction, so every row sees the same blended images.
        Xb, Yb = probe(float(f), Xte, Yte, n, seed + int(round(f * 100)))
        r = summarise(f"f={f:.1f}", logit_fn(Xb), Yb, T=T)
        r["fraction"] = float(f)
        rows.append(r)
        if verbose:
            print(f"    f={f:.1f}  acc {r['acc']:.4f}  ECE {r['ece']:.4f}  "
                  f"H_tot {r['h_total']:.3f}  H_epi {r['h_epi']:.3f}", flush=True)
    return rows


# Empty: the paper reports Fashion-MNIST and CIFAR-100, not CIFAR-10.  Kept so that figure
# scripts written against the CIFAR-100 sibling still import and simply render no published
# rows; the sibling's 100-class numbers must not be copied in here.
PAPER = []
