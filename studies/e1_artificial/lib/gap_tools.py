"""Exact variational-gap tools: enumerated true posterior, KL and moment metrics,
and refits of one recognition family to a fixed posterior table.  All by 2^K
enumeration, so K <= 14.
"""
from __future__ import annotations
import os as _os, sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import ebm_recognition as ebm          # puts pipeline/model on sys.path, enables x64
import jax
import jax.numpy as jnp
import tree_recognition as tr
import model as modelmod


CFG_BLOCK = 128       # likelihood table built in config blocks to bound memory


def true_posterior_logp(W0, B, A, X, Y, h, J, spec):
    """Exact log p(z|x,y) over all configs -> (N, 2^K)."""
    cfg = tr.all_configs(spec.K).astype(X.dtype)                     # (Z,K)
    en_prior = ebm.prior_pre(h, J, spec) @ ebm._suff(cfg, spec).T    # (N,Z)

    @jax.jit
    def _block(cz):
        return jax.vmap(lambda zc: modelmod.log_lik(
            W0, B, A, X, jnp.broadcast_to(zc, (X.shape[0], spec.K)), Y))(cz)
    LL = jnp.concatenate([_block(cfg[i:i + CFG_BLOCK])
                          for i in range(0, cfg.shape[0], CFG_BLOCK)], 0).T   # (N,Z)
    lo = en_prior + LL
    return lo - jax.scipy.special.logsumexp(lo, axis=1, keepdims=True)


def posterior_corr(logp, K):
    """Mean |off-diagonal correlation| of an enumerated distribution."""
    cfg = tr.all_configs(K).astype(logp.dtype)
    mu, M, _ = tr.moments_from_logq(logp, cfg)
    cov = M - jnp.einsum("nk,nj->nkj", mu, mu)
    sd = jnp.sqrt(jnp.clip(mu * (1 - mu), 1e-12))
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    off = ~jnp.eye(K, dtype=bool)
    return float(jnp.abs(corr[:, off]).mean())


def dist_metrics(logq, logp, K):
    """KL(q||p) and moment errors between two enumerated log-tables (N, 2^K)."""
    cfg = tr.all_configs(K).astype(logq.dtype)
    kl = (jnp.exp(logq) * (logq - logp)).sum(1)
    mu_q, M_q, _ = tr.moments_from_logq(logq, cfg)
    mu_p, M_p, _ = tr.moments_from_logq(logp, cfg)
    cov_q = M_q - jnp.einsum("nk,nj->nkj", mu_q, mu_q)
    cov_p = M_p - jnp.einsum("nk,nj->nkj", mu_p, mu_p)
    off = ~jnp.eye(K, dtype=bool)
    return dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()),
                mu_err=float(jnp.abs(mu_q - mu_p).mean()),
                cov_err=float(jnp.abs((cov_q - cov_p)[:, off]).mean()))


def adam_fit(loss_fn, params, steps=1500, lr=0.05):
    """Full-batch Adam with linear lr decay to 0.  The decay is needed for the
       amortized ebm fit, which oscillates under a constant lr."""
    b1, b2, eps = 0.9, 0.999, 1e-8

    def step(carry, t):
        prm, m, v = carry
        g = jax.grad(loss_fn)(prm)
        m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        mh = jax.tree.map(lambda a: a / (1 - b1 ** (t + 1.0)), m)
        vh = jax.tree.map(lambda a: a / (1 - b2 ** (t + 1.0)), v)
        lr_t = lr * (1.0 - t / steps)
        prm = jax.tree.map(lambda p, a, b: p - lr_t * a / (jnp.sqrt(b) + eps), prm, mh, vh)
        return (prm, m, v), None

    z = jax.tree.map(jnp.zeros_like, params)
    run = jax.jit(lambda p0: jax.lax.scan(step, (p0, z, z),
                                          jnp.arange(steps, dtype=jnp.float64))[0][0])
    return run(params)


def mst_spec_from_logp(logp, K):
    """Chow-Liu tree topology from the target posterior's own pairwise moments;
       pass the result to fit_family via spec_R."""
    import recognition as recog
    cfg = tr.all_configs(K).astype(logp.dtype)
    mu, M, _ = tr.moments_from_logq(logp, cfg)
    parents = tr.chow_liu_parents(mu, M, weight="abs_cov", root=0)
    spec = tr.make_tree_from_parents(parents)
    return spec, recog.REGISTRY["tree"]


def fit_family(key, family, K, feat, logp, amortized=True, steps=2500, lr=0.03,
               head_init=0.1, topology="chain", spec_R=None):
    """Fit one family's q to a fixed enumerated posterior by exact reverse-KL descent.
       amortized=True fits the linear head (U, c) on feat; amortized=False fits a
       free per-input pre, the family's best possible gap.  spec_R overrides the
       spec/bundle.  Returns (metrics dict, fitted logq table)."""
    import recognition as recog
    spec, R = spec_R if spec_R is not None else recog.make(family, K, None, topology, 0)
    N = feat.shape[0]
    if amortized:
        U = head_init * jax.random.normal(key, (spec.n_pre, feat.shape[1]))
        if family.startswith("ebm"):
            U = U.at[spec.K:, :].set(0.0)             # couplings start at zero, else the fit diverges
        prm = dict(U=U, c=jnp.zeros((spec.n_pre,)))
        pre_of = lambda p: feat @ p["U"].T + p["c"]
    else:
        P = head_init * jax.random.normal(key, (N, spec.n_pre))
        if family.startswith("ebm"):
            P = P.at[:, spec.K:].set(0.0)
        prm = dict(P=P)
        pre_of = lambda p: p["P"]

    def loss(p):
        logq = R["enumerate_logq"](pre_of(p), spec)
        return (jnp.exp(logq) * (logq - logp)).sum(1).mean()

    prm = adam_fit(loss, prm, steps=steps, lr=lr)
    logq = R["enumerate_logq"](pre_of(prm), spec)
    return dist_metrics(logq, logp, K), logq
