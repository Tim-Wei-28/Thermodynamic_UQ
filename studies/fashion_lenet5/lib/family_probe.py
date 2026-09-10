"""Exact diagnostics for the recognition families by 2^K enumeration: the variational
gap KL(q(z|x,y) || p(z|x,y)), the prior KL, and gate correlations under p and q.
The prior p(z|x) is itself trained under the family, so a small gap alone does not
show that the family approximates well; corr_post is reported per arm for this reason.
"""
from __future__ import annotations
import os
import sys

import numpy as np

from paths import MODEL_DIR as _MODEL, HERE as _H
_PIPE = os.path.dirname(os.path.dirname(_H))
for _p in (_MODEL, _PIPE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import model                                                      # noqa: E402
import field as fieldmod                                          # noqa: E402
import tree_recognition as tr                                     # noqa: E402

# configs are walked in blocks; the per-config logits (block, N, C) do not fit in one vmap
CFG_BLOCK = 256
K_MAX = 12


def _corr_stats(p, cfg, K):
    """Mean and max |correlation| between gate pairs under p (N, Z) over configs cfg (Z, K)."""
    mu = p @ cfg                                                  # (N, K)
    M = jnp.einsum("nz,zk,zj->nkj", p, cfg, cfg)                  # (N, K, K)
    cov = M - mu[:, :, None] * mu[:, None, :]
    sd = jnp.sqrt(jnp.clip(mu * (1.0 - mu), 1e-12))
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    # gates that are always on or off have no defined correlation; exclude them
    dead = (mu < 1e-6) | (mu > 1 - 1e-6)                          # (N, K)
    live = ~(dead[:, :, None] | dead[:, None, :])
    off = ~jnp.eye(K, dtype=bool)[None, :, :]
    keep = live & off
    a = jnp.abs(corr) * keep
    n = jnp.maximum(keep.sum((1, 2)), 1)
    return a.sum((1, 2)) / n, a.max((1, 2))                       # (N,), (N,)


def exact_gap(params, spec, family, X, Y, C, n=2000, seed=0):
    """Exact variational gap KL(q || p(z|x,y)) and gate correlations under p and q.
    X: prepared features (constant column included). Y: labels, an input to q."""
    W0, B, A, U, c, J = (params[k] for k in ("W0", "B", "A", "U", "c", "J"))
    K = int(J.shape[0])
    if K > K_MAX:
        raise ValueError(f"exact_gap enumerates 2^K and K={K} exceeds {K_MAX}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    X = jnp.asarray(np.asarray(X)[idx])
    Y = jnp.asarray(np.asarray(Y)[idx], dtype=jnp.int32)
    N = X.shape[0]
    feat = jnp.concatenate([X, jax.nn.one_hot(Y, C)], 1)          # as trainer.py builds it
    pre = feat @ U.T + c

    cfg = tr.all_configs(K)                                       # (Z, K)
    enum = tr.mf_enumerate_logq if family == "mf" else tr.tree_enumerate_logq
    logq = enum(pre, spec, cfg)                                   # (N, Z)
    logq = logq - jax.scipy.special.logsumexp(logq, 1, keepdims=True)

    # true posterior p(z|x,y) proportional to p(y|z,x) p(z|x)
    h = fieldmod.apply(params["field"][0], params["field"][1], X)  # (N, K)
    quad = 0.5 * jnp.einsum("zk,kj,zj->z", cfg, J, cfg)            # (Z,)
    en_prior = h @ cfg.T + quad[None, :]                           # (N, Z) log p(z|x) + const

    @jax.jit
    def _block(cz):
        """log p(y|z,x) for a block of configs -> (block, N)."""
        def one(z):
            lg = model.batch_logits(W0, B, A, X, jnp.broadcast_to(z, (N, K)))
            return jax.nn.log_softmax(lg, -1)[jnp.arange(N), Y]
        return jax.vmap(one)(cz)

    LL = jnp.concatenate([_block(cfg[i:i + CFG_BLOCK])
                          for i in range(0, cfg.shape[0], CFG_BLOCK)], 0).T   # (N, Z)

    logp = en_prior + LL
    logp = logp - jax.scipy.special.logsumexp(logp, 1, keepdims=True)

    q = jnp.exp(logq)
    kl = (q * (logq - logp)).sum(1)                                # (N,) nats
    cpost_m, cpost_x = _corr_stats(jnp.exp(logp), cfg, K)
    cq_m, _cq_x = _corr_stats(q, cfg, K)
    return dict(
        kl_qp=float(kl.mean()), kl_qp_se=float(kl.std(ddof=1) / np.sqrt(N)),
        corr_post=float(cpost_m.mean()), corr_post_max=float(cpost_x.mean()),
        corr_q=float(cq_m.mean()),
        corr_lost=float((cpost_m - cq_m).mean()),
        n_probe=int(N))


def exact_prior_kl(params, spec, family, X, Y, C, n=2000, seed=0):
    """Exact KL(q(z|x,y) || p(z|x)), the ELBO regulariser, with logZ included.
    The trainer's traced kl drops logZ and is not comparable across runs with
    different priors."""
    U, c, J = params["U"], params["c"], params["J"]
    K = int(J.shape[0])
    if K > K_MAX:
        raise ValueError(f"exact_prior_kl enumerates 2^K and K={K} exceeds {K_MAX}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    X = jnp.asarray(np.asarray(X)[idx])
    Y = jnp.asarray(np.asarray(Y)[idx], dtype=jnp.int32)
    pre = jnp.concatenate([X, jax.nn.one_hot(Y, C)], 1) @ U.T + c

    cfg = tr.all_configs(K)
    enum = tr.mf_enumerate_logq if family == "mf" else tr.tree_enumerate_logq
    logq = enum(pre, spec, cfg)
    logq = logq - jax.scipy.special.logsumexp(logq, 1, keepdims=True)

    h = fieldmod.apply(params["field"][0], params["field"][1], X)
    quad = 0.5 * jnp.einsum("zk,kj,zj->z", cfg, J, cfg)
    logpri = h @ cfg.T + quad[None, :]
    logpri = logpri - jax.scipy.special.logsumexp(logpri, 1, keepdims=True)

    q = jnp.exp(logq)
    kl = (q * (logq - logpri)).sum(1)
    return dict(kl_qprior=float(kl.mean()),
                kl_qprior_se=float(kl.std(ddof=1) / np.sqrt(len(kl))))


def edge_set(parents):
    """Parents array to undirected edge set (root-independent)."""
    return {tuple(sorted((int(k), int(p)))) for k, p in enumerate(parents)
            if int(p) >= 0 and int(p) != int(k)}


def edge_agreement(pa, pb):
    """Fraction of shared edges between two trees over the same K gates."""
    ea, eb = edge_set(pa), edge_set(pb)
    if not ea or not eb:
        return float("nan")
    return len(ea & eb) / len(ea)


def chain_agreement(parents, K):
    """Edge agreement of a learned tree with the warm-up chain 0-1-...-(K-1)."""
    return edge_agreement(parents, [-1] + list(range(K - 1)))
