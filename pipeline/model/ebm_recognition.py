"""
EBM recognition family q(z|x,y) = exp(f'z + 1/2 z'Jq z)/Z, enumerated backend.
Head layout: pre[:, :K] = fields, pre[:, K:] = upper-triangle couplings (i<j), so
n_pre = K + K(K-1)/2.  All quantities are exact sums over the 2^K configs (K <= 14).
"""
from __future__ import annotations
import os as _os, sys as _sys
from typing import NamedTuple
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # allow sibling imports

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import tree_recognition as tr

MAX_K_ENUM = 14        # 2^14 configs; larger K is refused by the enum backend


class EBMSpec(NamedTuple):
    K: int
    n_pre: int                  # K + K(K-1)/2
    iu: tuple                   # iu[p], ju[p] = the (i<j) gate pair of coupling column K+p
    ju: tuple


def make_ebm_spec(K: int) -> EBMSpec:
    if K > MAX_K_ENUM:
        raise ValueError(
            f"ebm (enum backend) supports K <= {MAX_K_ENUM}, got K={K}: the exact "
            f"forward sums 2^K configs.  Larger K needs the (stage-4) Gibbs backend.")
    # pair order (0,1),(0,2),...,(1,2),...; the head columns and _suff share it
    iu = tuple(i for i in range(K) for j in range(i + 1, K))
    ju = tuple(j for i in range(K) for j in range(i + 1, K))
    return EBMSpec(K=K, n_pre=K + K * (K - 1) // 2, iu=iu, ju=ju)


def _ebm_spec(K, node_order=None, topology="chain", seed=0, task_parents=None):
    # node_order / topology / task_parents have no meaning for a fully connected family
    return make_ebm_spec(K)


def _tables(spec: EBMSpec, dtype):
    """cfg (Z,K) all configs; S (Z,n_pre) their sufficient statistics [z, z_i z_j]."""
    cfg = tr.all_configs(spec.K).astype(dtype)                     # (Z,K)
    # int32 pinned: an empty tuple (K=1) would otherwise become a float indexer
    iu, ju = jnp.asarray(spec.iu, jnp.int32), jnp.asarray(spec.ju, jnp.int32)
    S = jnp.concatenate([cfg, cfg[:, iu] * cfg[:, ju]], axis=1)    # (Z,n_pre)
    return cfg, S


def _suff(z, spec: EBMSpec):
    """Sufficient statistics of given configs: z:(...,K) -> (...,n_pre)."""
    iu, ju = jnp.asarray(spec.iu, jnp.int32), jnp.asarray(spec.ju, jnp.int32)
    return jnp.concatenate([z, z[..., iu] * z[..., ju]], axis=-1)


def ebm_forward(pre, spec: EBMSpec):
    """Exact moments and entropy by enumeration.  pre:(N,n_pre) -> mu:(N,K), M:(N,K,K), H:(N,)."""
    cfg, S = _tables(spec, pre.dtype)
    logits = pre @ S.T                                             # (N,Z)
    logZ = jax.scipy.special.logsumexp(logits, axis=-1)            # (N,)
    p = jax.nn.softmax(logits, axis=-1)                            # (N,Z)
    ES = p @ S                                                     # (N,n_pre) E_q[s(z)]
    mu = ES[:, :spec.K]
    m2 = ES[:, spec.K:]
    iu, ju = jnp.asarray(spec.iu, jnp.int32), jnp.asarray(spec.ju, jnp.int32)
    dg = jnp.arange(spec.K)
    M = jnp.zeros(pre.shape[:-1] + (spec.K, spec.K), pre.dtype)
    M = M.at[..., iu, ju].set(m2)
    M = M.at[..., ju, iu].set(m2)
    M = M.at[..., dg, dg].set(mu)                                  # E[z^2]=E[z] for binary z
    H = logZ - (pre * ES).sum(-1)                                  # H = logZ - pre.E[s]
    return mu, M, H


def ebm_sample(key, pre, spec: EBMSpec, n_samples: int = 1):
    """Exact sampling: one categorical draw over the enumerated configs -> (T,N,K) float32."""
    cfg, S = _tables(spec, pre.dtype)
    logits = pre @ S.T                                             # (N,Z)
    idx = jax.random.categorical(key, logits,
                                 shape=(n_samples,) + logits.shape[:-1])   # (T,N)
    return cfg.astype(jnp.float32)[idx]                            # (T,N,K)


def ebm_sample_relaxed(key, pre, spec: EBMSpec, tau: float = 1.0, n_samples: int = 1):
    raise NotImplementedError(
        "family='ebm' has no relaxed sampler; run estimator='sfe-loo' (and match the "
        "comparison arms to sfe-loo as well -- see model/estimators.py).")


def ebm_logq(z, pre, spec: EBMSpec):
    """Exact normalised log q(z), differentiable in pre.  z:(...,N,K) -> (...,N)."""
    cfg, S = _tables(spec, pre.dtype)
    logZ = jax.scipy.special.logsumexp(pre @ S.T, axis=-1)         # (N,)
    return (_suff(z, spec) * pre).sum(-1) - logZ


def ebm_score(z, pre, spec: EBMSpec):
    """d log q / d pre at z:  s(z) - E_q[s].  z:(...,N,K) -> (...,N,n_pre)."""
    cfg, S = _tables(spec, pre.dtype)
    ES = jax.nn.softmax(pre @ S.T, axis=-1) @ S                    # (N,n_pre)
    return _suff(z, spec) - ES


def ebm_enumerate_logq(pre, spec: EBMSpec, configs=None):
    """Exact log q for every config (or the given ones); normalised over the full 2^K space."""
    cfg, S = _tables(spec, pre.dtype)
    logits = pre @ S.T
    logZ = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    if configs is None:
        return logits - logZ
    return pre @ _suff(configs.astype(pre.dtype), spec).T - logZ


def fields(pre, spec: EBMSpec):
    """Recognition fields f(x,y):  pre:(N,n_pre) -> (N,K)."""
    return pre[..., :spec.K]


def couplings(pre, spec: EBMSpec):
    """Recognition couplings Jq(x,y) as a symmetric hollow (N,K,K) matrix."""
    iu, ju = jnp.asarray(spec.iu, jnp.int32), jnp.asarray(spec.ju, jnp.int32)
    jq = pre[..., spec.K:]
    Jm = jnp.zeros(pre.shape[:-1] + (spec.K, spec.K), pre.dtype)
    Jm = Jm.at[..., iu, ju].set(jq)
    return Jm + jnp.swapaxes(Jm, -1, -2)


def prior_pre(h, J, spec: EBMSpec):
    """Pack prior fields h:(N,K) and global J:(K,K) into the same natural-parameter layout."""
    iu, ju = jnp.asarray(spec.iu, jnp.int32), jnp.asarray(spec.ju, jnp.int32)
    jp = J[iu, ju]                                     # (P,) one value per unordered pair
    return jnp.concatenate([h, jnp.broadcast_to(jp, h.shape[:-1] + jp.shape)], axis=-1)


BUNDLE = dict(make_spec=_ebm_spec, forward=ebm_forward,
              sample=ebm_sample, sample_relaxed=ebm_sample_relaxed,
              logq=ebm_logq, score=ebm_score, enumerate_logq=ebm_enumerate_logq)
