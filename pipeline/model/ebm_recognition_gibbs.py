"""
EBM recognition family, Gibbs backend ("ebm-gibbs").  Same parametrisation as
ebm_recognition.py, but moments are estimated from Gibbs chains with surrogate terms
whose autodiff gradient is the score-function estimator grad_pre E_q[f] = Cov_q(f, s).
Requires estimator='sfe-loo'. 
"""
from __future__ import annotations
import os as _os, sys as _sys
from typing import NamedTuple
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # allow sibling imports

import ebm_recognition as ebm
import jax
import jax.numpy as jnp


SWEEPS = 12          # Gibbs sweeps per estimate
N_CHAINS = 64        # parallel chains per input


class GibbsSpec(NamedTuple):
    K: int
    n_pre: int
    iu: tuple
    ju: tuple
    sweeps: int
    n_chains: int


def _gibbs_spec(K, node_order=None, topology="chain", seed=0, task_parents=None):
    # built directly, not via ebm.make_ebm_spec, so the K <= 14 enum limit does not apply
    iu = tuple(i for i in range(K) for j in range(i + 1, K))
    ju = tuple(j for i in range(K) for j in range(i + 1, K))
    return GibbsSpec(K=K, n_pre=K + K * (K - 1) // 2, iu=iu, ju=ju,
                     sweeps=int(SWEEPS), n_chains=int(N_CHAINS))


def _derive_key(pre):
    """Deterministic key from the head output; forward() has no key parameter."""
    h = jnp.sum(pre * 12289.0) + jnp.sum(pre * pre * 127.0)
    seed = jnp.mod(jnp.abs(h), 2147483647.0).astype(jnp.int32)
    return jax.random.fold_in(jax.random.key(1618), seed)


def _fields_couplings(pre, spec):
    f = pre[..., :spec.K]                                       # (N,K)
    Jq = ebm.couplings(pre, spec)                               # (N,K,K) symmetric, hollow
    return f, Jq


def _gibbs(key, f, Jq, z, sweeps, K):
    """Sequential single-site Gibbs with per-input couplings Jq:(N,K,K).  z:(T,N,K)."""
    def sweep(carry, _):
        z, key = carry
        for k in range(K):
            kk = jax.random.fold_in(key, k)
            field = f[None, :, k] + jnp.einsum("tnj,nj->tn", z, Jq[:, k, :])
            zk = jax.random.uniform(kk, field.shape) < jax.nn.sigmoid(field)
            z = z.at[..., k].set(zk.astype(z.dtype))
        return (z, jax.random.split(key)[0]), None
    (z, _), _ = jax.lax.scan(sweep, (z, key), None, length=sweeps)
    return z


def _chains(key, pre, spec, n):
    """n chains per input warm-started at sigmoid(fields) -> (n, N, K), no gradient."""
    f, Jq = _fields_couplings(jax.lax.stop_gradient(pre), spec)
    k0, k1 = jax.random.split(key)
    z0 = (jax.random.uniform(k0, (n,) + f.shape) < jax.nn.sigmoid(f)[None]).astype(
        jnp.float64)
    return _gibbs(k1, f, Jq, z0, spec.sweeps, spec.K)


def _zero_val(x):
    """Value 0, gradient of x."""
    return x - jax.lax.stop_gradient(x)


def _dice(val_tab, lq):
    """E_q[f] from chain samples: value = chain mean, gradient = LOO-baselined
       covariance with the differentiable unnormalised log-density lq:(T,N)."""
    v = jax.lax.stop_gradient(val_tab)
    T = v.shape[0]
    loo = (v.sum(0, keepdims=True) - v) / (T - 1)
    adv = v - loo                                               # (T,N,...) constants
    lqc = _zero_val(lq)                                         # (T,N) grad carrier
    extra = (adv * lqc.reshape(lqc.shape + (1,) * (v.ndim - 2))).mean(0)
    return v.mean(0) + extra


def gibbs_forward(pre, spec: GibbsSpec):
    """Sampled (mu, M, H) with surrogate gradients.  H is exact only up to +log Z_q."""
    z = _chains(_derive_key(pre), pre, spec, spec.n_chains)     # (T,N,K) constants
    s = ebm._suff(z, spec)                                      # (T,N,n_pre)
    lq = (s * pre).sum(-1)                                      # (T,N) differentiable
    Es = _dice(s, lq)                                           # (N,n_pre)
    mu = Es[..., :spec.K]
    m2 = Es[..., spec.K:]
    iu = jnp.asarray(spec.iu, jnp.int32)
    ju = jnp.asarray(spec.ju, jnp.int32)
    dg = jnp.arange(spec.K)
    M = jnp.zeros(pre.shape[:-1] + (spec.K, spec.K), pre.dtype)
    M = M.at[..., iu, ju].set(m2)
    M = M.at[..., ju, iu].set(m2)
    M = M.at[..., dg, dg].set(mu)
    lq_mean = lq.mean(0)
    H = (-jax.lax.stop_gradient(lq_mean)                        # value (up to +logZ)
         - _zero_val(lq_mean)                                   # pathwise part
         + _zero_val(_dice(-lq[..., None], lq)[..., 0])         # dice -Cov(lq, s)
         + _zero_val((pre * jax.lax.stop_gradient(Es)).sum(-1)))  # logZ grad +E[s]
    return mu, M, H


def gibbs_sample(key, pre, spec: GibbsSpec, n_samples: int = 1):
    """n_samples Gibbs draws per input -> (T,N,K) float32."""
    return _chains(key, pre, spec, n_samples).astype(jnp.float32)


def gibbs_sample_relaxed(key, pre, spec, tau: float = 1.0, n_samples: int = 1):
    raise NotImplementedError(
        "family='ebm-gibbs' has no relaxed sampler; run estimator='sfe-loo' "
        "(the LOO baseline is also what makes the unnormalised logq unbiased here).")


def gibbs_logq(z, pre, spec: GibbsSpec):
    """Unnormalised log q~(z) = pre . s(z).  Unbiased inside sfe-loo because the
       leave-one-out advantage has zero mean, which absorbs the missing grad log Z."""
    return (ebm._suff(z, spec) * pre).sum(-1)


def gibbs_score(z, pre, spec: GibbsSpec):
    """s(z) - E_q[s], with the expectation estimated from fresh chains."""
    zc = _chains(_derive_key(pre), pre, spec, spec.n_chains)
    Es = jax.lax.stop_gradient(ebm._suff(zc, spec).mean(0))
    return ebm._suff(z, spec) - Es


# enumerate_logq stays exact (same parametric family, K <= 14, diagnostics only)
BUNDLE = dict(make_spec=_gibbs_spec, forward=gibbs_forward,
              sample=gibbs_sample, sample_relaxed=gibbs_sample_relaxed,
              logq=gibbs_logq, score=gibbs_score,
              enumerate_logq=ebm.ebm_enumerate_logq)
