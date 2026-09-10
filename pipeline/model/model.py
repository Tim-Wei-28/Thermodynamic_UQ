"""
Base-model pieces shared by all recognition families: gated logits, log-likelihood,
linear field, single-site Gibbs on the prior, and prior-sampled prediction with BALD.
"""
from __future__ import annotations
import os as _os, sys as _sys
# model/ files import each other by plain name, so this directory must be on the path.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import jax
import jax.numpy as jnp

import field as fieldmod


def batch_logits(W0, B, A, X, Z):
    """W(z) x with W(z) = W0 + sum_k z_k B_k A_k.
       X:(N,D)  Z:(N,K)  W0:(C,D)  B:(K,C,r)  A:(K,r,D)  -> (N,C)."""
    a = jnp.einsum("krd,nd->nkr", A, X)              # a_k = A_k x
    adapter = jnp.einsum("nk,kcr,nkr->nc", Z, B, a)  # sum_k z_k B_k a_k
    return X @ W0.T + adapter


def log_lik(W0, B, A, X, Z, Y):
    """log p(y|x,z) for observed labels Y:(N,) -> (N,)."""
    lg = batch_logits(W0, B, A, X, Z)
    return jax.nn.log_softmax(lg)[jnp.arange(X.shape[0]), Y]


def fields(V, d, X):
    """h_theta(x) = V x + d  -> (N,K)."""
    return X @ V.T + d


def gibbs(key, h, J, Z, sweeps, K):
    """Sequential single-site Gibbs on the prior p(z|x):
         p(z_k=1 | .) = sigmoid(h_k + sum_j J_kj z_j).
       h:(N,K)  J:(K,K)  Z:(N,K) chain state.  Returns updated Z."""
    def sweep(carry, _):
        Z, key = carry
        for k in range(K):
            key_k = jax.random.fold_in(key, k)
            field = h[:, k] + Z @ J[k]               # local logit of gate k, not the field net
            zk = jax.random.uniform(key_k, (Z.shape[0],)) < jax.nn.sigmoid(field)
            Z = Z.at[:, k].set(zk.astype(Z.dtype))   # written back at once: sequential update
        return (Z, jax.random.split(key)[0]), None
    (Z, _), _ = jax.lax.scan(sweep, (Z, key), None, length=sweeps)
    return Z


def predict_bald(key, W0, B, A, field_spec, J, x, K, D, C, n_samp=200, sweeps=30):
    """Test-time prediction: sample gates from the prior p(z|x), average the sub-models.
       field_spec = (kind, field_params).  Returns (mean prob (C,), BALD mutual info)."""
    kind, fp = field_spec
    h = jnp.broadcast_to(fieldmod.apply(kind, fp, x[None])[0], (n_samp, K))
    z = (jax.random.uniform(key, (n_samp, K)) < 0.5).astype(jnp.float32)
    z = gibbs(key, h, J, z, sweeps, K)
    lg = batch_logits(W0, B, A, jnp.broadcast_to(x, (n_samp, D)), z)
    probs = jax.nn.softmax(lg, -1)                   # (n_samp, C)
    pm = probs.mean(0)
    H = lambda q: -(q * jnp.log(q + 1e-12)).sum(-1)
    return pm, H(pm) - H(probs).mean()               # (mean prob, BALD MI)
