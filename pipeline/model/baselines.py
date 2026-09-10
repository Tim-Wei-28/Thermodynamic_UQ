"""
Input-agnostic gate baselines: the same free adapters trained under MC-dropout
gates (keep < 1) or with every gate on (keep = 1).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

import model


def _gclip(g, c):
    nrm = jnp.sqrt(jnp.sum(g ** 2))
    return jnp.where(nrm > c, g * (c / nrm), g)


def train_masked(seed, build_lik, make_data, keep, epochs, lr, wd, clip, T):
    """Free adapters trained under i.i.d. Bernoulli(keep) gates (keep>=1.0 => AllOn).

       build_lik(key) -> lik dict with W0, B, A;  make_data(key) -> (X, Y).
       Returns a params dict carrying `keep`, so readouts.acc_masked can read it back."""
    key = jax.random.key(seed)
    key, klik, kd = jax.random.split(key, 3)
    lik = build_lik(klik)
    W0 = lik["W0"]
    X, Y = make_data(kd)
    N, K = X.shape[0], lik["B"].shape[0]

    def loss(sub, key_):
        z = (jnp.ones((N, K)) if keep >= 1.0
             else (jax.random.uniform(key_, (T, N, K)) < keep).astype(jnp.float64))
        if keep >= 1.0:
            return -model.log_lik(W0, sub["B"], sub["A"], X, z, Y).mean()
        logp = jax.vmap(lambda zt: model.log_lik(W0, sub["B"], sub["A"], X, zt, Y))(z)
        return -logp.mean()
    grad = jax.jit(jax.value_and_grad(loss))

    def step(carry, key_):
        B, A = carry
        _, g = grad(dict(B=B, A=A), key_)
        B = B - lr * (_gclip(g["B"], clip) + wd * B)
        A = A - lr * (_gclip(g["A"], clip) + wd * A)
        return (B, A), None
    (B, A), _ = jax.lax.scan(step, (lik["B"], lik["A"]), jax.random.split(key, epochs))
    return dict(W0=W0, B=B, A=A, keep=keep)
