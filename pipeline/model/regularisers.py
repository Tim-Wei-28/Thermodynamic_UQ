"""
Regulariser terms of the training objective.
All functions consume recognition moments (mu, M, H), not a family.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def kl_freebits(mu, M, H, h, J, free_bits):
    """Per-point KL(q || p(z|x)) in nats up to +logZ, with a free-bits floor on the
       per-gate marginal proxy.  mu:(N,K) M:(N,K,K) H:(N,) h:(N,K) J:(K,K) -> (N,)"""
    # logZ is dropped: it has no recognition gradient and re-enters via the prior update.
    quad = 0.5 * jnp.einsum("kj,nkj->n", J, M)
    kl_full = (-H - (h * mu).sum(-1) - quad)

    # Floor only the independent-marginal proxy; the correlation part passes unfloored.
    pi = jnp.clip(jax.nn.sigmoid(h), 1e-4, 1 - 1e-4)
    kl_marg = (mu * jnp.log(mu / pi + 1e-12)
               + (1 - mu) * jnp.log((1 - mu) / (1 - pi) + 1e-12))     # (N,K) per-gate KL
    return jnp.maximum(kl_marg, free_bits).sum(-1) + kl_full - kl_marg.sum(-1)


def group_q_target(mu, Gmat, target):
    """Squared error between each class group's total marginal mass and onehot(y).
       mu:(N,K)  Gmat:(K,C)  target:(N,C) -> scalar."""
    return ((mu @ Gmat - target) ** 2).sum(-1).mean()


def reader_diversity(A, pairs):
    """Mean squared cosine between the readers A_k of paired experts.
       A:(K,r,D)  pairs: ((k1,k2), ...) -> scalar."""
    Af = A.reshape(A.shape[0], -1)
    tot = 0.0
    for k1, k2 in pairs:                             # `pairs` is static
        a1, a2 = Af[k1], Af[k2]
        cosang = (a1 @ a2) / (jnp.linalg.norm(a1) * jnp.linalg.norm(a2) + 1e-9)
        tot = tot + cosang ** 2
    return tot / len(pairs)


def moe_pressures(mu, pairs, sparse_w, loadbal_w):
    """MoE pressures: sparse = per-point co-activation within a pair,
       loadbal = squared usage difference across the dataset.  Returns the weighted sum."""
    usage = mu.mean(0)                               # (K,)
    sparse = loadbal = 0.0
    for k1, k2 in pairs:
        sparse = sparse + (mu[:, k1] * mu[:, k2]).mean()
        loadbal = loadbal + (usage[k1] - usage[k2]) ** 2
    return sparse_w * sparse / len(pairs) + loadbal_w * loadbal / len(pairs)
