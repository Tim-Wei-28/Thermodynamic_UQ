"""
Adapter-anchor pull terms.  Each function returns the residual (current - target);
the trainer scales it by wd_anchor and subtracts it inside the SGD update.
"""
from __future__ import annotations
import jax.numpy as jnp


def pull_pergate(P, P_init):
    """Pull every entry back toward its init value (works for B and A)."""
    return P - P_init


def pull_group_mean_rank0(B, Gmat, gsize, Btgt):
    """Pull each class group's mean rank-0 handle toward Btgt; other ranks stay free.
       B:(K,C,r)  Gmat:(K,C) one-hot gate -> group."""
    gmean = (Gmat.T @ B[:, :, 0]) / gsize[:, None]         # (C_grp, C_cls) per-group mean
    return jnp.zeros_like(B).at[:, :, 0].set(Gmat @ (gmean - Btgt))


def pull_group_mean_bcast(B, Gmat, gsize, Btgt):
    """Group-mean anchor broadcast over all ranks; equals the rank-0 variant for r=1."""
    gmean = (Gmat.T @ B[:, :, 0]) / gsize[:, None]
    return (Gmat @ (gmean - Btgt))[:, :, None]


def pull_group_sum_rank0(B, Gmat, Bsum_tgt):
    """Weakest anchor: only the group sum of the rank-0 handles is pinned."""
    Bsum = Gmat.T @ B[:, :, 0]                             # (C_grp, C_cls) group sum
    return jnp.zeros_like(B).at[:, :, 0].set(Gmat @ (Bsum - Bsum_tgt))
