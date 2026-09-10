"""
Builders for the likelihood parameters (W0, B, A) of W(z) = W0 + sum_k z_k B_k A_k
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp


def build_class_handle_blocked(key, gain, K, C, D, r, w0_scale,
                               handle_free_scale=0.1, reader_scale=0.3):
    """Weak random frozen W0 plus blocked class-anchored adapters:
       B[:,:,0] is the class handle (gain * e_group), the other ranks and A are free."""
    kW, kB, kA = jax.random.split(key, 3)                   # fixed split order = reproducible
    W0 = w0_scale * jax.random.normal(kW, (C, D))
    grp = jnp.arange(K) // (K // C)                        # contiguous groups: gate k -> class
    B = jnp.zeros((K, C, r))
    B = B.at[:, :, 0].set(gain * jax.nn.one_hot(grp, C))
    B = B.at[:, :, 1].set(handle_free_scale * jax.random.normal(kB, (K, C)))
    A = reader_scale * jax.random.normal(kA, (K, r, D))
    Gmat = jax.nn.one_hot(grp, C)                          # (K,C) gate -> class-group
    # grp / Gmat / gsize are needed by the anchors and the q-target.
    return dict(W0=W0, B=B, A=A, grp=np.asarray(grp), Gmat=Gmat,
                gsize=Gmat.sum(0), g=gain)


def build_random_free(key, K, C, D, rank, w0_scale, adapter_init):
    """Weak random frozen W0 plus fully free adapters; no anchor structure."""
    kW, kB, kA = jax.random.split(key, 3)
    W0 = w0_scale * jax.random.normal(kW, (C, D))
    B = adapter_init * jax.random.normal(kB, (K, C, rank))
    A = adapter_init * jax.random.normal(kA, (K, rank, D))
    return dict(W0=W0, B=B, A=A)


REGISTRY = {
    "class-handle-blocked": build_class_handle_blocked,
    "random-free": build_random_free,
}
