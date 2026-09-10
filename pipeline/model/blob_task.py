"""
Blob task: three overlapping 2-D Gaussians on an equilateral triangle (C=3, D=2),
plus a far-field ring as an unlabelled OOD probe.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

C = 3                                        # classes
D = 2                                        # input dims
R_C = 2.5                                    # triangle radius
SIGMA = 1.1                                  # blob std
_ANG = jnp.array([90.0, 210.0, 330.0]) * jnp.pi / 180.0
CENTERS = R_C * jnp.stack([jnp.cos(_ANG), jnp.sin(_ANG)], 1)   # (3,2)


def configure(R_c=None, sigma=None):
    """Re-derive the import-time constants; CENTERS depends on R_C, so setting
       R_C directly would not take effect."""
    global R_C, SIGMA, CENTERS
    if R_c is not None:
        R_C = float(R_c)
    if sigma is not None:
        SIGMA = float(sigma)
    CENTERS = R_C * jnp.stack([jnp.cos(_ANG), jnp.sin(_ANG)], 1)


def make_data(n_per_class, key, group="train", sigma=None, ood_radius=7.0):
    """group='train'/'test': n_per_class points per class ~ N(centre_c, sigma^2 I).
       group='ood': a far-field ring at ood_radius, no labels.
       sigma=None reads the module SIGMA at call time."""
    sigma = SIGMA if sigma is None else sigma
    if group in ("train", "test"):
        ks = jax.random.split(key, C)
        Xs = [CENTERS[c] + sigma * jax.random.normal(ks[c], (n_per_class, D)) for c in range(C)]
        Ys = [jnp.full((n_per_class,), c, jnp.int32) for c in range(C)]
        return jnp.concatenate(Xs, 0), jnp.concatenate(Ys, 0)
    if group == "ood":
        n = n_per_class * C
        ka, kr = jax.random.split(key)
        ang = jax.random.uniform(ka, (n,), minval=0.0, maxval=2 * jnp.pi)
        rad = ood_radius + 0.4 * jax.random.normal(kr, (n,))
        X = jnp.stack([rad * jnp.cos(ang), rad * jnp.sin(ang)], 1)
        return X, None
    raise ValueError(group)


def bayes_accuracy(n=6000, sigma=None, seed=0):
    """Bayes accuracy for equal-prior isotropic Gaussians (nearest-centre rule)."""
    X, Y = make_data(n, jax.random.key(seed), group="test", sigma=sigma)
    d2 = ((X[:, None, :] - CENTERS[None]) ** 2).sum(-1)          # (N,3)
    return float((d2.argmin(1) == Y).mean())
