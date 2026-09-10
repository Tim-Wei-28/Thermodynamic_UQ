"""
Swappable field net h(x) of the EBM prior p(z|x) ~ exp(h(x).z + .5 z J z).
A field is a pair (kind, params dict); every evaluation goes through `apply`.
Kinds: linear, equivariant, equivariant_nl, routable_mlp, routable_sum, mlp.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

WEIGHT_LEAVES = frozenset({"V", "a", "W1", "W2"})     # leaves that get weight decay


def init(kind, key, K, D, n_hidden=8, v_scale=0.1):
    """Return the params dict for a field of the given kind."""
    if kind in ("linear", "equivariant"):
        V = v_scale * jax.random.normal(key, (K, D))
        d = jnp.zeros((K,))
        if kind == "equivariant":
            V, d = project_equivariant(V, d)
        return dict(V=V, d=d)
    if kind == "equivariant_nl":
        ks = jax.random.split(key, 2)
        # shared slope `a` per gate plus an MLP on the symmetric pool sum_k x_k
        return dict(a=jnp.zeros(()),
                    W1=0.5 * jax.random.normal(ks[0], (n_hidden, 1)), b1=jnp.zeros((n_hidden,)),
                    W2=0.1 * jax.random.normal(ks[1], (1, n_hidden)), b2=jnp.zeros((1,)))
    if kind == "routable_mlp":
        ks = jax.random.split(key, 2)
        # shared trunk on the mirror-symmetric features [u+v, (u-v)^2], one head row per gate
        return dict(W1=0.8 * jax.random.normal(ks[0], (n_hidden, 2)), b1=jnp.zeros((n_hidden,)),
                    W2=0.5 * jax.random.normal(ks[1], (K, n_hidden)), b2=jnp.full((K,), 0.3))
    if kind == "routable_sum":
        # ablation of routable_mlp: trunk sees only u+v
        ks = jax.random.split(key, 2)
        return dict(W1=0.8 * jax.random.normal(ks[0], (n_hidden, 1)), b1=jnp.zeros((n_hidden,)),
                    W2=0.5 * jax.random.normal(ks[1], (K, n_hidden)), b2=jnp.full((K,), 0.3))
    if kind == "mlp":
        # general one-hidden-layer MLP on the raw input; positive output bias so gates start usable
        ks = jax.random.split(key, 2)
        return dict(W1=0.5 * jax.random.normal(ks[0], (n_hidden, D)), b1=jnp.zeros((n_hidden,)),
                    W2=0.3 * jax.random.normal(ks[1], (K, n_hidden)), b2=jnp.full((K,), 0.3))
    raise ValueError(f"unknown field kind {kind!r}")


def apply(kind, fp, X):
    """h(x) for a batch X:(N,D) -> (N,K)."""
    if kind in ("linear", "equivariant"):
        return X @ fp["V"].T + fp["d"]
    if kind == "equivariant_nl":
        s = X.sum(-1, keepdims=True)                              # symmetric pool (N,1)
        g = jnp.tanh(s @ fp["W1"].T + fp["b1"]) @ fp["W2"].T + fp["b2"]   # (N,1)
        return fp["a"] * X + g                                    # (N,K), equivariant
    if kind == "routable_mlp":
        s = X.sum(-1, keepdims=True)                              # u+v      (N,1)
        d2 = ((X[:, 0] - X[:, 1]) ** 2)[:, None]                  # (u-v)^2  (N,1)
        feat = jnp.concatenate([s, d2], -1)                      # (N,2), swap-invariant
        hdn = jnp.tanh(feat @ fp["W1"].T + fp["b1"])             # (N,H) shared trunk
        return hdn @ fp["W2"].T + fp["b2"]                       # (N,K) independent heads
    if kind == "routable_sum":
        s = X.sum(-1, keepdims=True)                             # u+v      (N,1)
        hdn = jnp.tanh(s @ fp["W1"].T + fp["b1"])                # (N,H) shared trunk
        return hdn @ fp["W2"].T + fp["b2"]                       # (N,K) independent heads
    if kind == "mlp":
        hdn = jnp.tanh(X @ fp["W1"].T + fp["b1"])                # (N,H)
        return hdn @ fp["W2"].T + fp["b2"]                       # (N,K)
    raise ValueError(f"unknown field kind {kind!r}")


def project_equivariant(V, d):
    """Project (V,d) onto the u<->v equivariant subspace V=[[a,b],[b,a]], d=[m,m] (K=D=2)."""
    a = 0.5 * (V[0, 0] + V[1, 1])
    b = 0.5 * (V[0, 1] + V[1, 0])
    m = 0.5 * (d[0] + d[1])
    return jnp.array([[a, b], [b, a]]), jnp.array([m, m])


def maybe_project(kind, fp):
    """Re-project after a gradient step; no-op unless kind == 'equivariant'."""
    if kind == "equivariant":
        V, d = project_equivariant(fp["V"], fp["d"])
        return dict(V=V, d=d)
    return fp
