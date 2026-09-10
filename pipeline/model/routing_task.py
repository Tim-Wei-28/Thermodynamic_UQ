"""
Routing task: x = [onehot(context) * CTX, content], D = R + DC; the context picks
one of R fixed conflicting linear rules and the label is that rule's argmax on the
content.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

R = 3                                          # rules
C = 3                                          # classes
DC = 6                                         # content dims
D = R + DC                                     # input dims
CTX = 3.0                                      # default context strength
SIGMA_C = 1.2                                  # content std

# fixed key: the rules are the same in every run
RULES = jax.random.normal(jax.random.key(0), (R, C, DC))


def configure(R_=None, DC_=None, sigma_c=None):
    """Re-derive the import-time constants.  Changing R or DC also changes D and
       RULES, so callers that copied those must re-sync."""
    global R, DC, D, SIGMA_C, RULES
    if R_ is not None:
        R = int(R_)
    if DC_ is not None:
        DC = int(DC_)
    if sigma_c is not None:
        SIGMA_C = float(sigma_c)
    D = R + DC
    RULES = jax.random.normal(jax.random.key(0), (R, C, DC))


def make_data(n, key, ctx=None, sigma_c=None):
    """(X, Y, r): r ~ Uniform(rules); content ~ N(0, sigma_c^2);
       y = argmax(M_r . content); X = [onehot(r)*ctx, content].
       r is returned for analysis only; ctx/sigma_c=None read the module values."""
    ctx = CTX if ctx is None else ctx
    sigma_c = SIGMA_C if sigma_c is None else sigma_c
    kr, kc = jax.random.split(key)
    r = jax.random.randint(kr, (n,), 0, R)
    content = sigma_c * jax.random.normal(kc, (n, DC))
    logit = jnp.einsum("ncd,nd->nc", RULES[r], content)
    y = logit.argmax(1)
    X = jnp.concatenate([jax.nn.one_hot(r, R) * ctx, content], 1)   # (n, D)
    return X, y.astype(jnp.int32), r.astype(jnp.int32)


def chance(n=6000, seed=0):
    """Majority-class accuracy on the label distribution."""
    _, Y, _ = make_data(n, jax.random.key(seed))
    counts = jnp.bincount(Y, length=C)
    return float(counts.max() / n)
