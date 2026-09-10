"""The EBM-gated adapter mixture as a reusable head, in CIFAR-10 geometry.

Approach 1 takes raw mean-centred pixels with a weak random frozen W0; approach 3 takes the
ResNet's 256 pooled activations plus a constant with the trained Dense-10 layer as W0.  The
head configuration is held at the Fashion-MNIST baseline, so only D and W0's width change.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

# pipeline/model must be importable: its files import each other by bare name.
from paths import MODEL_DIR as _MODEL, HERE as _H
_PIPE = os.path.dirname(os.path.dirname(_H))          # .../pipeline
for _p in (_MODEL, _PIPE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                                    # noqa: E402
from jax import config as _jax_config                         # noqa: E402
_jax_config.update("jax_enable_x64", True)                    # float64, as in the engine
import jax.numpy as jnp                                       # noqa: E402
import trainer as trn                                         # noqa: E402


@dataclass
class HeadConfig:
    """The recipe, held at the Fashion-MNIST baseline so the tuning is not a variable."""
    K: int = 10                 # = C, so the q-target round-robin is a bijection
    rank: int = 8               # adapter rank
    family: str = "tree"        # tree | mf | tree_max_span
    topology: str = "chain"     # chain | binary | random | star
    estimator: str = "sfe-loo"  # set explicitly, never left to the default
    epochs: int = 2400          # (epochs, lr) trade off: (2400, 0.1) as (4800, 0.05)
    lr: float = 0.1
    T: int = 8                  # gate samples per step
    S: int = 12                 # Gibbs sweeps in the negative phase
    beta_max: float = 1.0
    anneal_frac: float = 0.4
    free_bits: float = 0.02
    wd: float = 0.01
    wd_field: Optional[float] = None   # None couples it to wd; set it to keep the field
    #   alive while the adapter decay is raised.
    clip: float = 1.0
    j_init: float = -1.5        # gates inhibit each other -> few-on patterns
    field_n_hidden: int = 24
    field_b2: float = 0.5       # gates-on bootstrap
    c_bias: float = 0.5
    head_init: float = 0.1
    gamma0: float = 1.0         # q-target warmup strength ...
    warm_frac: float = 0.15     # ... faded out over the first 15 % of the run
    w0_scale: float = 0.1       # only used when W0 is drawn at random (approach 1)
    adapter_init: float = 0.1
    seed: int = 0


def prepare_features(Xtr, Xother, W0=None, b0=None, centre=True, rescale=True):
    """Return (Xtr_out, [Xother_out...], W0_out) ready for the trainer.

       Without W0 the inputs are mean-centred on the training split; with a frozen W0 an
       uncentred input would add a constant to every logit that no parameter can remove.

       With W0 the features are rescaled and W0 divided by the same scalar, so the logits do
       not move, and a constant 1.0 column is appended whose weight is W0's last column, the
       layer bias.  Centring stays off here because W0 was trained on uncentred activations."""
    Xs = [np.asarray(x, np.float64) for x in ([Xtr] + list(Xother))]
    if W0 is None:
        if centre:
            mean = Xs[0].mean(0)
            Xs = [x - mean for x in Xs]
        return Xs[0], Xs[1:], None
    W0 = np.asarray(W0, np.float64)
    s = 1.0
    if rescale:
        # One scalar for all dims; a per-dim scaling could not be absorbed by W0 without
        # changing what the adapters see per dim.
        s = 1.0 / (Xs[0].std() + 1e-12)
    ones = lambda n: np.ones((n, 1), np.float64)
    Xs = [np.concatenate([x * s, ones(x.shape[0])], 1) for x in Xs]
    b0 = np.zeros(W0.shape[0]) if b0 is None else np.asarray(b0, np.float64)
    W0 = np.concatenate([W0 / s, b0[:, None]], 1)          # (C, D+1), bias as last column
    return Xs[0], Xs[1:], W0


def train_head(X, Y, C, W0=None, cfg=None, verbose=True):
    """Train the EBM-gated adapter mixture on (X, Y).  Returns (params, spec, forward).

       X: (N, D) float64 features, Y: (N,) int labels, C: number of classes.
       W0: (C, D) frozen base map, or None to draw the pipeline's weak random one."""
    cfg = cfg or HeadConfig()
    X = jnp.asarray(np.asarray(X, np.float64))
    Y = jnp.asarray(np.asarray(Y), dtype=jnp.int32)
    N, D = X.shape

    def _data(kd):
        # The trainer owns the RNG, but this data is fixed, so the key is ignored.
        return X, Y

    def _lik(key):
        kB, kA = jax.random.split(key, 2)
        # Adapters start random and free, so all class information travels through the gates.
        B = cfg.adapter_init * jax.random.normal(kB, (cfg.K, C, cfg.rank))
        A = cfg.adapter_init * jax.random.normal(kA, (cfg.K, cfg.rank, D))
        if W0 is None:
            kW = jax.random.fold_in(key, 12345)               # own key, weak random base
            W0_ = cfg.w0_scale * jax.random.normal(kW, (C, D))
        else:
            W0_ = jnp.asarray(W0)
        return dict(W0=W0_, B=B, A=A)

    if verbose:
        kind = "weak random (frozen)" if W0 is None else "trained layer (frozen)"
        print(f"  EBM head: N={N} D={D} C={C} K={cfg.K} rank={cfg.rank} "
              f"family={cfg.family} est={cfg.estimator}", flush=True)
        print(f"            W0 = {kind}, epochs={cfg.epochs} lr={cfg.lr}", flush=True)
    return trn.train(trn.TrainConfig(
        K=cfg.K, C=C, D=D, r=cfg.rank, seed=cfg.seed,
        make_data=_data, build_lik=_lik,
        family=cfg.family, tree_topology=cfg.topology,
        estimator=cfg.estimator, n_step_keys=3,
        T=cfg.T, tau=1.0,
        head_init=cfg.head_init, c_bias=cfg.c_bias,
        field_kind="mlp", field_n_hidden=cfg.field_n_hidden, field_b2=cfg.field_b2,
        j_init=cfg.j_init,
        gamma0=cfg.gamma0, warm_frac=cfg.warm_frac,
        epochs=cfg.epochs, lr=cfg.lr, beta_max=cfg.beta_max,
        anneal_frac=cfg.anneal_frac, warmup=0, free_bits=cfg.free_bits,
        clip=cfg.clip, wd=cfg.wd, wd_field=cfg.wd_field, S=cfg.S))


def n_trained(spec, D, C, cfg=None):
    """Trainable-parameter count of the head, for the comparison table.

       W0 is frozen and not counted; J counts its free entries only (symmetric, hollow)."""
    cfg = cfg or HeadConfig()
    K, r, H = cfg.K, cfg.rank, cfg.field_n_hidden
    n_pre = getattr(spec, "n_pre", K)
    parts = {"A": K * r * D, "B": K * C * r, "U": n_pre * (D + C), "c": n_pre,
             "field_W1": H * D, "field_b1": H, "field_W2": K * H, "field_b2": K,
             "J": K * (K - 1) // 2}
    parts["total"] = sum(parts.values())
    return parts
