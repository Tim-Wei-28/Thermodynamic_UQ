"""EBM-gated adapter mixture as a reusable head on caller-supplied features and W0.
W0 stays frozen; trained are the adapters, the recognition head, the field net and the
gate couplings.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass

import numpy as np

# pipeline/model must be importable: its files import each other by bare name
from paths import MODEL_DIR as _MODEL, HERE as _H
_PIPE = os.path.dirname(os.path.dirname(_H))
for _p in (_MODEL, _PIPE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                                    # noqa: E402
from jax import config as _jax_config                         # noqa: E402
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp                                       # noqa: E402
import trainer as trn                                         # noqa: E402


@dataclass
class HeadConfig:
    """Default head recipe."""
    K: int = 10
    rank: int = 8
    family: str = "tree"        # tree | mf | tree_max_span
    topology: str = "chain"
    estimator: str = "sfe-loo"
    refit: str | None = None    # None | "once" | "periodic"
    struct_warmup: float = 0.3  # fraction of epochs before the first refit
    struct_warmup_family: str = "tree"
    epochs: int = 2400
    lr: float = 0.1
    T: int = 8                  # gate samples per step
    S: int = 12                 # Gibbs sweeps in the negative phase
    beta_max: float = 1.0
    anneal_frac: float = 0.4
    free_bits: float = 0.02
    wd: float = 0.01
    clip: float = 1.0
    j_init: float = -1.5        # negative: gates inhibit each other
    field_n_hidden: int = 24
    field_b2: float = 0.5
    c_bias: float = 0.5
    head_init: float = 0.1
    gamma0: float = 1.0         # q-target warmup strength
    warm_frac: float = 0.15     # fraction of the run over which it fades out
    label_dropout: float = 0.0
    w0_scale: float = 0.1       # only used when W0 is drawn at random
    adapter_init: float = 0.1
    seed: int = 0


def prepare_features(Xtr, Xother, W0=None, b0=None, centre=True, rescale=True):
    """Return (Xtr_out, [Xother_out...], W0_out) ready for the trainer.

    Without W0 the inputs are only mean-centred. With W0 the features are rescaled by
    one scalar that is divided back out of W0, so the logits are unchanged, and a
    constant column plus b0 as the last W0 column carry the bias. No centring there,
    because W0 was trained on the uncentred activations."""
    Xs = [np.asarray(x, np.float64) for x in ([Xtr] + list(Xother))]
    if W0 is None:
        if centre:
            mean = Xs[0].mean(0)
            Xs = [x - mean for x in Xs]
        return Xs[0], Xs[1:], None
    W0 = np.asarray(W0, np.float64)
    s = 1.0
    if rescale:
        s = 1.0 / (Xs[0].std() + 1e-12)
    ones = lambda n: np.ones((n, 1), np.float64)
    Xs = [np.concatenate([x * s, ones(x.shape[0])], 1) for x in Xs]   # constant column not scaled
    b0 = np.zeros(W0.shape[0]) if b0 is None else np.asarray(b0, np.float64)
    W0 = np.concatenate([W0 / s, b0[:, None]], 1)
    return Xs[0], Xs[1:], W0


def train_head(X, Y, C, W0=None, cfg=None, verbose=True, refit_log=None, trace_out=None):
    """Train the head on (X, Y). Returns (params, spec, forward).

    W0 is the frozen (C, D) base map, or None to draw a weak random one.
    refit_log and trace_out are optional lists that collect refit entries and per-epoch
    scalars."""
    cfg = cfg or HeadConfig()
    if cfg.family == "tree_max_span" and not cfg.refit:
        raise ValueError(
            "family='tree_max_span' needs refit='once' or 'periodic'; with refit=None the "
            "run is bit-identical to family='tree' on the warm-up topology")
    if cfg.refit and cfg.family not in ("tree", "tree_max_span"):
        raise ValueError(f"refit={cfg.refit!r} has no meaning for family={cfg.family!r}: "
                         "mean-field has no edges to refit")
    X = jnp.asarray(np.asarray(X, np.float64))
    Y = jnp.asarray(np.asarray(Y), dtype=jnp.int32)
    N, D = X.shape

    def _data(kd):
        # fixed data; the key is part of the trainer's contract and ignored here
        return X, Y

    def _lik(key):
        kB, kA = jax.random.split(key, 2)
        B = cfg.adapter_init * jax.random.normal(kB, (cfg.K, C, cfg.rank))
        A = cfg.adapter_init * jax.random.normal(kA, (cfg.K, cfg.rank, D))
        if W0 is None:
            kW = jax.random.fold_in(key, 12345)
            W0_ = cfg.w0_scale * jax.random.normal(kW, (C, D))
        else:
            W0_ = jnp.asarray(W0)
        return dict(W0=W0_, B=B, A=A)

    if verbose:
        kind = "weak random (frozen)" if W0 is None else "trained layer (frozen)"
        print(f"  EBM head: N={N} D={D} C={C} K={cfg.K} rank={cfg.rank} "
              f"family={cfg.family} est={cfg.estimator}")
        print(f"            W0 = {kind}, epochs={cfg.epochs} lr={cfg.lr}")
    return trn.train(trn.TrainConfig(
        K=cfg.K, C=C, D=D, r=cfg.rank, seed=cfg.seed,
        make_data=_data, build_lik=_lik,
        family=cfg.family, tree_topology=cfg.topology,
        estimator=cfg.estimator, n_step_keys=3,
        refit=cfg.refit, struct_warmup=cfg.struct_warmup,
        struct_warmup_family=cfg.struct_warmup_family, refit_log=refit_log,
        trace_out=trace_out,
        T=cfg.T, tau=1.0,
        head_init=cfg.head_init, c_bias=cfg.c_bias,
        field_kind="mlp", field_n_hidden=cfg.field_n_hidden, field_b2=cfg.field_b2,
        j_init=cfg.j_init,
        gamma0=cfg.gamma0, warm_frac=cfg.warm_frac,
        label_dropout_p=cfg.label_dropout,
        epochs=cfg.epochs, lr=cfg.lr, beta_max=cfg.beta_max,
        anneal_frac=cfg.anneal_frac, warmup=0, free_bits=cfg.free_bits,
        clip=cfg.clip, wd=cfg.wd, S=cfg.S))


def n_trained(params, spec, D, C, cfg=None):
    """Trainable-parameter count of the head. W0 is frozen and not counted; J counts
    its free entries only (symmetric, zero diagonal)."""
    cfg = cfg or HeadConfig()
    K, r, H = cfg.K, cfg.rank, cfg.field_n_hidden
    n_pre = getattr(spec, "n_pre", K)
    parts = {"A": K * r * D, "B": K * C * r, "U": n_pre * (D + C), "c": n_pre,
             "field_W1": H * D, "field_b1": H, "field_W2": K * H, "field_b2": K,
             "J": K * (K - 1) // 2}
    parts["total"] = sum(parts.values())
    return parts
