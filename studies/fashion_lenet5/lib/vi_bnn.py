"""Bayesian LeNet-5 baseline in plain JAX: mean-field Gaussian weights, deterministic
biases, Flipout during training and plain reparameterisation at test time.
Follows Liu et al. (2022): N(0,1) prior, softplus scale, Adam 20 epochs at 1e-3,
100 posterior samples per prediction, 123 176 parameters.  Run directly for a smoke test.
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import jax
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

# lenet.py lives in pipeline/model/ and is imported at module load.
from paths import MODEL_DIR as _MODEL_DIR
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

import lenet                                                   # noqa: E402

DTYPE = jnp.float32
# Posterior init and prior follow the TFP defaults used by the reference implementation.
RHO_INIT_MEAN, RHO_INIT_STD = -3.0, 0.1
MU_INIT_STD = 0.1
PRIOR_STD = 1.0

LAYERS = ("c1", "c3", "c5", "f6", "out")


def init(seed=0):
    """Parameter dict: per layer a mean, an untransformed scale, and a deterministic bias."""
    keys = jax.random.split(jax.random.key(seed), 2 * len(LAYERS))
    p, i = {}, 0
    for name in LAYERS:
        shape = lenet.SHAPES[f"{name}_w"]
        p[f"{name}_mu"] = MU_INIT_STD * jax.random.normal(keys[i], shape, DTYPE)
        p[f"{name}_rho"] = (RHO_INIT_MEAN
                            + RHO_INIT_STD * jax.random.normal(keys[i + 1], shape, DTYPE))
        p[f"{name}_b"] = jnp.zeros(lenet.SHAPES[f"{name}_b"], DTYPE)
        i += 2
    return p


def n_params():
    """(mu, rho, bias, total); the total is the Liu et al. 123.2 K count."""
    w = sum(int(np.prod(lenet.SHAPES[f"{n}_w"])) for n in LAYERS)
    b = sum(int(np.prod(lenet.SHAPES[f"{n}_b"])) for n in LAYERS)
    return w, w, b, 2 * w + b


def _sigma(rho):
    return jax.nn.softplus(rho)


def kl_to_prior(p):
    """Closed-form KL(N(mu,sigma) || N(0, PRIOR_STD)) summed over all weights, in nats.
       Biases are deterministic and contribute nothing."""
    tot = 0.0
    for name in LAYERS:
        mu, s = p[f"{name}_mu"], _sigma(p[f"{name}_rho"])
        tot = tot + jnp.sum(jnp.log(PRIOR_STD / s)
                            + (s ** 2 + mu ** 2) / (2 * PRIOR_STD ** 2) - 0.5)
    return tot


# ----------------------------------------------------------------- layer primitives
# Flipout: the shared weight noise is multiplied by independent input/output sign vectors
# s, r ~ Uniform{-1,+1}, which decorrelates the perturbation across the batch.
def _signs(key, shape):
    return jnp.where(jax.random.bernoulli(key, 0.5, shape), 1.0, -1.0).astype(DTYPE)


def _dense(p, name, x, key, flipout):
    mu, sig = p[f"{name}_mu"], _sigma(p[f"{name}_rho"])
    out = x @ mu.T
    if key is not None:
        k1, k2, k3 = jax.random.split(key, 3)
        eps = jax.random.normal(k1, mu.shape, DTYPE)
        dW = sig * eps
        if flipout:
            s = _signs(k2, (x.shape[0], x.shape[1]))
            r = _signs(k3, (x.shape[0], mu.shape[0]))
            out = out + ((x * s) @ dW.T) * r
        else:
            out = out + x @ dW.T           # one weight sample, shared across the batch
    return out + p[f"{name}_b"]


def _conv(p, name, x, key, flipout):
    mu, sig = p[f"{name}_mu"], _sigma(p[f"{name}_rho"])
    out = jax.lax.conv_general_dilated(x, mu, (1, 1), "VALID")
    if key is not None:
        k1, k2, k3 = jax.random.split(key, 3)
        dW = sig * jax.random.normal(k1, mu.shape, DTYPE)
        if flipout:
            s = _signs(k2, (x.shape[0], x.shape[1], 1, 1))
            r = _signs(k3, (x.shape[0], mu.shape[0], 1, 1))
            out = out + jax.lax.conv_general_dilated(x * s, dW, (1, 1), "VALID") * r
        else:
            out = out + jax.lax.conv_general_dilated(x, dW, (1, 1), "VALID")
    return out + p[f"{name}_b"][None, :, None, None]


def forward(p, x, key=None, flipout=True):
    """x:(N,1,28,28) -> logits (N,10).  key=None gives the mean network (mu only).
       Shapes come from lenet.SHAPES, so the topology matches the deterministic baseline."""
    ks = [None] * 5 if key is None else list(jax.random.split(key, 5))
    x = jnp.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)))                  # -> 32x32
    h = lenet._pool2(jax.nn.relu(_conv(p, "c1", x, ks[0], flipout)))
    h = lenet._pool2(jax.nn.relu(_conv(p, "c3", h, ks[1], flipout)))
    h = h.reshape(h.shape[0], -1)
    h = jax.nn.relu(_dense(p, "c5", h, ks[2], flipout))
    h = jax.nn.relu(_dense(p, "f6", h, ks[3], flipout))
    return _dense(p, "out", h, ks[4], flipout)


# ----------------------------------------------------------------- training
def _loss(p, xb, yb, key, kl_weight, n_train):
    """CE + kl_weight * KL / N.  Dividing by the training-set size puts both terms on a
       per-example scale; kl_weight on top is the accuracy-versus-ECE knob."""
    lg = forward(p, xb, key=key, flipout=True)
    ce = -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(yb.shape[0]), yb])
    return ce + kl_weight * kl_to_prior(p) / n_train


def train(X, Y, epochs=20, lr=1e-3, batch=32, seed=0, kl_weight=1.0,
          Xva=None, Yva=None, verbose=True):
    """Train with Adam, same settings as the deterministic baseline."""
    Xd = jnp.asarray(X.reshape(-1, 1, 28, 28), DTYPE)
    Yd = jnp.asarray(Y, jnp.int32)
    n_batch, n_train = X.shape[0] // batch, X.shape[0]
    p = init(seed)
    st = lenet._adam_init(p)
    grad_fn = jax.value_and_grad(_loss)

    def step(carry, inp):
        p_, st_ = carry
        idx, k = inp
        loss, g = grad_fn(p_, Xd[idx], Yd[idx], k, kl_weight, n_train)
        p_, st_ = lenet._adam_step(p_, st_, g, lr)
        return (p_, st_), loss

    run_epoch = jax.jit(lambda p_, st_, xs: jax.lax.scan(step, (p_, st_), xs))
    rng = np.random.default_rng(seed)
    kroot = jax.random.key(seed + 4242)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(n_train)[:n_batch * batch].reshape(n_batch, batch)
        ks = jax.random.split(jax.random.fold_in(kroot, ep), n_batch)
        t0 = time.time()
        (p, st), losses = run_epoch(p, st, (jnp.asarray(order), ks))
        rec = dict(epoch=ep, loss=float(jnp.mean(losses)), secs=time.time() - t0,
                   kl=float(kl_to_prior(p)) / n_train)
        if Xva is not None:
            rec["val_acc"] = accuracy(p, Xva, Yva)
        hist.append(rec)
        if verbose:
            va = f"  val_acc {rec['val_acc']:.4f}" if "val_acc" in rec else ""
            print(f"  epoch {ep + 1:2d}/{epochs}  loss {rec['loss']:.4f}  "
                  f"KL/N {rec['kl']:.4f}{va}  ({rec['secs']:.1f}s)")
    return p, hist


# ----------------------------------------------------------------- prediction
def sample_logits(p, X, n_samp=100, seed=11, chunk=2000):
    """(N, n_samp, C) float64 logits: n_samp networks drawn from the posterior.
       Plain reparameterisation, one weight sample per pass shared across inputs.
       Same shape contract as metrics_liu.sample_logits_ebm."""
    ks = jax.random.split(jax.random.key(seed), n_samp)

    @jax.jit
    def block(xb):
        return jax.vmap(lambda k: forward(p, xb, key=k, flipout=False))(ks)  # (S,b,C)

    out = []
    for i in range(0, X.shape[0], chunk):
        xb = jnp.asarray(X[i:i + chunk].reshape(-1, 1, 28, 28), DTYPE)
        out.append(np.asarray(block(xb)).transpose(1, 0, 2))
    return np.concatenate(out).astype(np.float64)


def accuracy(p, X, Y, n_samp=16, seed=11):
    pm = jax.nn.softmax(jnp.asarray(sample_logits(p, X, n_samp, seed)), -1).mean(1)
    return float((np.asarray(pm).argmax(1) == np.asarray(Y)).mean())


def sigma_stats(p):
    """Summary of the learned posterior sigmas, overall and per layer."""
    s = np.concatenate([np.asarray(_sigma(p[f"{n}_rho"])).ravel() for n in LAYERS])
    return dict(mean=float(s.mean()), median=float(np.median(s)),
                p05=float(np.percentile(s, 5)), p95=float(np.percentile(s, 95)),
                per_layer={n: float(np.asarray(_sigma(p[f"{n}_rho"])).mean())
                           for n in LAYERS})


if __name__ == "__main__":
    import fashion_data as fd
    mu, rho, b, tot = n_params()
    print(f"parameters: {mu:,} mu + {rho:,} rho + {b:,} deterministic biases = {tot:,}")
    print(f"  Liu et al. BNN 123.2 K -> {tot:,}   "
          f"{'MATCH' if tot == 123176 else 'MISMATCH'}")
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = fd.splits()
    print("\n2-epoch smoke test on 5 000 images:")
    p, _ = train(Xtr[:5000], Ytr[:5000], epochs=2, kl_weight=1.0,
                 Xva=Xva[:1000], Yva=Yva[:1000])
    print("sigma:", {k: round(v, 4) for k, v in sigma_stats(p).items()
                     if k != "per_layer"})
    print("logits:", sample_logits(p, Xte[:8], n_samp=5).shape)
