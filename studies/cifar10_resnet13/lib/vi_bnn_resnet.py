"""Mean-field variational ResNet with Flipout in plain JAX: the variational baseline row.

Kernels carry (mu, rho); biases and batch-norm stay deterministic, as the paper prescribes.
The topology is imported from resnet.CONVS, so this net and the deterministic one cannot
drift apart.  The paper leaves the sigma cap unspecified, so `sigma_max` defaults to no cap.
"""
from __future__ import annotations
import sys
import time

import numpy as np
import jax
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

from paths import MODEL_DIR as _MODEL_DIR
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

# resnet.py defaults to the CIFAR-100 width, so n_class is passed explicitly wherever a
# network is built rather than switched on the module.
import resnet                                       # noqa: E402
N_CLASS_TRUNK = 10

DTYPE = jnp.float32
RHO_INIT_MEAN, RHO_INIT_STD = -3.0, 0.1      # TFP default_mean_field_normal_fn
MU_INIT_STD = 0.1                            # TFP loc initializer
PRIOR_STD = 1.0                              # TFP default prior: standard normal
KL_WEIGHT_PAPER = 0.2                        # the paper's KL weighting factor

# Every convolution plus the output dense layer carries a weight distribution.
WLAYERS = tuple(resnet.CONVS) + ("d",)


def init(seed=0):
    """Parameter dict: (mu, rho) per kernel, plus the deterministic biases, batch-norm
       scale/shift and batch-norm running statistics."""
    keys = jax.random.split(jax.random.key(seed), 2 * len(WLAYERS))
    p, i = {}, 0
    for name in WLAYERS:
        shape = _SHAPES[f"{name}_w"]
        p[f"{name}_mu"] = MU_INIT_STD * jax.random.normal(keys[i], shape, DTYPE)
        p[f"{name}_rho"] = (RHO_INIT_MEAN
                            + RHO_INIT_STD * jax.random.normal(keys[i + 1], shape, DTYPE))
        i += 2
    for name, (o, _in, _k, _st, bn) in resnet.CONVS.items():
        if bn:
            p[f"{name}_g"] = jnp.ones((o,), DTYPE)
            p[f"{name}_bt"] = jnp.zeros((o,), DTYPE)
            p[f"{name}_m"] = jnp.zeros((o,), DTYPE)
            p[f"{name}_v"] = jnp.ones((o,), DTYPE)
        else:
            p[f"{name}_b"] = jnp.zeros((o,), DTYPE)
    p["d_b"] = jnp.zeros((N_CLASS_TRUNK,), DTYPE)
    return p


def n_params():
    """(mu, rho, deterministic, total), derived from resnet.n_params() so it follows the
       trunk width.  Biases and batch-norm stay deterministic and are not doubled."""
    w, b, bn, _tot = resnet.n_params(n_class=N_CLASS_TRUNK)
    return w, w, b + bn, 2 * w + b + bn


def _sigma(rho, sigma_max=None):
    s = jax.nn.softplus(rho)
    return s if sigma_max is None else jnp.minimum(s, sigma_max)


def kl_to_prior(p, sigma_max=None):
    """Sum over all kernel weights of KL(N(mu,sigma) || N(0, PRIOR_STD)), in nats.

       Closed form, so no sampling noise enters the regulariser.  Biases and batch-norm are
       deterministic and contribute nothing."""
    tot = 0.0
    for name in WLAYERS:
        mu, s = p[f"{name}_mu"], _sigma(p[f"{name}_rho"], sigma_max)
        tot = tot + jnp.sum(jnp.log(PRIOR_STD / s)
                            + (s ** 2 + mu ** 2) / (2 * PRIOR_STD ** 2) - 0.5)
    return tot


# ----------------------------------------------------------------- layer primitives
def _signs(key, shape):
    return jnp.where(jax.random.bernoulli(key, 0.5, shape), 1.0, -1.0).astype(DTYPE)


def _conv(p, name, x, key, flipout, sigma_max):
    """Bayesian convolution, SAME padding and the stride the topology prescribes."""
    mu, sig = p[f"{name}_mu"], _sigma(p[f"{name}_rho"], sigma_max)
    st = resnet.CONVS[name][3]
    out = jax.lax.conv_general_dilated(x, mu, (st, st), "SAME")
    if key is not None:
        k1, k2, k3 = jax.random.split(key, 3)
        dW = sig * jax.random.normal(k1, mu.shape, DTYPE)
        if flipout:
            # Sign vectors over the channel axes, broadcast over space, so the shared noise
            # becomes a per-example perturbation.
            s = _signs(k2, (x.shape[0], x.shape[1], 1, 1))
            r = _signs(k3, (x.shape[0], mu.shape[0], 1, 1))
            out = out + jax.lax.conv_general_dilated(x * s, dW, (st, st), "SAME") * r
        else:
            out = out + jax.lax.conv_general_dilated(x, dW, (st, st), "SAME")
    return out


def _dense(p, name, x, key, flipout, sigma_max):
    mu, sig = p[f"{name}_mu"], _sigma(p[f"{name}_rho"], sigma_max)
    out = x @ mu.T
    if key is not None:
        k1, k2, k3 = jax.random.split(key, 3)
        dW = sig * jax.random.normal(k1, mu.shape, DTYPE)
        if flipout:
            s = _signs(k2, (x.shape[0], x.shape[1]))
            r = _signs(k3, (x.shape[0], mu.shape[0]))
            out = out + ((x * s) @ dW.T) * r
        else:
            out = out + x @ dW.T
    return out + p[f"{name}_b"]


def _stage(h, p, ks, i, main_a, main_b, short, flipout, sigma_max, train, upd):
    """One residual stage, mirroring resnet._stage: same joins, same batch-norm placement,
       none on the projection shortcut."""
    a, u = resnet._bn(_conv(p, main_a, h, ks[i], flipout, sigma_max), p, p, main_a, train)
    upd.update(u)
    a = jax.nn.relu(a)
    b, u = resnet._bn(_conv(p, main_b, a, ks[i + 1], flipout, sigma_max), p, p,
                      main_b, train)
    upd.update(u)
    if short is None:
        s = h
    else:
        s = (_conv(p, short, h, ks[i + 2], flipout, sigma_max)
             + p[f"{short}_b"][None, :, None, None])
    return jax.nn.relu(b + s)


def forward(p, x, key=None, flipout=True, train=False, sigma_max=None):
    """x:(N,3,32,32) -> logits.  key=None gives the mean network (mu only).

       In training mode returns (logits, batch-norm updates), as resnet.forward does."""
    ks = [None] * 13 if key is None else list(jax.random.split(key, 13))
    upd = {}
    h, u = resnet._bn(_conv(p, "c1", x, ks[0], flipout, sigma_max), p, p, "c1", train)
    upd.update(u)
    h = jax.nn.relu(h)
    h = _stage(h, p, ks, 1, "c2", "c3", None, flipout, sigma_max, train, upd)
    h = _stage(h, p, ks, 3, "c4", "c5", "c6", flipout, sigma_max, train, upd)
    h = _stage(h, p, ks, 6, "c7", "c8", "c9", flipout, sigma_max, train, upd)
    h = _stage(h, p, ks, 9, "c10", "c11", "c12", flipout, sigma_max, train, upd)
    feat = jnp.mean(h, axis=(2, 3))
    logits = _dense(p, "d", feat, ks[12], flipout, sigma_max)
    return (logits, upd) if train else logits


# ----------------------------------------------------------------- training
# The 10-way shape table, built once; only the dense head differs from the module default.
_SHAPES = resnet._shapes(N_CLASS_TRUNK)
_BUF = tuple(k for k in _SHAPES if k.endswith("_m") or k.endswith("_v"))


def _split(P):
    return ({k: v for k, v in P.items() if k not in _BUF},
            {k: v for k, v in P.items() if k in _BUF})


def _loss(Pt, Q, xb, yb, key, kl_weight, n_train, sigma_max):
    """CE + kl_weight * KL / N.

       The KL is divided by the training-set size to put both terms on a per-example scale;
       `kl_weight` on top is the paper's own knob."""
    lg, upd = forward({**Pt, **Q}, xb, key=key, flipout=True, train=True,
                      sigma_max=sigma_max)
    ce = -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(yb.shape[0]), yb])
    return ce + kl_weight * kl_to_prior(Pt, sigma_max) / n_train, upd


def train(X, Y, epochs=100, lr=1e-3, batch=32, seed=0, kl_weight=KL_WEIGHT_PAPER,
          Xva=None, Yva=None, verbose=True, augment=True, schedule="step",
          sigma_max=None):
    """Adam with the same learning-rate schedule the deterministic trunk uses."""
    Xd = jnp.asarray(X.reshape(-1, resnet.IN_CH, resnet.SIDE, resnet.SIDE), DTYPE)
    Yd = jnp.asarray(Y, jnp.int32)
    n_batch, n_train = X.shape[0] // batch, X.shape[0]
    P = init(seed)
    Pt, Q = _split(P)
    st = resnet._adam_init(Pt)
    grad_fn = jax.value_and_grad(_loss, has_aux=True)

    def _make_step(lr_):
        def step(carry, inp):
            Pt_, Q_, st_ = carry
            idx, k = inp
            xb, yb = Xd[idx], Yd[idx]
            ka, kw = jax.random.split(k, 2)
            if augment:
                xb = resnet._augment(xb, ka)
            (loss, upd), g = grad_fn(Pt_, Q_, xb, yb, kw, kl_weight, n_train, sigma_max)
            Pt_, st_ = resnet._adam_step(Pt_, st_, g, lr_)
            return (Pt_, {**Q_, **upd}, st_), loss
        return step

    run_epoch = jax.jit(lambda a, b, c, xs, lr_:
                        jax.lax.scan(_make_step(lr_), (a, b, c), xs))
    rng = np.random.default_rng(seed)
    kroot = jax.random.key(seed + 4242)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(n_train)[:n_batch * batch].reshape(n_batch, batch)
        ks = jax.random.split(jax.random.fold_in(kroot, ep), n_batch)
        lr_ep = jnp.asarray(resnet.lr_at(schedule, lr, ep, epochs), DTYPE)
        t0 = time.time()
        (Pt, Q, st), losses = run_epoch(Pt, Q, st, (jnp.asarray(order), ks), lr_ep)
        rec = dict(epoch=ep, loss=float(jnp.mean(losses)), secs=time.time() - t0,
                   lr=float(lr_ep),
                   kl=float(kl_to_prior(Pt, sigma_max)) / n_train)
        if Xva is not None:
            rec["val_acc"] = accuracy({**Pt, **Q}, Xva, Yva)
        hist.append(rec)
        if verbose:
            va = f"  val_acc {rec['val_acc']:.4f}" if "val_acc" in rec else ""
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {rec['loss']:.4f}  "
                  f"KL/N {rec['kl']:.4f}{va}  ({rec['secs']:.1f}s)", flush=True)
    return {**Pt, **Q}, hist


# ----------------------------------------------------------------- prediction
# Chunked over images and over posterior samples: vmapping S samples multiplies the channel
# axis by S, so peak memory goes with (image chunk) x (sample group).  Either constant moves
# the float32 logits by about 1e-4, so a table keeps both fixed.
_CHUNK = 100          # images per pass
_SAMP_CHUNK = 10      # posterior samples per pass


def sample_logits(p, X, n_samp=100, seed=11, sigma_max=None,
                  chunk=_CHUNK, samp_chunk=_SAMP_CHUNK):
    """(N, n_samp, C) float64 logits: n_samp networks drawn from the posterior.

       Plain reparameterisation, one weight sample per pass shared across inputs.  Same
       shape contract as metrics_liu.sample_logits_ebm, so one metric path scores both."""
    ks = jax.random.split(jax.random.key(seed), n_samp)

    @jax.jit
    def block(xb, kk):
        return jax.vmap(lambda k: forward(p, xb, key=k, flipout=False,
                                          sigma_max=sigma_max))(kk)          # (s,b,C)

    out = []
    for i in range(0, X.shape[0], chunk):
        xb = jnp.asarray(X[i:i + chunk].reshape(-1, resnet.IN_CH, resnet.SIDE,
                                                resnet.SIDE), DTYPE)
        parts = [np.asarray(block(xb, ks[j:j + samp_chunk])).transpose(1, 0, 2)
                 for j in range(0, n_samp, samp_chunk)]
        out.append(np.concatenate(parts, axis=1))                            # (b,S,C)
    return np.concatenate(out).astype(np.float64)


def accuracy(p, X, Y, n_samp=16, seed=11):
    pm = jax.nn.softmax(jnp.asarray(sample_logits(p, X, n_samp, seed)), -1).mean(1)
    return float((np.asarray(pm).argmax(1) == np.asarray(Y)).mean())


def sigma_stats(p, sigma_max=None):
    """Distribution of the learned sigmas (the paper's Figure 6B)."""
    s = np.concatenate([np.asarray(_sigma(p[f"{n}_rho"], sigma_max)).ravel()
                        for n in WLAYERS])
    return dict(mean=float(s.mean()), median=float(np.median(s)),
                p05=float(np.percentile(s, 5)), p95=float(np.percentile(s, 95)),
                per_layer={n: float(np.asarray(_sigma(p[f"{n}_rho"], sigma_max)).mean())
                           for n in WLAYERS})


# ================================================================= selftest
if __name__ == "__main__":
    mu, rho, det, tot = n_params()
    print(f"parameters: {mu:,} mu + {rho:,} rho + {det:,} deterministic = {tot:,}")
    print(f"  Liu et al. BNN 2.50 M -> {tot:,}   "
          f"{'= 2*w + b + bn' if tot == 2*mu + det else 'INCONSISTENT'}")
    print(f"  the DNN it must be iso-topological with: "
          f"{resnet.n_params(n_class=N_CLASS_TRUNK)[3]:,}")

    P = init(0)
    x = jnp.asarray(np.random.default_rng(0).random(
        (4, resnet.IN_CH, resnet.SIDE, resnet.SIDE)), DTYPE)
    print(f"\nmean network  : {forward(P, x).shape}   (expect (4,C))")
    print(f"one sample    : {forward(P, x, key=jax.random.key(0), flipout=False).shape}")
    print(f"flipout       : {forward(P, x, key=jax.random.key(0), flipout=True).shape}")
    print(f"KL/weight init: {float(kl_to_prior(P)) / (mu):.4f} nats")
    print(f"sigma at init : {sigma_stats(P)['mean']:.4f}  (TFP default ~0.049)")
