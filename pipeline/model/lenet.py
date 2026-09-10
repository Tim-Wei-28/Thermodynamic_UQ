"""
LeNet-5 in plain JAX after Liu et al. 
"""
from __future__ import annotations
import time

import numpy as np
import jax
from jax import config as _jax_config
# float64 process-wide, as in engine.py; the convnet itself pins float32 explicitly
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

DTYPE = jnp.float32

N_FEAT = 84                 # width of the F6 layer = the feature dim the head sees
N_CLASS = 10


def shapes(width=1.0):
    """Layer shapes for a width-scaled variant.  F6 stays at 84 so the feature dim
       downstream is constant; width=1.0 is the paper architecture."""
    c1 = max(1, int(round(6 * width)))
    c3 = max(1, int(round(16 * width)))
    c5 = max(1, int(round(120 * width)))
    return {
        "c1_w": (c1, 1, 5, 5),      "c1_b": (c1,),
        "c3_w": (c3, c1, 5, 5),     "c3_b": (c3,),
        "c5_w": (c5, c3 * 25),      "c5_b": (c5,),
        "f6_w": (N_FEAT, c5),       "f6_b": (N_FEAT,),
        "out_w": (N_CLASS, N_FEAT), "out_b": (N_CLASS,),
    }


SHAPES = shapes()


# ================================================================= 1. init
def _glorot(key, shape):
    """Glorot uniform; conv fans count the whole receptive field."""
    if len(shape) == 4:
        o, i, kh, kw = shape
        fan_in, fan_out = i * kh * kw, o * kh * kw
    else:
        o, i = shape
        fan_in, fan_out = i, o
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, DTYPE, -lim, lim)


def init(seed=0, width=1.0):
    """Parameter dict: Glorot-uniform kernels, zero biases."""
    sh = shapes(width)
    keys = jax.random.split(jax.random.key(seed), 5)
    p, i = {}, 0
    for name in ("c1", "c3", "c5", "f6", "out"):
        p[f"{name}_w"] = _glorot(keys[i], sh[f"{name}_w"])
        p[f"{name}_b"] = jnp.zeros(sh[f"{name}_b"], DTYPE)
        i += 1
    return p


def n_params(p=None, width=1.0):
    """(weights, biases, total)."""
    sh = shapes(width)
    w = sum(int(np.prod(s)) for k, s in sh.items() if k.endswith("_w"))
    b = sum(int(np.prod(s)) for k, s in sh.items() if k.endswith("_b"))
    return w, b, w + b


# ================================================================= 2. forward
def _pool2(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                                 (1, 1, 2, 2), (1, 1, 2, 2), "VALID")


def _drop(h, key, rate, i):
    """Inverted dropout; identity when rate == 0 or key is None."""
    if key is None or rate <= 0.0:
        return h
    keep = 1.0 - rate
    m = (jax.random.uniform(jax.random.fold_in(key, i), h.shape) < keep).astype(h.dtype)
    return h * m / keep


def forward(p, x, with_features=False, key=None, p_drop=0.0):
    """x: (N,1,28,28) float -> logits (N,10).  with_features also returns F6 (N,84).
       key/p_drop enable dropout after the two dense layers (used for MC dropout)."""
    x = jnp.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)))                 # -> (N,1,32,32)
    h = jax.lax.conv_general_dilated(x, p["c1_w"], (1, 1), "VALID")  # C1 -> (N,6,28,28)
    h = _pool2(jax.nn.relu(h + p["c1_b"][None, :, None, None]))      # S2 -> (N,6,14,14)
    h = jax.lax.conv_general_dilated(h, p["c3_w"], (1, 1), "VALID")  # C3 -> (N,16,10,10)
    h = _pool2(jax.nn.relu(h + p["c3_b"][None, :, None, None]))      # S4 -> (N,16,5,5)
    h = h.reshape(h.shape[0], -1)                                    # -> (N,400)
    h = jax.nn.relu(h @ p["c5_w"].T + p["c5_b"])                     # C5 -> (N,120)
    h = _drop(h, key, p_drop, 0)
    f6 = jax.nn.relu(h @ p["f6_w"].T + p["f6_b"])                    # F6 -> (N,84)
    f6d = _drop(f6, key, p_drop, 1)
    logits = f6d @ p["out_w"].T + p["out_b"]                         # Out -> (N,10)
    return (logits, f6) if with_features else logits


def _xent(p, xb, yb, key=None, p_drop=0.0):
    lg = forward(p, xb, key=key, p_drop=p_drop)
    return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(yb.shape[0]), yb])


# ================================================================= 3. train
# Adam with the Keras defaults
B1, B2, EPS = 0.9, 0.999, 1e-7


def _adam_init(p):
    z = lambda a: jnp.zeros_like(a)
    return dict(m=jax.tree.map(z, p), v=jax.tree.map(z, p), t=jnp.array(0, jnp.int32))


def _adam_step(p, st, g, lr):
    t = st["t"] + 1
    m = jax.tree.map(lambda m_, g_: B1 * m_ + (1 - B1) * g_, st["m"], g)
    v = jax.tree.map(lambda v_, g_: B2 * v_ + (1 - B2) * g_ ** 2, st["v"], g)
    # explicit cast: with x64 on, float ** int32 array would promote the tree to float64
    lr_t = jnp.asarray(lr * jnp.sqrt(1 - B2 ** t) / (1 - B1 ** t), DTYPE)
    p = jax.tree.map(lambda p_, m_, v_: p_ - lr_t * m_ / (jnp.sqrt(v_) + EPS), p, m, v)
    return p, dict(m=m, v=v, t=t)


def train(X, Y, epochs=20, lr=1e-3, batch=32, seed=0, Xva=None, Yva=None, verbose=True,
          p_drop=0.0, width=1.0):
    """Train LeNet-5.  X: (N,784) in [0,1], Y: (N,) int.  Returns (params, history).
       One lax.scan per epoch; the partial last batch is dropped (reshuffled each epoch)."""
    Xd = jnp.asarray(X.reshape(-1, 1, 28, 28), DTYPE)
    Yd = jnp.asarray(Y, jnp.int32)
    n_batch = X.shape[0] // batch
    p = init(seed, width)
    st = _adam_init(p)
    grad_fn = jax.value_and_grad(_xent)

    def step(carry, inp):
        p_, st_ = carry
        idx, k = inp
        xb, yb = Xd[idx], Yd[idx]
        loss, g = grad_fn(p_, xb, yb, k, p_drop)
        p_, st_ = _adam_step(p_, st_, g, lr)
        return (p_, st_), loss

    run_epoch = jax.jit(lambda p_, st_, xs: jax.lax.scan(step, (p_, st_), xs))
    rng = np.random.default_rng(seed)
    kroot = jax.random.key(seed + 7777)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(X.shape[0])[:n_batch * batch].reshape(n_batch, batch)
        t0 = time.time()
        ks = jax.random.split(jax.random.fold_in(kroot, ep), n_batch)
        (p, st), losses = run_epoch(p, st, (jnp.asarray(order), ks))
        rec = dict(epoch=ep, loss=float(jnp.mean(losses)), secs=time.time() - t0)
        if Xva is not None:
            rec["val_acc"] = accuracy(p, Xva, Yva)
        hist.append(rec)
        if verbose:
            va = f"  val_acc {rec['val_acc']:.4f}" if "val_acc" in rec else ""
            print(f"  epoch {ep + 1:2d}/{epochs}  loss {rec['loss']:.4f}{va}"
                  f"  ({rec['secs']:.1f}s)")
    return p, hist


# ================================================================= 4. evaluate
_CHUNK = 2000       # images per forward pass


def logits(p, X):
    """(N,784) -> (N,10) float64 logits, computed in chunks."""
    f = jax.jit(lambda xb: forward(p, xb))
    out = [np.asarray(f(jnp.asarray(X[i:i + _CHUNK].reshape(-1, 1, 28, 28), DTYPE)))
           for i in range(0, X.shape[0], _CHUNK)]
    return np.concatenate(out).astype(np.float64)


def features(p, X):
    """(N,784) -> (N,84) float64 F6 activations; float64 because the pipeline runs in it."""
    f = jax.jit(lambda xb: forward(p, xb, with_features=True)[1])
    out = [np.asarray(f(jnp.asarray(X[i:i + _CHUNK].reshape(-1, 1, 28, 28), DTYPE)))
           for i in range(0, X.shape[0], _CHUNK)]
    return np.concatenate(out).astype(np.float64)


def mc_dropout_logits(p, X, p_drop=0.5, n_samp=100, seed=5):
    """(N, n_samp, C) float64 logits with dropout on at test time (MC dropout)."""
    ks = jax.random.split(jax.random.key(seed), n_samp)

    @jax.jit
    def block(xb):
        return jax.vmap(lambda k: forward(p, xb, key=k, p_drop=p_drop))(ks)  # (S,b,C)

    out = []
    for i in range(0, X.shape[0], _CHUNK):
        xb = jnp.asarray(X[i:i + _CHUNK].reshape(-1, 1, 28, 28), DTYPE)
        out.append(np.asarray(block(xb)).transpose(1, 0, 2))          # -> (b,S,C)
    return np.concatenate(out).astype(np.float64)


def head(p):
    """(W_out, b_out) as float64: the trained 84 -> 10 layer the EBM head replaces."""
    return np.asarray(p["out_w"], np.float64), np.asarray(p["out_b"], np.float64)


def accuracy(p, X, Y):
    return float((logits(p, X).argmax(1) == np.asarray(Y)).mean())


# ================================================================= selftest
if __name__ == "__main__":
    import fashion_task as ft
    w, b, tot = n_params()
    print(f"parameters: {w:,} weights + {b:,} biases = {tot:,} total")
    print(f"  Liu et al. DNN 61.7 K   -> {tot:,}")
    print(f"  Liu et al. BNN 123.2 K  -> {2 * w + b:,}  (weights doubled, biases kept)")
    ft.configure(trunk="none")
    Xall, Yall = ft._pool("train")
    Xtr, Ytr = Xall[:55000], Yall[:55000]
    Xva, Yva = ft._pool("val")
    Xte, Yte = ft._pool("test")
    print("\n2-epoch smoke test on 5 000 images:")
    p, _ = train(Xtr[:5000], Ytr[:5000], epochs=2, Xva=Xva[:1000], Yva=Yva[:1000])
    print(f"test acc after 2 epochs: {accuracy(p, Xte[:2000], Yte[:2000]):.4f}")
    print(f"features: {features(p, Xte[:8]).shape}   head: {head(p)[0].shape}")
