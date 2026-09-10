"""
The 13-layer CIFAR ResNet of Liu et al. in plain JAX.
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

N_CLASS = 100
N_FEAT = 256                # width after global average pooling
SIDE = 32
IN_CH = 3

# (out_ch, in_ch, k, stride, bn) per convolution, in layer order
CONVS = {
    #      out  in  k  stride  bn
    "c1":  (32,   3, 3, 1, True),      # stem
    "c2":  (32,  32, 3, 1, True),      # stage 0 block
    "c3":  (32,  32, 3, 1, True),
    "c4":  (64,  32, 3, 2, True),      # stage 1 block
    "c5":  (64,  64, 3, 1, True),
    "c6":  (64,  32, 1, 2, False),     # stage 1 projection shortcut
    "c7":  (128, 64, 3, 2, True),      # stage 2 block
    "c8":  (128, 128, 3, 1, True),
    "c9":  (128, 64, 1, 2, False),     # stage 2 projection shortcut
    "c10": (256, 128, 3, 2, True),     # stage 3 block
    "c11": (256, 256, 3, 1, True),
    "c12": (256, 128, 1, 2, False),    # stage 3 projection shortcut
}
BN_LAYERS = [k for k, v in CONVS.items() if v[4]]
NO_BN = [k for k, v in CONVS.items() if not v[4]]

BN_MOM, BN_EPS = 0.99, 1e-3            # Keras defaults


def _scale_ch(ch, width):
    return max(1, int(round(ch * width)))


def _shapes(n_class=None, width=1.0):
    """{name: shape} for every array.  Only the dense head depends on n_class.
       Under the width knob the feature width scales with stage 3 (no pinned F6 here)."""
    C = N_CLASS if n_class is None else int(n_class)
    s = {}
    for name, (o, i, k, _st, bn) in CONVS.items():
        o2 = o if width == 1.0 else _scale_ch(o, width)
        i2 = i if (width == 1.0 or name == "c1") else _scale_ch(i, width)
        s[f"{name}_w"] = (o2, i2, k, k)
        if not bn:
            s[f"{name}_b"] = (o2,)     # no BatchNorm to supply the shift
        else:
            s[f"{name}_g"] = (o2,)
            s[f"{name}_bt"] = (o2,)
            s[f"{name}_m"] = (o2,)     # buffer: running mean
            s[f"{name}_v"] = (o2,)     # buffer: running variance
    s["d_w"] = (C, N_FEAT if width == 1.0 else _scale_ch(N_FEAT, width))
    s["d_b"] = (C,)
    return s


SHAPES = _shapes()
_BUF = tuple(k for k in SHAPES if k.endswith("_m") or k.endswith("_v"))


def _split(P):
    """(trainable, buffers); the BatchNorm running statistics never receive a gradient."""
    return ({k: v for k, v in P.items() if k not in _BUF},
            {k: v for k, v in P.items() if k in _BUF})


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


def init(seed=0, n_class=None, width=1.0):
    """Parameter dict with Keras defaults (Glorot kernels, zero biases and beta,
       gamma one, running mean zero, running variance one)."""
    shapes = (SHAPES if n_class in (None, N_CLASS) and width == 1.0
              else _shapes(n_class, width))
    C = N_CLASS if n_class is None else int(n_class)
    names = list(CONVS) + ["d"]
    keys = jax.random.split(jax.random.key(seed), len(names))
    P = {}
    for key, name in zip(keys, names):
        P[f"{name}_w"] = _glorot(key, shapes[f"{name}_w"])
    # widths come from the shapes dict, not CONVS (which holds the unscaled numbers)
    for name, (_o, _i, _k, _st, bn) in CONVS.items():
        o = shapes[f"{name}_w"][0]
        if bn:
            P[f"{name}_g"] = jnp.ones((o,), DTYPE)
            P[f"{name}_bt"] = jnp.zeros((o,), DTYPE)
            P[f"{name}_m"] = jnp.zeros((o,), DTYPE)
            P[f"{name}_v"] = jnp.ones((o,), DTYPE)
        else:
            P[f"{name}_b"] = jnp.zeros((o,), DTYPE)
    P["d_b"] = jnp.zeros((C,), DTYPE)
    return P


def n_params(P=None, n_class=None, width=1.0):
    """(weights, biases, batchnorm, total); 1 228 522 at n_class=10, 1 251 652 at 100."""
    shapes = (SHAPES if n_class in (None, N_CLASS) and width == 1.0
              else _shapes(n_class, width))
    w = sum(int(np.prod(s)) for k, s in shapes.items() if k.endswith("_w"))
    b = sum(int(np.prod(s)) for k, s in shapes.items() if k.endswith("_b"))
    bn = sum(int(np.prod(s)) for k, s in shapes.items()
             if k.endswith("_g") or k.endswith("_bt"))
    return w, b, bn, w + b + bn


# ================================================================= 2. forward
def _conv(x, w, stride):
    """NCHW convolution with SAME padding (keeps 32 -> 16 -> 8 -> 4 exact)."""
    return jax.lax.conv_general_dilated(x, w, (stride, stride), "SAME")


def _bn(h, P, Q, name, train):
    """BatchNorm per channel.  Train mode uses batch statistics and returns the updated
       running estimates; eval mode uses the running estimates."""
    g, bt = P[f"{name}_g"], P[f"{name}_bt"]
    if train:
        m = jnp.mean(h, axis=(0, 2, 3))
        v = jnp.var(h, axis=(0, 2, 3))
        upd = {f"{name}_m": BN_MOM * Q[f"{name}_m"] + (1 - BN_MOM) * m,
               f"{name}_v": BN_MOM * Q[f"{name}_v"] + (1 - BN_MOM) * v}
    else:
        m, v, upd = Q[f"{name}_m"], Q[f"{name}_v"], {}
    hn = (h - m[None, :, None, None]) / jnp.sqrt(v[None, :, None, None] + BN_EPS)
    return hn * g[None, :, None, None] + bt[None, :, None, None], upd


def _stage(h, P, Q, main_a, main_b, short, train, upd):
    """One residual stage; short=None means identity shortcut (stage 0)."""
    a, u = _bn(_conv(h, P[f"{main_a}_w"], CONVS[main_a][3]), P, Q, main_a, train)
    upd.update(u)
    a = jax.nn.relu(a)
    b, u = _bn(_conv(a, P[f"{main_b}_w"], CONVS[main_b][3]), P, Q, main_b, train)
    upd.update(u)
    if short is None:
        s = h
    else:
        s = _conv(h, P[f"{short}_w"], CONVS[short][3]) + P[f"{short}_b"][None, :, None, None]
    return jax.nn.relu(b + s)


def _drop(h, key, rate):
    """Inverted dropout on the pooled feature vector; identity when rate == 0 or key is None."""
    if key is None or rate <= 0.0:
        return h
    keep = 1.0 - rate
    m = (jax.random.uniform(key, h.shape) < keep).astype(h.dtype)
    return h * m / keep


def forward(P, x, with_features=False, key=None, p_drop=0.0, train=False):
    """x: (N,3,32,32) float32 -> logits (N,C).  with_features also returns the pooled
       feature vector.  In training mode returns (out, buffer_updates)."""
    Pt, Q = _split(P)
    upd = {}
    h, u = _bn(_conv(x, Pt["c1_w"], 1), Pt, Q, "c1", train)
    upd.update(u)
    h = jax.nn.relu(h)                                        # stem -> (N,32,32,32)
    h = _stage(h, Pt, Q, "c2", "c3", None, train, upd)        # stage 0
    h = _stage(h, Pt, Q, "c4", "c5", "c6", train, upd)        # stage 1 -> (N,64,16,16)
    h = _stage(h, Pt, Q, "c7", "c8", "c9", train, upd)        # stage 2 -> (N,128,8,8)
    h = _stage(h, Pt, Q, "c10", "c11", "c12", train, upd)     # stage 3 -> (N,256,4,4)
    feat = jnp.mean(h, axis=(2, 3))                           # global average pool
    logits = _drop(feat, key, p_drop) @ Pt["d_w"].T + Pt["d_b"]
    out = (logits, feat) if with_features else logits
    return (out, upd) if train else out


def _xent(Pt, Q, xb, yb, key=None, p_drop=0.0):
    """Cross entropy; buffers are passed separately so grad only sees the trainables."""
    lg, upd = forward({**Pt, **Q}, xb, key=key, p_drop=p_drop, train=True)
    loss = -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(yb.shape[0]), yb])
    return loss, upd


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


def _augment(xb, key):
    """Per-image random horizontal flip and random shift of up to 3 px (pad-then-crop)."""
    kf, kx, ky = jax.random.split(key, 3)
    n = xb.shape[0]
    flip = jax.random.bernoulli(kf, 0.5, (n, 1, 1, 1))
    xb = jnp.where(flip, xb[:, :, :, ::-1], xb)
    pad = jnp.pad(xb, ((0, 0), (0, 0), (3, 3), (3, 3)))
    ox = jax.random.randint(kx, (n,), 0, 7)
    oy = jax.random.randint(ky, (n,), 0, 7)
    crop = jax.vmap(lambda im, a, b: jax.lax.dynamic_slice(
        im, (0, a, b), (im.shape[0], SIDE, SIDE)))
    return crop(pad, oy, ox)


SCHEDULES = ("constant", "cosine", "step")


def lr_at(schedule, lr0, ep, epochs):
    """Learning rate for epoch `ep` (0-based): constant, cosine to zero, or step
       (x0.1 at 50 %, x0.01 at 75 %)."""
    if schedule == "constant":
        return lr0
    if schedule == "cosine":
        return lr0 * 0.5 * (1.0 + np.cos(np.pi * ep / max(1, epochs)))
    if schedule == "step":
        return lr0 * 0.1 ** ((ep >= 0.5 * epochs) + (ep >= 0.75 * epochs))
    raise ValueError(f"schedule must be one of {SCHEDULES} (got {schedule!r})")


def train(X, Y, epochs=100, lr=1e-3, batch=32, seed=0, Xva=None, Yva=None, verbose=True,
          p_drop=0.0, augment=True, schedule="constant", n_class=None, width=1.0):
    """Train the ResNet.  X: (N,3072) in [0,1], Y: (N,) int.  Returns (params, history).
       One lax.scan per epoch; the partial last batch is dropped (reshuffled each epoch).
       The learning rate is a traced argument so a schedule does not recompile."""
    Xd = jnp.asarray(X.reshape(-1, IN_CH, SIDE, SIDE), DTYPE)
    Yd = jnp.asarray(Y, jnp.int32)
    n_batch = X.shape[0] // batch
    P = init(seed, n_class=n_class, width=width)
    Pt, Q = _split(P)
    st = _adam_init(Pt)
    grad_fn = jax.value_and_grad(_xent, has_aux=True)

    def _make_step(lr_):
        def step(carry, inp):
            Pt_, Q_, st_ = carry
            idx, k = inp
            xb, yb = Xd[idx], Yd[idx]
            ka, kd = jax.random.split(k, 2)
            if augment:
                xb = _augment(xb, ka)
            (loss, upd), g = grad_fn(Pt_, Q_, xb, yb, kd, p_drop)
            Pt_, st_ = _adam_step(Pt_, st_, g, lr_)
            return (Pt_, {**Q_, **upd}, st_), loss
        return step

    run_epoch = jax.jit(lambda a, b, c, xs, lr_:
                        jax.lax.scan(_make_step(lr_), (a, b, c), xs))
    rng = np.random.default_rng(seed)
    kroot = jax.random.key(seed + 7777)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(X.shape[0])[:n_batch * batch].reshape(n_batch, batch)
        t0 = time.time()
        ks = jax.random.split(jax.random.fold_in(kroot, ep), n_batch)
        lr_ep = jnp.asarray(lr_at(schedule, lr, ep, epochs), DTYPE)
        (Pt, Q, st), losses = run_epoch(Pt, Q, st, (jnp.asarray(order), ks), lr_ep)
        rec = dict(epoch=ep, loss=float(jnp.mean(losses)), secs=time.time() - t0,
                   lr=float(lr_ep))
        if Xva is not None:
            rec["val_acc"] = accuracy({**Pt, **Q}, Xva, Yva)
        hist.append(rec)
        if verbose:
            va = f"  val_acc {rec['val_acc']:.4f}" if "val_acc" in rec else ""
            sc = f"  lr {rec['lr']:.2e}" if schedule != "constant" else ""
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {rec['loss']:.4f}{va}{sc}"
                  f"  ({rec['secs']:.1f}s)", flush=True)
    return {**Pt, **Q}, hist


# ================================================================= 4. evaluate
_CHUNK = 500        # images per forward pass


def _batched(fn, X):
    out = [np.asarray(fn(jnp.asarray(X[i:i + _CHUNK].reshape(-1, IN_CH, SIDE, SIDE), DTYPE)))
           for i in range(0, X.shape[0], _CHUNK)]
    return np.concatenate(out)


def logits(P, X):
    """(N,3072) -> (N,C) float64 logits, chunked, BatchNorm in eval mode."""
    f = jax.jit(lambda xb: forward(P, xb))
    return _batched(f, X).astype(np.float64)


def features(P, X):
    """(N,3072) -> (N,256) float64 pooled activations; float64 because the pipeline runs in it."""
    f = jax.jit(lambda xb: forward(P, xb, with_features=True)[1])
    return _batched(f, X).astype(np.float64)


def head(P):
    """(W_d, b_d) as float64: the trained 256 -> C layer the mixture replaces."""
    return np.asarray(P["d_w"], np.float64), np.asarray(P["d_b"], np.float64)


def mc_dropout_logits(P, X, p_drop=0.5, n_samp=100, seed=5):
    """(N, n_samp, C) float64 logits with dropout on at test time (MC dropout)."""
    ks = jax.random.split(jax.random.key(seed), n_samp)

    @jax.jit
    def block(xb):
        return jax.vmap(lambda k: forward(P, xb, key=k, p_drop=p_drop))(ks)   # (S,b,C)

    out = []
    for i in range(0, X.shape[0], _CHUNK):
        xb = jnp.asarray(X[i:i + _CHUNK].reshape(-1, IN_CH, SIDE, SIDE), DTYPE)
        out.append(np.asarray(block(xb)).transpose(1, 0, 2))                  # -> (b,S,C)
    return np.concatenate(out).astype(np.float64)


def accuracy(P, X, Y):
    return float((logits(P, X).argmax(1) == np.asarray(Y)).mean())


# ================================================================= selftest
if __name__ == "__main__":
    w, b, bn, tot = n_params()
    print(f"parameters: {w:,} conv/dense weights + {b:,} biases + {bn:,} batchnorm "
          f"= {tot:,} total")
    print(f"  Liu et al. DNN 1.25 M   -> {tot:,}")
    print(f"  Liu et al. BNN 2.50 M   -> {2 * w + b + bn:,}  (weights doubled)")
    print(f"  the replaced Dense-100  -> {N_CLASS * N_FEAT + N_CLASS:,}")

    P = init(0)
    x = jnp.asarray(np.random.default_rng(0).random((8, IN_CH, SIDE, SIDE)), DTYPE)
    lg, ft = forward(P, x, with_features=True)
    print(f"\nforward: logits {lg.shape}  features {ft.shape}  (expect (8,100) / (8,256))")
    assert lg.shape == (8, N_CLASS) and ft.shape == (8, N_FEAT)

    rng = np.random.default_rng(0)
    Xs = rng.random((256, IN_CH * SIDE * SIDE))
    Ys = rng.integers(0, N_CLASS, 256)
    t0 = time.time()
    P2, hist = train(Xs, Ys, epochs=2, batch=32, verbose=True)
    print(f"smoke train ok in {time.time() - t0:.1f}s   losses "
          f"{[round(h['loss'], 3) for h in hist]}")
    moved = sum(1 for k in P if not np.allclose(np.asarray(P[k]), np.asarray(P2[k])))
    print(f"arrays changed by training: {moved}/{len(P)}  (buffers included)")
