"""
Readouts: post-training measurements on a params dict (accuracy, gate structure,
routing, correlation).  Dims K, C, D are read off the params; nothing trains here.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

import tree_recognition as tr
import model
import field as fieldmod


# ------------------------------------------------------------------ generic (any task)
# Peak-operand budget (GiB) for the prior-predictive vmap; affects speed and memory only.
ACC_PRIOR_CHUNK_GIB = float(_os.environ.get("THRML_ACC_PRIOR_CHUNK_GIB", "4.0"))


def _chunk_rows(n, n_samp, D, itemsize):
    """Test points per chunk so the (n * n_samp, D) operand stays under the budget."""
    per_point = max(1, n_samp * D * itemsize)
    rows = int(ACC_PRIOR_CHUNK_GIB * (1024 ** 3)) // per_point
    return n if rows >= n else max(1, rows)


def _vmap_points(pb, keys, X, rows):
    """vmap `pb` over X in blocks of `rows`; keys are pre-split so chunking keeps per-point RNG."""
    n = X.shape[0]
    if rows >= n:
        return jax.vmap(pb)(keys, X)
    outs = [jax.vmap(pb)(keys[i:i + rows], X[i:i + rows]) for i in range(0, n, rows)]
    if isinstance(outs[0], tuple):
        return tuple(jnp.concatenate([o[j] for o in outs]) for j in range(len(outs[0])))
    return jnp.concatenate(outs)


def acc_prior_bald(params, X, Y, seed=7, n_samp=200, sweeps=30):
    """Prior-sampled predictive accuracy (gates from p(z|x), label unseen) and mean BALD.
       Y=None -> accuracy is None."""
    W0, B, A, J = (params[k] for k in ("W0", "B", "A", "J"))
    field_spec = params["field"]
    Kk, C_, D_ = J.shape[0], W0.shape[0], W0.shape[1]
    pb = jax.jit(lambda k, x: model.predict_bald(k, W0, B, A, field_spec, J, x,
                                                 Kk, D_, C_, n_samp, sweeps))
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    pms, mis = _vmap_points(pb, keys, X,
                            _chunk_rows(X.shape[0], n_samp, D_, X.dtype.itemsize))
    acc = float((np.asarray(pms.argmax(1)) == np.asarray(Y)).mean()) if Y is not None else None
    return acc, float(np.asarray(mis).mean())


def per_class_acc_prior(params, X, Y, seed=7):
    """acc_prior split by true class."""
    acc, _ = acc_prior_bald(params, X, Y, seed=seed)
    W0, B, A, J = (params[k] for k in ("W0", "B", "A", "J"))
    fs = params["field"]
    Kk, C_, D_ = J.shape[0], W0.shape[0], W0.shape[1]
    pb = jax.jit(lambda k, x: model.predict_bald(k, W0, B, A, fs, J, x,
                                                 Kk, D_, C_, 200, 30)[0])
    keys = jax.random.split(jax.random.key(seed), X.shape[0])
    pms = _vmap_points(pb, keys, X, _chunk_rows(X.shape[0], 200, D_, X.dtype.itemsize))
    pred = np.asarray(pms.argmax(1)); Yn = np.asarray(Y)
    per = {c: float((pred[Yn == c] == c).mean()) for c in range(C_)}
    return acc, per


def onehotness_stats(params, X):
    """Exact gate-prior structure by 2^K enumeration, averaged over X:
       P(all off), P(exactly one on), P(two or more on), per-gate marginals."""
    kind, fp = params["field"]
    J = params["J"]
    Kk = J.shape[0]
    h = fieldmod.apply(kind, fp, X)                          # (N,K)
    cfg = tr.all_configs(Kk)                                 # (2^K, K)
    en = h @ cfg.T + 0.5 * jnp.einsum("zk,kj,zj->z", cfg, J, cfg)[None, :]
    p = jax.nn.softmax(en, axis=1)                           # (N, 2^K)
    pavg = np.asarray(p.mean(0))
    non = np.asarray(cfg.sum(1))
    marg = np.asarray((p[:, :, None] * cfg[None]).sum(1).mean(0))
    return dict(p_alloff=float(pavg[non == 0].sum()),
                p_onehot=float(pavg[non == 1].sum()),
                p_multi=float(pavg[non >= 2].sum()),
                marginals=marg, pavg=pavg, configs=np.asarray(cfg))


def joint_map_acc(params, spec, forward, family, Xl, Yl):
    """Accuracy under q's joint-MAP gate config.  q sees the label, so this is a
       fit diagnostic, not a predictor."""
    W0, B, A, U, c = (params[k] for k in ("W0", "B", "A", "U", "c"))
    C_ = W0.shape[0]
    pre = jnp.concatenate([Xl, jax.nn.one_hot(Yl, C_)], 1) @ U.T + c
    import recognition as recog
    enum = recog.REGISTRY[family]["enumerate_logq"]
    cfg = tr.all_configs(spec.K)
    zmap = cfg[jnp.argmax(enum(pre, spec, cfg), axis=1)]      # (n,K) joint-MAP config
    return float((np.asarray(model.batch_logits(W0, B, A, Xl, zmap).argmax(1))
                  == np.asarray(Yl)).mean())


def prior_corr_energy(params, X, groups):
    """Exact prior p(z|x): marginal variance, within/cross-group covariance and the
       field-vs-coupling energy split.  Saturated field -> var ~ 0 -> J has no effect."""
    kind, fp = params["field"]
    J = np.asarray(params["J"])
    K = J.shape[0]
    cfg = np.asarray(tr.all_configs(K)).astype(float)          # (Z, K)
    h = np.asarray(fieldmod.apply(kind, fp, X))                # (N, K)
    field_e = h @ cfg.T                                        # (N, Z)
    quad = 0.5 * np.einsum("zk,kj,zj->z", cfg, J, cfg)         # (Z,)
    en = field_e + quad[None, :]
    en = en - en.max(1, keepdims=True)
    p = np.exp(en); p = p / p.sum(1, keepdims=True)            # (N, Z)
    mu = p @ cfg                                               # (N, K)
    M = np.einsum("nz,zk,zj->nkj", p, cfg, cfg)                # (N, K, K)
    cov = M - mu[:, :, None] * mu[:, None, :]
    var = mu * (1.0 - mu)
    Ef = (p * field_e).sum(1)
    Ec = (p * quad[None, :]).sum(1)

    g = np.asarray(groups)
    iu = np.triu_indices(K, 1)
    same = g[iu[0]] == g[iu[1]]
    cov_iu = cov[:, iu[0], iu[1]]                              # (N, n_pairs)
    ef, ec = float(np.abs(Ef).mean()), float(np.abs(Ec).mean())
    return {
        "prior_var_mean": float(var.mean()),
        "prior_cov_within": float(cov_iu[:, same].mean()) if same.any() else float("nan"),
        "prior_cov_cross": float(cov_iu[:, ~same].mean()) if (~same).any() else float("nan"),
        "prior_abscov_mean": float(np.abs(cov_iu).mean()),
        "energy_field_abs": ef,
        "energy_coupling_abs": ec,
        "energy_field_frac": float(ef / (ef + ec)) if (ef + ec) > 0 else float("nan"),
    }


def recog_corr(params, spec, forward, X, Y, groups):
    """Recognition q(z|x,y) covariance from the family's own moments (mu, M).
       Mean-field gives 0 by construction; compare with prior_corr_energy."""
    W0, U, c = params["W0"], params["U"], params["c"]
    C_ = W0.shape[0]
    pre = jnp.concatenate([jnp.asarray(X), jax.nn.one_hot(jnp.asarray(Y), C_)], 1) @ U.T + c
    mu, M, _H = forward(pre, spec)
    mu, M = np.asarray(mu), np.asarray(M)
    cov = M - mu[:, :, None] * mu[:, None, :]
    var = mu * (1.0 - mu)
    g = np.asarray(groups)
    K = mu.shape[1]
    iu = np.triu_indices(K, 1)
    same = g[iu[0]] == g[iu[1]]
    cov_iu = cov[:, iu[0], iu[1]]
    return {
        "recog_var_mean": float(var.mean()),
        "recog_cov_within": float(cov_iu[:, same].mean()) if same.any() else float("nan"),
        "recog_cov_cross": float(cov_iu[:, ~same].mean()) if (~same).any() else float("nan"),
        "recog_abscov_mean": float(np.abs(cov_iu).mean()),
    }


def mean_gate_rate(params, X, seed=1, n_samp=64, sweeps=30):
    """Mean prior gate activation; used to set a matching dropout keep-rate."""
    kind, fp = params["field"]
    J = params["J"]
    Kk = J.shape[0]
    h_fn = lambda x: fieldmod.apply(kind, fp, x[None])[0]
    def one(key, x):
        h = jnp.broadcast_to(h_fn(x), (n_samp, Kk))
        z0 = (jax.random.uniform(key, (n_samp, Kk)) < 0.5).astype(jnp.float64)
        return model.gibbs(key, h, J, z0, sweeps, Kk).mean()
    ks = jax.random.split(jax.random.key(seed), X.shape[0])
    return float(jax.vmap(one)(ks, X).mean())


def dropout_acc_bald(params, kappa, X, Y, seed=7, n_samp=200):
    """Same predictor (W0,B,A) with input-agnostic Bernoulli(kappa) gates."""
    W0, B, A = params["W0"], params["B"], params["A"]
    Kk, D_ = params["J"].shape[0], W0.shape[1]
    def one(key, x):
        z = (jax.random.uniform(key, (n_samp, Kk)) < kappa).astype(jnp.float64)
        lg = model.batch_logits(W0, B, A, jnp.broadcast_to(x, (n_samp, D_)), z)
        probs = jax.nn.softmax(lg, -1)
        pm = probs.mean(0)
        Hh = lambda q: -(q * jnp.log(q + 1e-12)).sum(-1)
        return pm, Hh(pm) - Hh(probs).mean()
    pms, mis = jax.vmap(one)(jax.random.split(jax.random.key(seed), X.shape[0]), X)
    acc = float((np.asarray(pms.argmax(1)) == np.asarray(Y)).mean())
    return acc, float(np.asarray(mis).mean())


def acc_masked(params, X, Y, seed=0, n_samp=200):
    """Accuracy of a masked-gate baseline (baselines.train_masked): Bernoulli(keep)
       gates, or all-on when keep >= 1.0."""
    W0, B, A, keep = params["W0"], params["B"], params["A"], params["keep"]
    Kk, D_ = B.shape[0], W0.shape[1]
    def one(key, x):
        z = (jnp.ones((n_samp, Kk)) if keep >= 1.0
             else (jax.random.uniform(key, (n_samp, Kk)) < keep).astype(jnp.float64))
        return jax.nn.softmax(
            model.batch_logits(W0, B, A, jnp.broadcast_to(x, (n_samp, D_)), z), -1).mean(0)
    pms = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(seed), X.shape[0]), X)
    return float((np.asarray(pms.argmax(1)) == np.asarray(Y)).mean())


# ------------------------------------------------------------------ routing-specific
def routing_matrix(params, X, rtrue, R, seed=1, sweeps=40):
    """M[k, r] = mean prior activation of expert k over inputs whose true rule is r."""
    kind, fp = params["field"]; J = params["J"]
    Kk = J.shape[0]
    h = fieldmod.apply(kind, fp, X)
    z0 = (jax.random.uniform(jax.random.key(seed), (X.shape[0], Kk)) < 0.5).astype(jnp.float64)
    z = np.asarray(model.gibbs(jax.random.key(seed + 1), h, J, z0, sweeps, Kk))
    rtrue = np.asarray(rtrue)
    M = np.zeros((Kk, R))
    for rr in range(R):
        sel = rtrue == rr
        if sel.any():
            M[:, rr] = z[sel].mean(0)
    return M


# ------------------------------------------------------------------ redundancy-specific
def within_group_cos(params):
    """Cosine similarity of the two redundant experts' effective adapters B@A per
       class group (params['grp']).  1 means duplicates."""
    B = np.asarray(params["B"]); A = np.asarray(params["A"])
    grp = np.asarray(params["grp"])
    Kk = B.shape[0]
    C_ = int(grp.max()) + 1
    Weff = np.einsum("kcr,krd->kcd", B, A).reshape(Kk, -1)
    out = []
    for cc in range(C_):
        gk = np.where(grp == cc)[0]
        a, b = Weff[gk[0]], Weff[gk[1]]
        out.append(float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return out


