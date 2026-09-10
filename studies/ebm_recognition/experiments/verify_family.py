"""Correctness checks C0..C8 for lib/ebm_recognition.py (enum backend): registry,
mean-field limit, moments, normalisation, score, sampler, KL identity, finite
differences and end-to-end training.  Exit code 0 iff every check passes.
"""
from __future__ import annotations
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"),
                os.path.join(_HERE, "..", "..", "..", "pipeline", "model")]

import ebm_recognition as ebm          # noqa: E402
import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402
import numpy as np                      # noqa: E402
import tree_recognition as tr           # noqa: E402
import recognition as recog             # noqa: E402
import regularisers as reg              # noqa: E402
import model as modelmod                # noqa: E402
from trainer import TrainConfig, train  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")


def maxerr(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def c0_registry():
    spec, R = recog.make("ebm", 6)
    ok = (spec.n_pre == 6 + 15) and (R is ebm.BUNDLE)
    check("C0 registry/make roundtrip", ok, f"n_pre={spec.n_pre} (expect 21)")
    try:                                   # the K cap must raise
        ebm.make_ebm_spec(ebm.MAX_K_ENUM + 1)
        check("C0 K cap refuses", False, "no error raised")
    except ValueError:
        check("C0 K cap refuses", True)


def c1_mf_limit():
    K, N, T = 5, 7, 6
    key = jax.random.key(0)
    spec = ebm.make_ebm_spec(K)
    spec_mf = tr.make_mf_spec(K)
    pre_mf = 1.2 * jax.random.normal(key, (N, K))
    pre = jnp.concatenate([pre_mf, jnp.zeros((N, spec.n_pre - K))], 1)

    mu_e, M_e, H_e = ebm.ebm_forward(pre, spec)
    mu_m, M_m, H_m = tr.mf_forward(pre_mf, spec_mf)
    check("C1 forward mu == mf", maxerr(mu_e, mu_m) < 1e-12, f"maxerr={maxerr(mu_e, mu_m):.2e}")
    check("C1 forward M  == mf", maxerr(M_e, M_m) < 1e-12, f"maxerr={maxerr(M_e, M_m):.2e}")
    check("C1 forward H  == mf", maxerr(H_e, H_m) < 1e-12, f"maxerr={maxerr(H_e, H_m):.2e}")

    z = (jax.random.uniform(jax.random.key(1), (T, N, K)) < 0.5).astype(pre.dtype)
    lq_e = ebm.ebm_logq(z, pre, spec)
    lq_m = tr.mf_logq(z, pre_mf, spec_mf)      # mf adds eps inside logs, hence the slack
    check("C1 logq == mf", maxerr(lq_e, lq_m) < 1e-9, f"maxerr={maxerr(lq_e, lq_m):.2e}")

    en_e = ebm.ebm_enumerate_logq(pre, spec)
    en_m = tr.mf_enumerate_logq(pre_mf, spec_mf)
    check("C1 enumerate == mf", maxerr(en_e, en_m) < 1e-9, f"maxerr={maxerr(en_e, en_m):.2e}")

    sc_e = ebm.ebm_score(z, pre, spec)
    sc_m = tr.mf_score(z, pre_mf, spec_mf)
    check("C1 score fields == mf", maxerr(sc_e[..., :K], sc_m) < 1e-12,
          f"maxerr={maxerr(sc_e[..., :K], sc_m):.2e}")
    # factorised q: E[z_i z_j] = mu_i mu_j
    iu, ju = jnp.asarray(spec.iu), jnp.asarray(spec.ju)
    sc_pair = z[..., iu] * z[..., ju] - (mu_m[:, iu] * mu_m[:, ju])[None]
    check("C1 score couplings == zz - mumu", maxerr(sc_e[..., K:], sc_pair) < 1e-12,
          f"maxerr={maxerr(sc_e[..., K:], sc_pair):.2e}")


def _random_pre(key, N, spec, f_scale=1.0, j_scale=0.7):
    kf, kj = jax.random.split(key)
    K = spec.K
    return jnp.concatenate([f_scale * jax.random.normal(kf, (N, K)),
                            j_scale * jax.random.normal(kj, (N, spec.n_pre - K))], 1)


def c2_moments():
    spec = ebm.make_ebm_spec(6)
    pre = _random_pre(jax.random.key(2), 5, spec)
    mu, M, H = ebm.ebm_forward(pre, spec)
    cfg = tr.all_configs(spec.K).astype(pre.dtype)
    mu2, M2, H2 = tr.moments_from_logq(ebm.ebm_enumerate_logq(pre, spec), cfg)
    ok = maxerr(mu, mu2) < 1e-10 and maxerr(M, M2) < 1e-10 and maxerr(H, H2) < 1e-10
    check("C2 forward == moments_from_logq", ok,
          f"mu {maxerr(mu, mu2):.2e}  M {maxerr(M, M2):.2e}  H {maxerr(H, H2):.2e}")


def c3_normalise():
    spec = ebm.make_ebm_spec(5)
    pre = _random_pre(jax.random.key(3), 4, spec)
    en = ebm.ebm_enumerate_logq(pre, spec)
    lse = jax.scipy.special.logsumexp(en, axis=1)
    check("C3 sums to one", float(jnp.max(jnp.abs(lse))) < 1e-10,
          f"max|logsumexp|={float(jnp.max(jnp.abs(lse))):.2e}")
    cfg = tr.all_configs(spec.K).astype(pre.dtype)
    errs = []
    for i in (0, 7, 19, 31):                       # incl. all-off and all-on
        z = jnp.broadcast_to(cfg[i], (pre.shape[0], spec.K))
        errs.append(maxerr(ebm.ebm_logq(z, pre, spec), en[:, i]))
    check("C3 logq matches enumerate rows", max(errs) < 1e-10, f"maxerr={max(errs):.2e}")


def c4_score_grad():
    spec = ebm.make_ebm_spec(5)
    pre = _random_pre(jax.random.key(4), 3, spec)
    z = (jax.random.uniform(jax.random.key(5), (3, spec.K)) < 0.5).astype(pre.dtype)
    g = jax.grad(lambda p: ebm.ebm_logq(z, p, spec).sum())(pre)   # rows are independent
    sc = ebm.ebm_score(z, pre, spec)
    check("C4 score == grad(logq)", maxerr(g, sc) < 1e-10, f"maxerr={maxerr(g, sc):.2e}")


def c5_sampler():
    spec = ebm.make_ebm_spec(4)
    pre = _random_pre(jax.random.key(6), 3, spec)
    T = 200_000
    z = ebm.ebm_sample(jax.random.key(7), pre, spec, T)            # (T,N,4)
    idx = (z @ (2.0 ** jnp.arange(spec.K))).astype(jnp.int32)      # all_configs bit order
    probs = jnp.exp(ebm.ebm_enumerate_logq(pre, spec))             # (N,16)
    emp = jax.vmap(lambda col: jnp.bincount(col, length=16) / T, in_axes=1)(idx)
    err = maxerr(emp, probs)                                       # tolerance ~4.5 sd
    check("C5 sampler matches enumeration", err < 5e-3, f"maxfreqerr={err:.2e} (T={T})")


def c6_kl_identity():
    spec = ebm.make_ebm_spec(6)
    N = 5
    pre = _random_pre(jax.random.key(8), N, spec)
    kh, kJ = jax.random.split(jax.random.key(9))
    h = jax.random.normal(kh, (N, spec.K))
    Jr = 0.5 * jax.random.normal(kJ, (spec.K, spec.K))
    J = 0.5 * (Jr + Jr.T) * (1.0 - jnp.eye(spec.K))                # symmetric, hollow
    mu, M, H = ebm.ebm_forward(pre, spec)
    kl_fb = reg.kl_freebits(mu, M, H, h, J, 0.0)                   # KL up to +logZ_prior
    # enumerated truth via the prior's natural parameters
    pre_p = ebm.prior_pre(h, J, spec)
    logq = ebm.ebm_enumerate_logq(pre, spec)
    logp = ebm.ebm_enumerate_logq(pre_p, spec)
    kl_true = (jnp.exp(logq) * (logq - logp)).sum(1)
    cfg, S = jnp.asarray(tr.all_configs(spec.K), pre.dtype), None
    logZ_p = jax.scipy.special.logsumexp(pre_p @ ebm._suff(cfg, spec).T, axis=1)
    err = maxerr(kl_fb + logZ_p, kl_true)
    check("C6 kl_freebits + logZ_p == enumerated KL", err < 1e-8, f"maxerr={err:.2e}")


def c7_finite_diff():
    spec = ebm.make_ebm_spec(4)
    N = 3
    pre0 = _random_pre(jax.random.key(10), N, spec)
    h = 0.5 * jax.random.normal(jax.random.key(11), (N, spec.K))
    J = -0.4 * (jnp.ones((spec.K, spec.K)) - jnp.eye(spec.K))
    z = (jax.random.uniform(jax.random.key(12), (4, N, spec.K)) < 0.5).astype(pre0.dtype)

    def L(pre):
        mu, M, H = ebm.ebm_forward(pre, spec)
        return reg.kl_freebits(mu, M, H, h, J, 0.02).mean() + 0.3 * ebm.ebm_logq(z, pre, spec).mean()

    g = jax.grad(L)(pre0)
    rng = np.random.default_rng(0)
    eps, worst = 1e-6, 0.0
    for _ in range(15):
        i, j = int(rng.integers(N)), int(rng.integers(spec.n_pre))
        d = jnp.zeros_like(pre0).at[i, j].set(eps)
        fd = (L(pre0 + d) - L(pre0 - d)) / (2 * eps)
        worst = max(worst, abs(float(fd) - float(g[i, j])) / max(1e-8, abs(float(fd))))
    check("C7 autodiff vs central FD", worst < 1e-5, f"worst relerr={worst:.2e}")


def _blob_cfg(family):
    C, D, K, r, n_per = 3, 6, 6, 2, 15

    def make_data(kd):
        ky, kx = jax.random.split(kd)
        Y = jax.random.randint(ky, (C * n_per,), 0, C)
        means = 2.0 * jax.nn.one_hot(jnp.arange(C), D)
        X = means[Y] + 0.6 * jax.random.normal(kx, (C * n_per, D))
        return X, Y.astype(jnp.int32)

    def build_lik(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return dict(W0=0.1 * jax.random.normal(k1, (C, D)),        # weak frozen base
                    B=0.1 * jax.random.normal(k2, (K, C, r)),
                    A=0.1 * jax.random.normal(k3, (K, r, D)))

    return TrainConfig(K=K, C=C, D=D, r=r, seed=0, make_data=make_data, build_lik=build_lik,
                       family=family, estimator="sfe-loo", T=8, epochs=400, lr=0.05,
                       beta_max=0.5, anneal_frac=0.4, warmup=20, free_bits=0.02,
                       clip=1.0, wd=0.005, S=12)


def _run_and_score(family):
    tc = _blob_cfg(family)
    params, spec, forward, X, Y = train(tc, return_data=True)
    feat = jnp.concatenate([X, jax.nn.one_hot(Y, tc.C)], 1)
    pre = feat @ params["U"].T + params["c"]
    mu, M, H = forward(pre, spec)
    _, R = recog.make(family, tc.K)
    z = R["sample"](jax.random.key(99), pre, spec, 64)             # (64,N,K) from q
    probs = jax.vmap(lambda zt: jax.nn.softmax(
        modelmod.batch_logits(params["W0"], params["B"], params["A"], X, zt)))(z).mean(0)
    acc = float((probs.argmax(1) == Y).mean())
    finite = all(bool(jnp.all(jnp.isfinite(params[k]))) for k in ("B", "A", "U", "c", "J"))
    return acc, float(mu.mean()), finite


def c8_trainer():
    acc_e, gate_e, fin_e = _run_and_score("ebm")
    check("C8 ebm trains (finite params)", fin_e)
    check("C8 ebm gates alive", 0.05 < gate_e < 0.98, f"mean mu={gate_e:.3f}")
    check("C8 ebm train acc > 0.7", acc_e > 0.7, f"acc={acc_e:.3f}")
    acc_m, gate_m, fin_m = _run_and_score("mf")                    # control: registry injection must not disturb mf
    check("C8 mf control still trains", fin_m and acc_m > 0.7,
          f"acc={acc_m:.3f}  mean mu={gate_m:.3f}")
    print(f"      (ebm acc {acc_e:.3f} vs mf acc {acc_m:.3f} on the smoke blob task)")


if __name__ == "__main__":
    print(f"jax {jax.__version__}  x64={jax.config.jax_enable_x64}  "
          f"devices={[d.platform for d in jax.devices()]}")
    for fn in (c0_registry, c1_mf_limit, c2_moments, c3_normalise, c4_score_grad,
               c5_sampler, c6_kl_identity, c7_finite_diff, c8_trainer):
        fn()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
