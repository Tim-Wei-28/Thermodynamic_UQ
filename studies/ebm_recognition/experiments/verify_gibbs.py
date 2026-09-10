"""Checks G1..G5 of the Gibbs backend against the enum backend: moments, KL
gradient, entropy gradient, sfe-loo with unnormalised logq, end-to-end training.
Exit code 0 iff every check passes.
"""
from __future__ import annotations
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"),
                os.path.join(_HERE, "..", "..", "..", "pipeline", "model")]

import ebm_recognition as ebm            # noqa: E402
import ebm_recognition_gibbs as ebg      # noqa: E402
import jax                                # noqa: E402
import jax.numpy as jnp                   # noqa: E402
import numpy as np                        # noqa: E402
import recognition as recog               # noqa: E402
import regularisers as reg                # noqa: E402
import estimators as est                  # noqa: E402
import model as modelmod                  # noqa: E402
from trainer import TrainConfig, train    # noqa: E402
from verify_family import _blob_cfg, _run_and_score, RESULTS, check, maxerr  # noqa: E402



def _spec_pre(K=6, N=6, sweeps=60, n_chains=8192, seed=2):
    spec = ebg.GibbsSpec(**{**ebm.make_ebm_spec(K)._asdict()}, sweeps=sweeps,
                         n_chains=n_chains)
    kf, kj = jax.random.split(jax.random.key(seed))
    pre = jnp.concatenate([1.0 * jax.random.normal(kf, (N, K)),
                           0.5 * jax.random.normal(kj, (N, spec.n_pre - K))], 1)
    return spec, pre


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def g1_moments():
    spec, pre = _spec_pre()
    mu_g, M_g, _ = ebg.gibbs_forward(pre, spec)
    mu_e, M_e, _ = ebm.ebm_forward(pre, ebm.make_ebm_spec(spec.K))
    e1, e2 = maxerr(mu_g, mu_e), maxerr(M_g, M_e)
    check("G1 sampled moments -> exact", e1 < 0.04 and e2 < 0.04,
          f"mu {e1:.3f}  M {e2:.3f}  (MC tol ~3/sqrt({spec.n_chains}))")


def g2_kl_gradient():
    spec, pre = _spec_pre()
    K, N = spec.K, pre.shape[0]
    h = 0.5 * jax.random.normal(jax.random.key(3), (N, K))
    J = -0.6 * (jnp.ones((K, K)) - jnp.eye(K))

    def L(fwd, sp):
        return lambda p: reg.kl_freebits(*fwd(p, sp), h, J, 0.02).mean()

    g_g = jax.grad(L(ebg.gibbs_forward, spec))(pre)
    g_e = jax.grad(L(ebm.ebm_forward, ebm.make_ebm_spec(K)))(pre)
    c = _cos(g_g, g_e)
    ratio = float(jnp.linalg.norm(g_g) / jnp.linalg.norm(g_e))
    check("G2 KL gradient reaches the head", c > 0.90 and 0.7 < ratio < 1.3,
          f"cos {c:.3f}  |g| ratio {ratio:.2f}")


def g3_entropy_gradient():
    spec, pre = _spec_pre()

    def Hg(p):
        return ebg.gibbs_forward(p, spec)[2].mean()

    def He(p):
        return ebm.ebm_forward(p, ebm.make_ebm_spec(spec.K))[2].mean()

    c = _cos(jax.grad(Hg)(pre), jax.grad(He)(pre))
    check("G3 entropy gradient (-Cov(lq,s))", c > 0.90, f"cos {c:.3f}")


def g4_sfe_logq():
    spec, pre = _spec_pre()
    T = 8
    z = ebm.ebm_sample(jax.random.key(5), pre, spec, T)          # exact q draws
    logp = jax.random.normal(jax.random.key(6), (T, pre.shape[0]))
    adv = est.loo_advantage(logp)                                 # zero-mean by LOO

    def S(logq_fn):
        return lambda p: est.sfe_recon(adv, jax.vmap(
            lambda zz: logq_fn(zz, p, spec))(z))

    g_g = jax.grad(S(ebg.gibbs_logq))(pre)                        # unnormalised logq
    g_e = jax.grad(S(lambda zz, p, sp: ebm.ebm_logq(zz, p, ebm.make_ebm_spec(sp.K))))(pre)
    c = _cos(g_g, g_e)
    check("G4 LOO absorbs the log Z gradient", c > 0.95, f"cos {c:.3f}")


def g5_end_to_end():
    ebg.SWEEPS, ebg.N_CHAINS = 12, 64
    acc, gate, fin = _run_and_score("ebm-gibbs")
    check("G5 ebm-gibbs trains (finite params)", fin)
    check("G5 gates alive", 0.05 < gate < 0.98, f"mean mu={gate:.3f}")
    check("G5 train acc > 0.7", acc > 0.7, f"acc={acc:.3f}")


if __name__ == "__main__":
    print(f"jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
    for fn in (g1_moments, g2_kl_gradient, g3_entropy_gradient, g4_sfe_logq,
               g5_end_to_end):
        fn()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
