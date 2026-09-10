"""
Analytic FLOPs model of one full-batch training epoch of the unified trainer
(pipeline/model/trainer.py, fixed-topology path).  predict() returns FLOPs per
block, transcendental counts and sampler ops for a run's symbols; META carries
the per-block drivers and formulas for the thesis tables (thesis E5).

Conventions: one multiply-add counts as two FLOPs.  Backward passes are counted
per block, not with a blanket factor.  Each quantity is counted once, so the
measured/model ratio kappa absorbs the trainer's recomputations.  Transcendental
constants (tau_transc, tau_samp) are per backend; tau = 0 gives pure-MAC counts.

Symbols: N train points, D input dim, C classes, F=D+C, K gates, r adapter rank,
T estimator samples, S prior Gibbs sweeps, n_pre head width per family,
H_f field-MLP hidden width; ebm-gibbs adds n_chains and S_q.
"""
from __future__ import annotations

SYMBOLS = ("family", "estimator", "N", "D", "C", "K", "r", "T", "S",
           "field_kind", "H_f", "gamma0", "n_chains", "S_q")


def n_pre(family: str, K: int) -> int:
    """Head width per family: the number of natural parameters q needs per input."""
    if family == "mf":
        return K
    if family in ("tree", "tree_max_span"):
        return 2 * K - 1
    if family.startswith("ebm"):
        return K + K * (K - 1) // 2
    raise ValueError(f"unknown family {family!r}")


def predict(family="tree", estimator="sfe-loo", N=512, D=32, C=3, K=8, r=4,
            T=8, S=12, field_kind="mlp", H_f=24, gamma0=0.0,
            n_chains=64, S_q=12, tau_transc=45.0, tau_samp=None):
    # tau_samp: transcendental constant of the estimator's per-sample blocks (backend-specific).
    if tau_samp is None:
        tau_samp = tau_transc
    """The per-epoch cost model.  Returns a dict with
         blocks           {name: FLOPs}          dense algebra, per block
         transcendentals  {name: op count}
         sampler_ops      {name: op count}       site updates + RNG draws
         total_flops      sum of blocks
         sampling_share   samplable FLOPs / total  (Gibbs blocks as digitally executed)
         n_params         trainable parameter count (for the O(P) update term)
    """
    F = D + C
    P = n_pre(family, K)
    fl, tc, so = {}, {}, {}

    # head: pre = feat @ U.T + c, forward plus the single backward matmul dU.
    fl["head"] = 4.0 * N * F * P

    # q family forward (mu, M, H) and its backward through the KL; c_q is a fitted constant.
    cq = {"mf": 2.0, "tree": 6.0, "tree_max_span": 6.0}.get(family, 6.0)
    fl["q_forward"] = 3.0 * cq * N * K * K
    if family in ("tree", "tree_max_span"):
        # XLA fuses the unrolled ancestor recurrence into its consumers, so the tree moments compile to O(N*K^3).
        fl["q_forward"] += 0.95 * N * K ** 3
    tc["q_forward"] = 2 * N * K

    # KL assembly: quad einsum of M against J, plus the per-gate marginal floor.
    fl["kl_terms"] = 4.0 * N * K * K + 10.0 * N * K
    tc["kl_logs"] = 4 * N * K

    # prior field h(x): one apply plus the surrogate's backward (its forward is CSE'd with the apply).
    ap = 2.0 * N * H_f * (D + K) if field_kind == "mlp" else 2.0 * N * D * K
    fl["field"] = 2.0 * ap
    if field_kind == "mlp":
        tc["field_tanh"] = 2 * N * H_f

    # Gibbs negative phase: S sweeps x K sites x K MACs per point, plus one expanded transcendental per site.
    fl["gibbs"] = 2.0 * S * N * K * K + 1.3 * tau_transc * S * N * K
    tc["gibbs_sigmoid"] = S * N * K
    so["site_updates"] = S * N * K
    so["rng_draws"] = S * N * K

    # prior statistics and coupling update
    fl["zz_neg"] = 2.0 * N * K * K
    fl["coupling_update"] = 3.0 * N * K * K

    # likelihood: the z-independent parts (base logits, readers) are hoisted out of the T-vmap.
    fl["base_logits"] = 2.0 * N * D * C
    fl["readers"] = 4.0 * N * K * r * D
    if estimator == "sfe-loo":
        # writers einsum per sample: forward, dB and da backward.
        fl["writers"] = 6.0 * T * N * C * K * r
        fl["softmax"] = 5.0 * T * N * C
        tc["softmax_exp"] = 2 * T * N * C
        # ancestral sampling and logq: about two transcendentals per (gate, sample) each.
        fl["q_sample"] = 4.0 * T * N * K + 2.0 * tau_samp * T * N * K
        fl["q_logq"] = 6.0 * T * N * K + 2.0 * tau_samp * T * N * K
        fl["sfe_glue"] = 6.0 * T * N
        so["rng_draws"] += T * N * K
        tc["sample_sigmoid"] = T * N * K
        tc["logq_logs"] = 2 * T * N * K
    else:
        # relaxed: gumbel noise + sigmoid, and one extra backward route dz through the writers.
        fl["writers"] = 8.0 * T * N * C * K * r
        fl["softmax"] = 5.0 * T * N * C
        tc["softmax_exp"] = 2 * T * N * C
        fl["q_sample"] = 8.0 * T * N * K + 3.0 * tau_samp * T * N * K
        so["rng_draws"] = so["rng_draws"] + T * N * K
        tc["sample_transc"] = 3 * T * N * K

    # optional q-target warmup term (gamma0 > 0): mu @ Gmat plus squared error, fwd + bwd.
    if gamma0:
        fl["q_target"] = 6.0 * N * K * C

    # EBM family variants replace the q_forward block.
    if family == "ebm":                               # enum backend: 2^K states
        Z = 2 ** K
        fl["q_forward"] = 6.0 * N * Z * P
        tc["q_forward"] = 2 * N * Z
    chains_q = 0.0                                    # q-side chain FLOPs (offloadable)
    if family == "ebm-gibbs":
        # q-side sampler: n_chains moment chains plus the T sample chains, S_q sweeps each; suff-stats and DiCE glue stay digital.
        chains_q = (2.0 * n_chains * S_q * N * K * K
                    + 2.0 * T * S_q * N * K * K + 1.3 * tau_transc * T * S_q * N * K)
        fl["q_forward"] = (2.0 * n_chains * S_q * N * K * K
                           + 1.3 * tau_transc * n_chains * S_q * N * K
                           + 23.5 * n_chains * N * P
                           + 1.3 * tau_transc * n_chains * N * K)
        chains_q += 1.3 * tau_transc * n_chains * S_q * N * K
        fl["q_sample"] = 2.0 * T * S_q * N * K * K + 1.3 * tau_transc * T * S_q * N * K
        fl["q_logq"] = 4.0 * T * N * P                # suff(z).pre dot, fwd + bwd
        tc["q_forward"] = n_chains * S_q * N * K
        so["site_updates"] += (n_chains + T) * S_q * N * K
        so["rng_draws"] += (n_chains + T) * S_q * N * K

    # parameter updates, clips, decay
    n_params = (P * F + P) + K * C * r + K * r * D + K * K + (
        H_f * (D + K) + H_f + K if field_kind == "mlp" else K * D + K)
    fl["param_updates"] = 8.0 * n_params

    total = float(sum(fl.values()))
    # offload_flops: digital FLOPs spent emulating a sampler (prior Gibbs, plus the q-side chains under ebm-gibbs).
    offload = fl["gibbs"] + chains_q
    return dict(blocks=fl, transcendentals=tc, sampler_ops=so,
                total_flops=total, offload_flops=offload,
                sampling_share=offload / total, n_params=n_params)


def pick(config: dict) -> dict:
    """The SYMBOLS subset of a config dict, for predict(**pick(row))."""
    return {k: config[k] for k in SYMBOLS if k in config}


# Per-block drivers, formula and hardware target for the thesis tables ("TSU" = the sampler absorbs it).
META = {
    "head":            dict(drivers="N, D, C, K(n_pre)", hw="GPU",
                            formula="4*N*F*n_pre"),
    "q_forward":       dict(drivers="N, K (family)", hw="GPU>TSU (ebm-gibbs)",
                            formula="3*c_q*N*K^2   (c_q: mf 2, tree 6; "
                                    "ebm-gibbs: 2*n_chains*S_q*N*K^2 + 6*n_chains*N*n_pre)"),
    "kl_terms":        dict(drivers="N, K", hw="GPU",
                            formula="4*N*K^2 + 10*N*K"),
    "field":           dict(drivers="N, D, K, H_f", hw="GPU",
                            formula="2*ap;  ap = 2*N*H_f*(D+K) [mlp] | 2*N*D*K [linear]"),
    "gibbs":           dict(drivers="S, N, K", hw="TSU",
                            formula="2*S*N*K^2 + 1.3*tau*S*N*K"),
    "zz_neg":          dict(drivers="N, K", hw="GPU (near-sampler)",
                            formula="2*N*K^2"),
    "coupling_update": dict(drivers="N, K", hw="GPU",
                            formula="3*N*K^2"),
    "base_logits":     dict(drivers="N, D, C", hw="GPU",
                            formula="2*N*D*C"),
    "readers":         dict(drivers="N, K, r, D", hw="GPU",
                            formula="4*N*K*r*D"),
    "writers":         dict(drivers="T, N, C, K, r", hw="GPU",
                            formula="6*T*N*C*K*r   (relaxed: 8*)"),
    "softmax":         dict(drivers="T, N, C", hw="GPU",
                            formula="5*T*N*C"),
    "q_sample":        dict(drivers="T, N, K", hw="GPU>TSU (ebm-gibbs)",
                            formula="(4+2*tau)*T*N*K   (relaxed: (8+3*tau))"),
    "q_logq":          dict(drivers="T, N, K", hw="GPU",
                            formula="(6+2*tau)*T*N*K   (sfe-loo only)"),
    "sfe_glue":        dict(drivers="T, N", hw="GPU",
                            formula="6*T*N"),
    "q_target":        dict(drivers="N, K, C", hw="GPU",
                            formula="6*N*K*C   (gamma0 > 0 only)"),
    "param_updates":   dict(drivers="#params", hw="GPU",
                            formula="8*P_params"),
}
