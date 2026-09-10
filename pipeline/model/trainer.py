"""
Unified training loop for every task and recognition family 
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp

import model
import field as fieldmod
import recognition as recog
import regularisers as reg
import anchoring as anch
import estimators as est
import tree_recognition as tr


@dataclass
class TrainConfig:
    # dims
    K: int; C: int; D: int; r: int      # gates / classes / input dims / adapter rank
    seed: int = 0
    # data + likelihood, drawn with the trainer's own keys
    make_data: Callable = None          # kd -> (X, Y[, q_idx])
    build_lik: Callable = None          # key -> lik dict with W0,B,A[,Gmat,gsize,grp,g]
    adapter_noise: float = 0.0          # warm-start noise on the adapters
    b_random: bool = False              # B base is random 0.1 instead of lik["B"]
    # recognition
    family: str = "tree"                
    zero_couplings: bool = False        # ebm families: start coupling rows of the head at 0
    q_sweeps: Optional[int] = None      # ebm-gibbs: Gibbs sweeps per estimate
    q_chains: int = 64                  # ebm-gibbs: chains per input
    node_order: Optional[list] = None   # chain wiring, chain topology only
    tree_task_parents: Optional[Any] = None   # parent array for tree_topology='hier'
    tree_topology: str = "chain"        # chain | binary | star | random | hier
    # learned topology (family=tree_max_span)
    refit: Optional[str] = None         # None | "once" | "periodic"
    edge_weight: str = "abs_cov"        # Chow-Liu edge weight: "abs_cov" | "mi"
    struct_warmup: float = 0.3          # fraction of epochs before the first refit
    struct_warmup_family: str = "tree"  # "tree" | "mf" family of the warm-up segment
    refit_log: Optional[list] = None    # if a list: each refit appends {"epoch","W","parents"}
    init_log: Optional[dict] = None     # if a dict: filled with the state before the first step
    trace_out: Optional[list] = None    # if a list: per-epoch scalar dict appended per scan segment
    step_out: Optional[list] = None     # if a list: jitted step + example (carry, inp) appended
    estimator: str = "sfe-loo"          # "sfe-loo" | "relaxed"
    n_step_keys: int = 3                # per-step key splits (relaxed needs 2)
    relaxed_kl_first: bool = True       # relaxed path: trace forward(KL) before the sample
    T: int = 8; tau: float = 1.0        # gate samples per step; relaxation temperature
    # head / field / J init
    head_init: float = 0.3; c_bias: float = 0.0
    field_kind: str = "mlp"; field_n_hidden: int = 24; field_v: float = 0.1
    field_b2: Optional[float] = None    # field output-bias override
    j_init: float = -0.5; j_zero: bool = False    # negative J = gates inhibit each other
    # anchoring
    freeze: bool = False                # adapters are never updated
    anchor_B: str = "none"              # none | per-gate | group-mean | group-sum
    anchor_A: str = "none"              # none | per-gate
    wd_anchor: float = 0.1
    group_target: Optional[float] = None  # gain for the group anchor target (g*eye)
    group_sum: bool = False
    # regularisers (zero weight = absent)
    gamma0: float = 0.0; warm_frac: float = 0.15   # q_target warmup
    q_gmat: Optional[Any] = None        # (K, G') gate->group one-hot for a custom-keyed q-target
    q_tmat: Optional[Any] = None        # (n_idx, G') target pattern per q_idx; needs q_gmat
    div: float = 0.0; sparse: float = 0.0; loadbal: float = 0.0   # MoE
    label_dropout_p: float = 0.0        # fraction of samples whose onehot(y) is hidden from q
    pairs: Optional[Any] = None         # expert pairs for div/sparse/loadbal; None = K//C round-robin
    # recipe (thesis Chapter 4.5.1)
    epochs: int = 1500; lr: float = 0.05; beta_max: float = 0.5
    anneal_frac: float = 0.4; warmup: int = 0
    free_bits: float = 0.02; clip: float = 1.0; wd: float = 0.005; S: int = 12
    wd_field: Optional[float] = None     # weight decay on the field's weight leaves; None = wd


def _gclip(g, c):
    nrm = jnp.sqrt(jnp.sum(g ** 2))
    return jnp.where(nrm > c, g * (c / nrm), g)


def _setup(key, tc, spec):
    """Init RNG sequence; return (scan_key, X, Y, q_idx, lik, state).
       The split order here fixes every random number of a run."""
    K, C, D, r = tc.K, tc.C, tc.D, tc.r
    key, klik = jax.random.split(key)
    lik = tc.build_lik(klik)
    key, kd = jax.random.split(key)
    _dat = tc.make_data(kd)
    X, Y = _dat[0], _dat[1]
    q_idx = _dat[2] if len(_dat) > 2 else None     
    N = X.shape[0]
    key, *ks = jax.random.split(key, 6)            
    U = tc.head_init * jax.random.normal(ks[0], (spec.n_pre, D + C))
    c = jnp.full((spec.n_pre,), tc.c_bias) if tc.c_bias else jnp.zeros((spec.n_pre,))
    if tc.zero_couplings and spec.n_pre > K:
        # rows K: of the ebm head are the couplings; U is drawn in full so the stream is unchanged
        U = U.at[K:, :].set(0.0)
        c = c.at[K:].set(0.0)
    fp = fieldmod.init(tc.field_kind, ks[1], K, D, n_hidden=tc.field_n_hidden,
                       v_scale=tc.field_v)
    if tc.field_b2 is not None:
        fp = {**fp, "b2": jnp.full((K,), tc.field_b2)}
    chains = (jax.random.uniform(ks[2], (N, K)) < 0.5).astype(jnp.float64)   # persistent PCD chains
    if tc.b_random:
        B = 0.1 * jax.random.normal(ks[3], (K, C, r))
    elif tc.adapter_noise:
        B = lik["B"] + tc.adapter_noise * jax.random.normal(ks[3], lik["B"].shape)
    else:
        B = lik["B"]
    A = (lik["A"] + tc.adapter_noise * jax.random.normal(ks[4], lik["A"].shape)
         if tc.adapter_noise else lik["A"])
    J = jnp.zeros((K, K)) if tc.j_zero else tc.j_init * (jnp.ones((K, K)) - jnp.eye(K))   # diag(J) = 0
    return key, X, Y, q_idx, lik, (B, A, U, c, fp, J, chains)


def train(tc: TrainConfig, return_data=False):
    """Run the loop; return (params, spec, forward), plus (X, Y) with return_data=True."""
    K, C, D = tc.K, tc.C, tc.D
    if tc.family == "ebm-gibbs" and tc.q_sweeps is not None:
        import ebm_recognition_gibbs as _ebmg
        _ebmg.SWEEPS, _ebmg.N_CHAINS = int(tc.q_sweeps), int(tc.q_chains)
    spec, R = recog.make(tc.family, K, tc.node_order,
                         tc.tree_topology, tc.seed,
                         tc.tree_task_parents)
    LR, S, T = tc.lr, tc.S, tc.T
    _warm_mf = (bool(tc.refit) and tc.struct_warmup_family == "mf"
                and tc.family in ("tree", "tree_max_span"))
    spec_w, R_w = recog.make("mf", K, None) if _warm_mf else (spec, R)
    key = jax.random.key(tc.seed)
    key, X, Y, q_idx, lik, state = _setup(key, tc, spec_w)
    W0 = lik["W0"]                          # frozen, not part of state
    N = X.shape[0]
    if tc.init_log is not None:
        tc.init_log.update(U=state[2], c=state[3], field=(tc.field_kind, state[4]),
                           J=state[5], spec=spec_w, warm_family="mf" if _warm_mf else tc.family)
    feat = jnp.concatenate([X, jax.nn.one_hot(Y, C)], 1)   # q sees x and the label
    if tc.label_dropout_p:
        kdrop = jax.random.fold_in(key, 987654321)     # own key, main stream untouched
        keep = (jax.random.uniform(kdrop, (N, 1)) >= tc.label_dropout_p).astype(feat.dtype)
        feat = feat.at[:, D:].multiply(keep)
    B_init, A_init = state[0], state[1]                # per-gate anchor targets

    # Gmat (gate -> class group) serves the group anchors and the q-target
    Gmat = lik.get("Gmat")
    if Gmat is None and (tc.group_target is not None or tc.gamma0):
        Gmat = jax.nn.one_hot(jnp.arange(K) % C, C)    
    gsize = lik.get("gsize") if lik.get("gsize") is not None else (
        Gmat.sum(0) if Gmat is not None else None)
    Gtgt = (tc.group_target * jnp.eye(C)) if tc.group_target is not None else None
    pairs = tc.pairs if tc.pairs is not None else tuple(
        (K // C * g, K // C * g + 1) for g in range(C))
    group_target_oh = jax.nn.one_hot(Y, C)
    if tc.q_gmat is not None and q_idx is not None:
        q_gmat = jnp.asarray(tc.q_gmat)
        q_target_oh = (jnp.asarray(tc.q_tmat)[q_idx] if tc.q_tmat is not None
                       else jax.nn.one_hot(q_idx, q_gmat.shape[1]))
    else:
        q_gmat, q_target_oh = Gmat, group_target_oh

    def _make_step(spec, R):
        forward = R["forward"]
        def kl(pre, h, Jc, beta):
            mu, M, H = forward(pre, spec)
            return beta * reg.kl_freebits(mu, M, H, h, Jc, tc.free_bits).mean()

        def add_extra(loss, pre, A_, gamma):
            """Add q-target and MoE terms one at a time"""
            if tc.gamma0:
                loss = loss + gamma * reg.group_q_target(forward(pre, spec)[0], q_gmat, q_target_oh)
            if tc.div:
                loss = loss + tc.div * reg.reader_diversity(A_, pairs)
            if tc.sparse or tc.loadbal:
                loss = loss + reg.moe_pressures(forward(pre, spec)[0], pairs, tc.sparse, tc.loadbal)
            return loss

        if tc.estimator == "sfe-loo":
            def phi_loss(sub, h, Jc, key_, beta, gamma, z_hard, adv):
                # z_hard is drawn outside and constant here
                pre = feat @ sub["U"].T + sub["c"]
                logq = jax.vmap(lambda z: R["logq"](z, pre, spec))(z_hard)
                recon = est.sfe_recon(adv, logq) + jax.vmap(
                    lambda z: model.log_lik(W0, sub["B"], sub["A"], X, jax.lax.stop_gradient(z), Y))(z_hard).mean()
                return add_extra(-recon + kl(pre, h, Jc, beta), pre, sub["A"], gamma)
        else:
            def phi_loss(sub, h, Jc, key_, beta, gamma, z_hard, adv):
                # z is drawn inside the traced loss as a soft value
                pre = feat @ sub["U"].T + sub["c"]
                def recon_term():
                    ztil = R["sample_relaxed"](key_, pre, spec, tc.tau, T)
                    return jax.vmap(lambda z: model.log_lik(W0, sub["B"], sub["A"], X, z, Y))(ztil).mean()
                if tc.relaxed_kl_first:
                    kl_term = kl(pre, h, Jc, beta)
                    recon = recon_term()
                    return add_extra(-recon + kl_term, pre, sub["A"], gamma)
                recon = recon_term()
                return add_extra(-recon + kl(pre, h, Jc, beta), pre, sub["A"], gamma)
        grad_phi = jax.jit(jax.value_and_grad(phi_loss))    # grad w.r.t. sub = (B, A, U, c) only

        def pull_B(B):
            if tc.anchor_B == "per-gate":
                return tc.wd_anchor * anch.pull_pergate(B, B_init)
            if tc.anchor_B == "group-sum":
                return tc.wd_anchor * anch.pull_group_sum_rank0(B, Gmat, Gtgt)
            if tc.anchor_B == "group-mean":
                return tc.wd_anchor * anch.pull_group_mean_rank0(B, Gmat, gsize, Gtgt)
            if tc.anchor_B == "group-mean-bcast":
                return tc.wd_anchor * anch.pull_group_mean_bcast(B, Gmat, gsize, Gtgt)
            return tc.wd * B                                   # plain weight decay case

        def pull_A(A):
            if tc.anchor_A == "per-gate":
                return tc.wd_anchor * anch.pull_pergate(A, A_init)
            return tc.wd * A

        def step(carry, inp):
            # one full-batch epoch
            key_, beta, gamma, active = inp
            B, A, U, c, fp, J, chains = carry
            ks = jax.random.split(key_, tc.n_step_keys)
            k1, k2 = ks[0], ks[1]

            # positive phase: q moments (closed form or family estimate)
            pre = feat @ U.T + c
            mu, M, H = forward(pre, spec)
            h = fieldmod.apply(tc.field_kind, fp, X)

            # negative phase: persistent Gibbs chains on the prior
            chains = model.gibbs(k1, h, J, chains, S, K)
            zz_neg = jnp.einsum("nk,nj->nkj", chains, chains)

            # per-epoch trace, captured before any parameter moves; None leaves the graph unchanged
            _tr = None
            if tc.trace_out is not None:
                _tr = dict(
                    beta=beta, gamma=gamma,
                    kl=reg.kl_freebits(mu, M, H, h, J, tc.free_bits).mean(),   
                    gate_rate=mu.mean(),
                    mu_spread=mu.std(0).mean(),
                    b_norm=jnp.linalg.norm(B), a_norm=jnp.linalg.norm(A),
                    h_mean=h.mean(), j_off=jnp.abs(J).sum() / max(K * (K - 1), 1))

            # gradients for B, A, U, c
            sub = dict(B=B, A=A, U=U, c=c)
            if tc.estimator == "sfe-loo":
                z_hard = R["sample"](ks[2], pre, spec, T)       # (T,N,K)
                logp = jax.vmap(lambda z: model.log_lik(W0, B, A, X, z, Y))(z_hard)   # (T,N)
                adv = est.loo_advantage(logp)
                if _tr is not None:
                    _tr.update(recon=logp.mean(), adv_abs=jnp.abs(adv).mean(),
                               adv_std=adv.std())
                # h and J are not trained by autodiff; they get the contrastive updates below
                loss_val, g = grad_phi(sub, jax.lax.stop_gradient(h), jax.lax.stop_gradient(J), k2,
                                beta, gamma, jax.lax.stop_gradient(z_hard), jax.lax.stop_gradient(adv))
            else:
                loss_val, g = grad_phi(sub, jax.lax.stop_gradient(h), jax.lax.stop_gradient(J), k2,
                                beta, gamma, 0.0, 0.0)          # z is drawn inside the loss
                if _tr is not None:
                    _nan = jnp.asarray(jnp.nan)
                    _tr.update(recon=_nan, adv_abs=_nan, adv_std=_nan)
            if not tc.freeze:
                B = B - LR * (_gclip(g["B"], tc.clip) + pull_B(B))
                A = A - LR * (_gclip(g["A"], tc.clip) + pull_A(A))
            U = U - active * LR * (_gclip(g["U"], tc.clip) + tc.wd * U)
            c = c - active * LR * _gclip(g["c"], tc.clip)

            dh = beta * (mu - chains)

            def field_surrogate(fp_):
                return (fieldmod.apply(tc.field_kind, fp_, X) * jax.lax.stop_gradient(dh)).sum() / N
            gfield = jax.grad(field_surrogate)(fp)
            _wdf = tc.wd if tc.wd_field is None else tc.wd_field
            fp = {kk: v + active * LR * (_gclip(gfield[kk], tc.clip)
                                         - (_wdf * v if kk in fieldmod.WEIGHT_LEAVES else 0.0))
                  for kk, v in fp.items()}
            fp = fieldmod.maybe_project(tc.field_kind, fp)

            # coupling update: q's second moments minus the prior's
            gJ = beta * (M - zz_neg).mean(0)
            gJ = 0.5 * (gJ + gJ.T)                   # symmetric
            gJ = gJ - jnp.diag(jnp.diag(gJ))         
            J = J + active * LR * _gclip(gJ, tc.clip)
            if _tr is not None:
                _tr["loss"] = loss_val
            return (B, A, U, c, fp, J, chains), _tr
        return jax.jit(step)

    # per-epoch schedules
    ep = jnp.arange(tc.epochs)
    betas = tc.beta_max * jnp.clip((ep - tc.warmup) / (tc.anneal_frac * tc.epochs), 0.0, 1.0)
    gammas = (tc.gamma0 * jnp.clip(1.0 - ep / (tc.warm_frac * tc.epochs), 0.0, 1.0)
              if tc.gamma0 else jnp.zeros_like(ep, dtype=jnp.float64))
    actives = (ep >= tc.warmup).astype(jnp.float64)
    keys = jax.random.split(key, tc.epochs)           # one key per epoch

    if not tc.refit:
        step_fn = _make_step(spec, R)
        if tc.step_out is not None:
            tc.step_out.append(dict(step=step_fn, carry=state,
                                    inp=(keys[0], betas[0], gammas[0], actives[0])))
        carry, _tr = jax.lax.scan(step_fn, state, (keys, betas, gammas, actives))
        if tc.trace_out is not None:
            tc.trace_out.append({k: np.asarray(v) for k, v in _tr.items()})
    else:
        # learned topology: segmented scans, Chow-Liu refit on the exact posterior moments
        def _moments(carry):
            B, A, U, c, fp, J, chains = carry
            h = fieldmod.apply(tc.field_kind, fp, X)
            return _enumerate_posterior_moments(W0, B, A, X, Y, h, J, K, C)
        spec, carry = _train_with_refit(tc, spec_w, R_w, R, state, keys, betas, gammas,
                                        actives, _moments, _make_step)
    (B, A, U, c, fp, J, chains) = carry
    forward = R["forward"] if tc.refit else R_w["forward"]
    params = dict(W0=W0, B=B, A=A, U=U, c=c, J=J, field=(tc.field_kind, fp))   
    if "grp" in lik:
        params["grp"] = lik["grp"]
    return (params, spec, forward, X, Y) if return_data else (params, spec, forward)


def _refit_boundaries(tc):
    """Segment boundaries: 'once' -> [0, n_warm, epochs]; 'periodic' -> every n_warm epochs."""
    nw = int(round(tc.struct_warmup * tc.epochs))
    nw = max(1, min(nw, tc.epochs - 1))              
    if tc.refit == "periodic":
        b = list(range(0, tc.epochs, nw))
        if b[-1] != tc.epochs:
            b.append(tc.epochs)
        return b
    return [0, nw, tc.epochs]


def _enumerate_posterior_moments(W0, B, A, X, Y, h, J, K, C):
    """Exact moments (mu, M) of the true posterior p(z|x,y) by 2^K enumeration.
       Used as the Chow-Liu target instead of q, which is tree-constrained.  Note the
       target still depends on parameters trained under the warm-up topology."""
    cfg = tr.all_configs(K)                                    # (Z, K), Z = 2^K
    N = X.shape[0]
    quad = 0.5 * jnp.einsum("zk,kj,zj->z", cfg, J, cfg)        # (Z,)
    en_prior = h @ cfg.T + quad[None, :]                       # (N, Z) log p(z|x) + const

    def _logpy(z):                                             # z: (K,) one config
        lg = model.batch_logits(W0, B, A, X, jnp.broadcast_to(z, (N, K)))   # (N, C)
        return jax.nn.log_softmax(lg, -1)[jnp.arange(N), Y]    # (N,)
    LL = jax.vmap(_logpy)(cfg).T                               # (N, Z)
    p = jax.nn.softmax(en_prior + LL, axis=1)                  # (N, Z)
    mu = p @ cfg                                               # (N, K)
    M = jnp.einsum("nz,zk,zj->nkj", p, cfg, cfg)               # (N, K, K)
    return mu, M


def _mf_head_to_tree(U, c, spec):
    """Lossless mean-field -> tree head conversion: each mf row is copied into both
       its alpha and beta column, so alpha == beta and q is unchanged at the switch."""
    Ut = jnp.zeros((spec.n_pre, U.shape[1]))
    ct = jnp.zeros((spec.n_pre,))
    Ut = Ut.at[spec.pos_root].set(U[spec.root])
    ct = ct.at[spec.pos_root].set(c[spec.root])
    for k in range(spec.K):
        if k == spec.root:
            continue
        for col in (spec.pos_alpha[k], spec.pos_beta[k]):
            Ut = Ut.at[col].set(U[k])
            ct = ct.at[col].set(c[k])
    return Ut, ct


def _train_with_refit(tc, spec, R_warm, R_tree, state, keys, betas, gammas, actives,
                      moments_fn, make_step):
    """Segmented scan for a learned-tree family.  Between segments: posterior moments ->
       maximum spanning tree -> new spec.  R_warm runs the first segment, R_tree the rest;
       the mf -> tree head conversion happens once, at the first boundary."""
    bnds = _refit_boundaries(tc)
    carry = state
    R_cur = R_warm
    for i in range(len(bnds) - 1):
        lo, hi = bnds[i], bnds[i + 1]
        if hi <= lo:
            continue
        if i > 0:
            mu, M = moments_fn(carry)
            W = tr.pairwise_weights(np.asarray(mu), np.asarray(M),
                                    weight=(tc.edge_weight or "abs_cov"))
            parents = tr.max_spanning_tree_parents(W, root=0)
            spec = tr.make_tree_from_parents(parents)
            if R_cur is not R_tree:
                B, A, U, c, fp, J, chains = carry
                U, c = _mf_head_to_tree(U, c, spec)
                carry = (B, A, U, c, fp, J, chains)
                R_cur = R_tree
            if tc.refit_log is not None:
                tc.refit_log.append({"epoch": lo, "W": W, "parents": list(parents)})
        carry, _tr = jax.lax.scan(make_step(spec, R_cur), carry,
                                (keys[lo:hi], betas[lo:hi], gammas[lo:hi], actives[lo:hi]))
        if tc.trace_out is not None:
            tc.trace_out.append({k: np.asarray(v) for k, v in _tr.items()})   
    return spec, carry
