"""
Composition layer: one spreadsheet row -> one trained and measured run.

train_and_eval(resolved, out_dir, seed) builds a TrainConfig from the row, calls
trainer.train, measures, writes the run figure into out_dir and returns a flat metrics
dict with the keys of metrics.ORDER.  One function per task: blob, routing, fashion,
cifar10_resnet.
"""
from __future__ import annotations
import os
import sys

import numpy as np

# ---- paths
_PIPE = os.path.dirname(os.path.abspath(__file__))
for _p in (_PIPE, os.path.join(_PIPE, "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from jax import config as _jax_config          # noqa: E402
_jax_config.update("jax_enable_x64", True)

import trainer as trn                          # noqa: E402
import likelihood as likmod                    # noqa: E402
import readouts as ro                          # noqa: E402
import baselines as bl                         # noqa: E402
import calibration as cal                      # noqa: E402
import field as fieldmod                       # noqa: E402
from defaults import DEFAULTS                  # noqa: E402


class EngineNotImplemented(NotImplementedError):
    """Raised for knob combinations the composition does not define."""


# ---- helpers
def _d(v, default):
    """Resolved value, or the path's default when the spreadsheet cell was blank."""
    return default if v is None else v


def _joff(J):
    # Mean off-diagonal coupling.  Strongly negative means the gates inhibit each other.
    J = np.asarray(J)
    K = J.shape[0]
    return float(J[np.triu_indices(K, 1)].mean()) if K > 1 else 0.0


def _j_rows(J):
    # One column per coupling, so the results sheet holds the whole matrix.
    J = np.asarray(J)
    vals = J[np.triu_indices(J.shape[0], 1)]
    return {f"J_off_{i}": float(v) for i, v in enumerate(vals)}


def _node_order(s):
    """'0,1,2,3' -> [0,1,2,3]; 'auto'/None -> None (natural chain)."""
    if s in (None, "auto"):
        return None
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def _num(v):
    """A numeric cell that may arrive as float or as text; None if it is neither."""
    # dropout_keep, for instance, may hold a number or the word "matched".
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str) and v.replace(".", "", 1).isdigit():
        return float(v)
    return None


def _yes(v):
    """A yes/no cell as a real bool; None when blank or unrecognised, so that _d can
       supply the path default.  Not bool(v): bool('no') would be True."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "1", "y"):
        return True
    if s in ("no", "false", "0", "n"):
        return False
    return None


_ANCHOR_DEFAULT = {
    "routing":      ("none",       "none",     False),
    "fashion":      ("none",       "none",     False),
    "cifar10_resnet": ("none",     "none",     False),
    "blob":         ("group-mean", "none",     False),
}


def _anchor(path, resolved, gain):
    """(anchor_B, anchor_A, b_random, group_target, freeze) for the trainer.
       Group anchors need a class handle in the likelihood, so they exist on blob only."""
    anc = resolved["anchoring"]
    atype, freeze = anc["type"], anc["freeze"]
    aB, aA, brand = _ANCHOR_DEFAULT[path]
    gt = gain if aB.startswith("group") else None       # group anchors need a target value
    if atype is None or freeze:
        # frozen-exact never pulls, so only freeze matters.
        return aB, aA, brand, gt, freeze
    mode = anc["mode"]
    if mode == "none":                                  # anchoring off (free adapters)
        return "none", "none", False, None, False
    if mode == "per-gate":
        return "per-gate", "none", False, None, False
    if mode == "group-mean" and path == "blob":
        return "group-mean", "none", False, gain, False
    if mode == "group-sum" and path == "blob":
        return "group-sum", "none", False, gain, False
    raise EngineNotImplemented(
        f"anchor_type={atype!r} (mode {mode!r}) has no definition on path {path!r}; "
        f"group anchors need a class-handle likelihood (blob)")


def _est_ok(fam, est, tree_ok=("sfe-loo", "gumbel-concrete"),
            mf_ok=("gumbel-concrete", "sfe-loo")):
    # tree and mf accept either estimator; the energy-based families run sfe-loo only.
    if fam in ("tree", "tree_max_span"):        # tree_max_span shares the tree estimator
        return est is None or est in tree_ok
    if fam == "mf":
        return est is None or est in mf_ok
    if fam in ("ebm", "ebm-gibbs"):
        return est is None or est == "sfe-loo"
    return False


def _ebm_kw(rec, S, zero):
    """TrainConfig kwargs for the energy-based recognition families, empty otherwise.
       `zero` starts the head's coupling rows at zero.  The Gibbs backend reuses the
       prior's sweep budget S and runs 64 chains per input."""
    if rec["family"] not in ("ebm", "ebm-gibbs"):
        return {}
    kw = dict(zero_couplings=bool(zero))
    if rec["family"] == "ebm-gibbs":
        kw.update(q_sweeps=S, q_chains=64)
    return kw


def _impl_estimator(rec):
    """Semantic estimator name -> the trainer's implementation name."""
    return "relaxed" if rec["estimator"] == "gumbel-concrete" else "sfe-loo"


def _est_label(impl):
    """Implementation name -> the workbook's vocabulary, for the results file."""
    return "sfe-loo" if impl == "sfe-loo" else "gumbel"


def _refit_kw(rec):
    """Chow-Liu structure-learning kwargs, for family=tree_max_span only; empty dict
       otherwise, so tree and mf TrainConfigs are untouched.  struct_warmup_family
       defaults to the chain warm-up, 'mf' is the structure-free alternative."""
    if rec["family"] != "tree_max_span":
        return {}
    return dict(refit=_d(rec.get("refit"), "once"),
                edge_weight=_d(rec.get("edge_weight"), "abs_cov"),
                struct_warmup=_d(rec.get("struct_warmup"), 0.3),
                struct_warmup_family=_d(rec.get("struct_warmup_family"), "tree"))


def _calibrate(name, params, Xc, Yc, Xv, Yv, curves=None):
    """Temperature fitted on the val split by NLL, then raw and calibrated metrics on the
       cal split.  Pass a dict as `curves` to also get the per-sample probabilities."""
    # The gates come from the learned prior, never from q: at test time the label is unknown.
    row = cal.evaluate(name, params, cal.prior_gate_fn(params), Xc, Yc, Xv, Yv, out=curves)
    return {
        "cal_acc": float(row["acc"]),
        "ece_raw": float(row["ece_raw"]), "ece_cal": float(row["ece_cal"]),
        "nll_raw": float(row["nll_raw"]), "nll_cal": float(row["nll_cal"]),
        "T_star": float(row["T"]),
        "sel_at_0p5": float(row["sel"][0.5]),
        "sel_at_0p75": float(row["sel"][0.75]),
        "sel_at_0p25": float(row["sel"][0.25]),
        "gate_rate": float(cal.mean_gate_rate(params, Xc)),
    }


def _tree_figure(fam, params, spec, init, X, Y, C, out_dir, seed, flat):
    """Recognition-tree figure, drawn whenever the q family has a tree.  Six panels:
       the class-vs-gate matrix for recognition and prior, the tree, and J before and after.
       A readout only, so a failure is recorded and the run still counts."""
    if fam not in ("tree", "tree_max_span"):
        return
    try:
        import jax
        import jax.numpy as jnp
        import tree_recognition as trec
        import plots as _plots
        Yn = np.asarray(Y).astype(int)
        feat = jnp.concatenate([jnp.asarray(X), jax.nn.one_hot(jnp.asarray(Yn), C)], 1)

        def _mom(U, c, sp):
            """Recognition moments for a given head and topology, on this eval set."""
            pre = feat @ jnp.asarray(U).T + jnp.asarray(c)
            mu, M, _H = trec.tree_forward(pre, sp)
            return np.asarray(mu), np.asarray(M)

        def _edges(mu, M, sp):
            """Cov_q(child, parent) for the K-1 tree edges."""
            return {(min(k, p), max(k, p)):
                    float((M[:, k, p] - mu[:, k] * mu[:, p]).mean())
                    for k, p in enumerate(sp.parents) if p != -1}

        def _per_class(mu):
            """(K, C): mean gate marginal per true class."""
            return np.stack([mu[Yn == cc].mean(0) if (Yn == cc).any()
                             else np.zeros(mu.shape[1]) for cc in range(C)], axis=1)

        mu_a, M_a = _mom(params["U"], params["c"], spec)
        # Before training.  Under an mf warm-up that spec has no edges, so nothing to draw.
        sp_b = init.get("spec")
        has_b = sp_b is not None and getattr(sp_b, "parents", None) is not None
        if has_b:
            mu_b, M_b = _mom(init["U"], init["c"], sp_b)
            par_b, cov_b, mub = list(sp_b.parents), _edges(mu_b, M_b, sp_b), mu_b.mean(0)
        else:
            par_b, cov_b, mub = None, {}, np.zeros(spec.K)
        panels = dict(
            cls_recog=_per_class(mu_a),
            cls_prior=ro.routing_matrix(params, X, Yn, C),
            parents_before=par_b, mu_before=mub, cov_before=cov_b,
            parents_after=list(spec.parents), mu_after=mu_a.mean(0),
            cov_after=_edges(mu_a, M_a, spec),
            J_before=np.asarray(init.get("J", params["J"])),
            J_after=np.asarray(params["J"]),
            learned=(fam == "tree_max_span" and flat.get("n_refits", 0) >= 1))
        sub = (f"{flat.get('run_id', '')}  |  task={flat.get('task')}  family={fam}  "
               f"seed={seed}  |  acc_prior={flat.get('acc_prior', float('nan')):.3f}")
        p = _plots.tree_maps(os.path.join(out_dir, f"tree_maps_seed{seed}.png"), sub, panels)
        flat["figure_tree"] = os.path.basename(p)
    except Exception as e:                      
        flat["figure_tree"] = f"FAILED: {type(e).__name__}: {e}"


def _figure(task, params, out_dir, seed, flat, extra=None):
    import plots
    sub = (f"{flat.get('run_id', '')}  |  task={task}  family={flat.get('family')}  "
           f"seed={seed}  |  acc_prior={flat.get('acc_prior'):.3f}"
           if flat.get("acc_prior") is not None else f"task={task}")
    try:
        p = plots.make_figure(task, params, out_dir, seed, subtitle=sub, extra=extra)
        if p:
            flat["figure"] = os.path.basename(p)
    except Exception as e:                       
        flat["figure"] = f"FAILED: {type(e).__name__}: {e}"


# ===== routing
def _run_routing(resolved, out_dir, seed):
    import jax
    import routing_task as task
    DF = DEFAULTS["routing"]                     # what a blank cell means on this path
    _init = {}                                   # state before training, for the tree figure

    # ---- step 1: the row
    rec, fld, ebm, dims, w0 = (resolved["recognition"], resolved["field"],
                               resolved["ebm"], resolved["dims"], resolved["w0"])
    fam, est = rec["family"], rec["estimator"]
    if not _est_ok(fam, est):
        raise EngineNotImplemented(
            f"routing: unsupported family/estimator {fam}+{est} -- family must be "
            f"tree | tree_max_span | mf | ebm | ebm-gibbs, estimator sfe-loo | gumbel-concrete (ebm: sfe-loo only) "
            f"(both estimators are wired for every family since the estimator column)")
    est_impl = _impl_estimator(rec)

    # ---- step 2: rebuild the task constants before anything reads them
    p = resolved["task"]["params"]
    # routing_R and routing_DC cascade into the dims and into K = R.
    task.configure(R_=p.get("routing_R"), DC_=p.get("routing_DC"),
                   sigma_c=p.get("routing_sigma_c"))
    ctx = _d(_num(p.get("ctx")), task.CTX)       # how unmistakable the context signal is
    sgc = _d(_num(p.get("routing_sigma_c")), task.SIGMA_C)
    n_te = _d(resolved["task"]["n_test"], DF["n_te"])

    # ---- step 3: compose the TrainConfig from the row
    # K defaults to R, one expert per rule; D and C come from the task's construction.
    Kk, Cc, Dd = _d(dims["K"], task.R), task.C, task.D
    rank = _d(dims["rank"], DF["rank"])
    w0s = _d(w0["scale"], DF["w0_scale"])
    ainit = _d(resolved["adapters"]["init_scale"], DF["adapter_init"])
    aB, aA, brand, gtgt, freeze = _anchor("routing", resolved, gain=None)
    rcp = resolved["recipe"]

    # Data and likelihood are passed as functions, so the trainer draws with its own keys.
    def _data(kd):
        # Three returns: the rule index keys the q-target warm-up below.
        return task.make_data(DF["n_tr"], kd, ctx=ctx, sigma_c=sgc)[:3]

    def _lik(key):
        # Free adapters: nothing given away, only the task can force the experts apart.
        return likmod.build_random_free(key, K=Kk, C=Cc, D=Dd, rank=rank,
                                        w0_scale=w0s, adapter_init=ainit)

    # Gate -> rule map for the warm-up; round-robin keeps it defined when K != R.
    import jax.numpy as _jnp
    q_gmat = _jnp.asarray(np.eye(task.R)[np.arange(Kk) % task.R])          # (K, R)

    epochs, lr = _d(rcp["epochs"], DF["epochs"]), _d(rcp["lr"], DF["lr"])
    wd, clip, T = _d(rcp["wd"], DF["wd"]), _d(rcp["clip"], DF["clip"]), _d(rec["T"], DF["T"])
    refit_log = []                                  # each Chow-Liu refit appends W + parents
    pt, st, ft = trn.train(trn.TrainConfig(
        K=Kk, C=Cc, D=Dd, r=rank, seed=seed,
        make_data=_data, build_lik=_lik, refit_log=refit_log,
        family=fam, node_order=_node_order(rec["node_order"]),
        tree_topology=_d(rec.get("topology"), "chain"), init_log=_init,
        estimator=est_impl, n_step_keys=3,
        T=T, tau=_d(rec["tau"], DF["tau"]),
        head_init=_d(rec["head_init_scale"], DF["head_init"]),
        c_bias=_d(rec["c_bias"], DF["c_bias"]),
        field_kind=_d(fld["kind"], DF["field_kind"]),
        field_n_hidden=_d(fld["n_hidden"], DF["field_n_hidden"]),
        field_b2=_d(fld["bias_b2"], DF["field_b2"]), j_init=_d(ebm["j_init"], DF["j_init"]),
        freeze=freeze, anchor_B=aB, anchor_A=aA, b_random=brand, group_target=gtgt,
        wd_anchor=_d(resolved["anchoring"]["wd_anchor"], DF["wd_anchor"]),
        gamma0=_d(rcp["gamma0"], DF["gamma0"]), warm_frac=_d(rcp["warm_frac"], DF["warm_frac"]),
        q_gmat=q_gmat,
        epochs=epochs, lr=lr, beta_max=_d(rcp["beta_max"], DF["beta_max"]),
        anneal_frac=_d(rcp["anneal_frac"], DF["anneal_frac"]),
        warmup=_d(rcp["warmup"], DF["warmup"]),
        free_bits=_d(rcp["free_bits"], DF["free_bits"]),
        clip=clip, wd=wd, S=_d(ebm["gibbs_S"], DF["S"]), **_refit_kw(rec),
        **_ebm_kw(rec, _d(ebm["gibbs_S"], DF["S"]), zero=False)))
    # ---- step 4: measure.  Fixed test key, so only the training seed varies.
    Xte, Yte, rte = task.make_data(n_te, jax.random.key(123), ctx=ctx, sigma_c=sgc)

    acc, bald = ro.acc_prior_bald(pt, Xte, Yte, seed=0)     # the deployable accuracy
    Mrt = ro.routing_matrix(pt, Xte, rte, task.R)           # expert k vs true rule r
    diag = float(np.mean([Mrt[k, k] for k in range(Kk)]))   # "the right expert fired"
    off = float((Mrt.sum() - np.trace(Mrt)) / (Kk * task.R - Kk))   # "wrong experts fired"
    oh = ro.onehotness_stats(pt, Xte)                       # collapse / one-hot / multi-on

    flat = {"task": "routing", "run_id": resolved["meta"].get("run_id"), "family": fam,
            "estimator": _est_label(est_impl), "seed": seed, "acc_ctx": ctx,
            "acc_prior": float(acc), "bald": float(bald), "chance": float(task.chance()),
            "p_alloff": float(oh["p_alloff"]), "p_onehot": float(oh["p_onehot"]),
            "p_multi": float(oh["p_multi"]), "j_offdiag": _joff(pt["J"]),
            "routing_diag": diag, "routing_off": off,
            "routing_diag_minus_off": diag - off}
    flat.update(_j_rows(pt["J"]))

    # ---- topology diagnostics.  No ground-truth gate grouping here, so the well-posed
    # question is the topology itself and whether the refit moved it.
    _sp = getattr(st, "parents", None)              # MeanFieldSpec has none -> stays absent
    if _sp is not None:
        _par = tuple(int(p) for p in _sp)
        flat["tree_parents"] = ",".join(str(p) for p in _par)
        flat["tree_is_chain"] = float(_par == (-1,) + tuple(range(Kk - 1)))
    flat["n_refits"] = len(refit_log)               # 0 for tree/mf, >=1 for tree_max_span
    if refit_log:                                   # tree_max_span only
        W = np.asarray(refit_log[-1]["W"])
        offd = [W[i, j] for i in range(Kk) for j in range(Kk) if i != j]
        flat["w_mean_offdiag"] = float(np.mean(offd)) if offd else float("nan")
        flat["w_max_offdiag"] = float(np.max(offd)) if offd else float("nan")

    # ---- step 5: baselines, calibration, figure
    # Input-agnostic gate controls: same free adapters, gates that ignore x.
    # `[:2]`: train_masked unpacks (X, Y), while _data also returns the rule.
    keep = _d(_num(resolved.get("dropout_keep")), DF["keep"])
    if "dropout" in resolved["baselines"]:
        pd_ = bl.train_masked(seed, _lik, lambda kd: _data(kd)[:2], keep,
                              epochs, lr, wd, clip, T)
        flat["acc_dropout"] = float(ro.acc_masked(pd_, Xte, Yte))
    if "allon" in resolved["baselines"]:
        pa = bl.train_masked(seed, _lik, lambda kd: _data(kd)[:2], 1.0,
                             epochs, lr, wd, clip, T)
        flat["acc_allon"] = float(ro.acc_masked(pa, Xte, Yte))

    # Calibration and validation splits from key(7), the convention on every path.
    kt, kv = jax.random.split(jax.random.key(7))
    Xc, Yc, _ = task.make_data(500, kt, ctx=ctx, sigma_c=sgc)
    Xv, Yv, _ = task.make_data(400, kv, ctx=ctx, sigma_c=sgc)
    flat.update(_calibrate(fam, pt, Xc, Yc, Xv, Yv))

    _figure("routing", pt, out_dir, seed, flat, extra={"routing_matrix": Mrt})
    _tree_figure(fam, pt, st, _init, Xte, Yte, Cc, out_dir, seed, flat)
    return flat


# ===== blob
def _run_blob(resolved, out_dir, seed):
    # Three Gaussian clouds.  No input-dependent gating is needed, so the gates should buy
    # calibration rather than accuracy; that is the control against the routing result.
    # Geometry: K = 6 gates for C = 3 classes, plus a group anchor.
    import jax
    import blob_task
    DF = DEFAULTS["blob"]
    _init = {}                                   # state before training, for the tree figure
    rec, ebm, fld, dims = (resolved["recognition"], resolved["ebm"],
                           resolved["field"], resolved["dims"])
    fam, est = rec["family"], rec["estimator"]
    if not _est_ok(fam, est):
        raise EngineNotImplemented(f"blob: unsupported {fam}+{est}")
    est_impl = _impl_estimator(rec)
    p = resolved["task"]["params"]
    # CENTERS depends on R_C, and a plain attribute assignment would not reach make_data.
    blob_task.configure(R_c=p.get("blob_R_C"), sigma=p.get("blob_sigma"))

    # ---- compose the TrainConfig from the row
    Kk, Cc, Dd = _d(dims["K"], DF["K"]), dims["C"], dims["D"]
    rank = _d(dims["rank"], DF["rank"])
    gain = _d(resolved["adapters"]["gain"], DF["gain"])
    w0s = _d(resolved["w0"]["scale"], DF["w0_scale"])
    n_per = _d(resolved["task"]["n_per"], DF["n_per"])
    aB, aA, brand, gtgt, freeze = _anchor("blob", resolved, gain)
    rcp, moe = resolved["recipe"], resolved["moe"]
    params, spec, forward = trn.train(trn.TrainConfig(
        K=Kk, C=Cc, D=Dd, r=rank, seed=seed,
        make_data=lambda kd: blob_task.make_data(n_per, kd, group="train"),
        build_lik=lambda key: likmod.build_class_handle_blocked(
            key, gain, K=Kk, C=Cc, D=Dd, r=rank, w0_scale=w0s),
        family=fam, node_order=_node_order(rec["node_order"]),
        tree_topology=_d(rec.get("topology"), "chain"), init_log=_init,
        estimator=est_impl, n_step_keys=3,   # gumbel needs 2 keys, sfe-loo 3
        T=_d(rec["T"], DF["T"]), tau=_d(rec["tau"], DF["tau"]),
        head_init=_d(rec["head_init_scale"], DF["head_init"]),
        field_kind=_d(fld["kind"], DF["field_kind"]),
        field_n_hidden=_d(fld["n_hidden"], DF["field_n_hidden"]),
        j_init=_d(ebm["j_init"], DF["j_init"]),
        freeze=freeze, anchor_B=aB, anchor_A=aA, b_random=brand, group_target=gtgt,
        wd_anchor=_d(resolved["anchoring"]["wd_anchor"], DF["wd_anchor"]),
        gamma0=_d(rcp["gamma0"], DF["gamma0"]), warm_frac=_d(rcp["warm_frac"], DF["warm_frac"]),
        div=_d(moe["div"], DF["div"]), sparse=_d(moe["sparse"], DF["sparse"]),
        loadbal=_d(moe["loadbal"], DF["loadbal"]),
        epochs=_d(rcp["epochs"], DF["epochs"]), lr=_d(rcp["lr"], DF["lr"]),
        beta_max=_d(rcp["beta_max"], DF["beta_max"]),
        anneal_frac=_d(rcp["anneal_frac"], DF["anneal_frac"]),
        warmup=_d(rcp["warmup"], DF["warmup"]),
        free_bits=_d(rcp["free_bits"], DF["free_bits"]),
        clip=_d(rcp["clip"], DF["clip"]), wd=_d(rcp["wd"], DF["wd"]),
        S=_d(ebm["gibbs_S"], DF["S"]), **_refit_kw(rec),
        **_ebm_kw(rec, _d(ebm["gibbs_S"], DF["S"]), zero=False)))
    Xte, Yte = blob_task.make_data(150, jax.random.key(123), group="test")
    accp, per = ro.per_class_acc_prior(params, Xte, Yte)
    _, bald = ro.acc_prior_bald(params, Xte, Yte)
    accjm = ro.joint_map_acc(params, spec, forward, fam, Xte, Yte)
    oh = ro.onehotness_stats(params, Xte)

    flat = {"task": "blob", "run_id": resolved["meta"].get("run_id"), "family": fam,
            "estimator": _est_label(est_impl), "seed": seed,
            "acc_prior": float(accp), "bald": float(bald),
            # chance = always guess one class; bayes_ceiling = the best any model could do
            "chance": 1.0 / 3.0, "bayes_ceiling": float(blob_task.bayes_accuracy()),
            "acc_q_map": float(accjm), "j_offdiag": _joff(params["J"]),
            "p_alloff": float(oh["p_alloff"]), "p_onehot": float(oh["p_onehot"]),
            "p_multi": float(oh["p_multi"])}
    flat.update({f"acc_class_{c}": float(v) for c, v in per.items()})
    flat.update(_j_rows(params["J"]))

    # Dropout baseline: same predictor, gates ~ Bernoulli(kappa) matched to the gate rate.
    if "dropout" in resolved["baselines"]:
        kappa = ro.mean_gate_rate(params, Xte)
        da, db = ro.dropout_acc_bald(params, kappa, Xte, Yte)
        flat["acc_dropout"], flat["bald_dropout"] = float(da), float(db)

    # calibration sets: 200 / 150 points per class from key(7)
    kt, kv = jax.random.split(jax.random.key(7))
    Xc, Yc = blob_task.make_data(200, kt, group="test")
    Xv, Yv = blob_task.make_data(150, kv, group="test")
    flat.update(_calibrate(fam, params, Xc, Yc, Xv, Yv))
    _figure("blob", params, out_dir, seed, flat)
    _tree_figure(fam, params, spec, _init, Xte, Yte, Cc, out_dir, seed, flat)
    return flat


# ===== fashion
def _run_fashion(resolved, out_dir, seed):
    # Fashion-MNIST behind a frozen LeNet-5 trunk.  Same composition as routing, plus
    # fashion_w0_alpha, which interpolates the frozen base map between the trunk's trained
    # last layer (alpha = 0) and a weak random one (alpha = 1).
    # The trunk was trained on labels, so part of any accuracy here is the trunk's.
    import jax
    import fashion_task as task
    DF = DEFAULTS["fashion"]
    _init = {}

    # ---- step 1: the row
    rec, fld, ebm, dims, w0 = (resolved["recognition"], resolved["field"],
                               resolved["ebm"], resolved["dims"], resolved["w0"])
    fam, est = rec["family"], rec["estimator"]
    if not _est_ok(fam, est):
        raise EngineNotImplemented(
            f"fashion: unsupported family/estimator {fam}+{est} -- family must be "
            f"tree | tree_max_span | mf | ebm | ebm-gibbs, estimator sfe-loo | gumbel-concrete (ebm: sfe-loo only)")
    est_impl = _impl_estimator(rec)

    # ---- step 2: rebuild the task constants before anything reads them
    # C, D and the feature scale follow from the trunk, which is trained on first use.
    p = resolved["task"]["params"]
    n_per = _d(resolved["task"]["n_per"], DF["n_per"])       # per class
    n_te = _d(resolved["task"]["n_test"], DF["n_te"])        # per class
    # n_per goes to configure(): the feature scale is fitted on the images the run trains on.
    task.configure(trunk=_d(p.get("fashion_trunk"), DF["trunk"]),
                   trunk_epochs=_d(p.get("fashion_trunk_epochs"), DF["trunk_epochs"]),
                   classes=_d(p.get("fashion_classes"), DF["classes"]),
                   probe=_d(p.get("fashion_probe"), DF["probe"]),
                   ood_fraction=_d(_num(p.get("fashion_ood_fraction")), DF["ood_fraction"]),
                   seed=seed, n_per=n_per)
    alpha = float(_d(_num(p.get("fashion_w0_alpha")), DF["w0_alpha"]))
    ood_n = int(_d(_num(p.get("fashion_ood_n")), DF["ood_n"]))
    n_samp = int(_d(_num(p.get("fashion_n_samples")), DF["n_samples"]))

    # ---- step 3: compose the TrainConfig
    Kk, Cc, Dd = _d(dims["K"], task.K_DEFAULT), task.C, task.D
    rank = _d(dims["rank"], DF["rank"])
    w0s = _d(w0["scale"], DF["w0_scale"])
    ainit = _d(resolved["adapters"]["init_scale"], DF["adapter_init"])
    aB, aA, brand, gtgt, freeze = _anchor("fashion", resolved, gain=None)
    rcp, moe = resolved["recipe"], resolved["moe"]
    W0 = task.w0_matrix(alpha, w0s, seed=seed)               # (C, D), frozen

    def _data(kd):
        return task.make_data(n_per, kd, group="train")

    def _lik(key):
        # W0 comes from the task (it owns the trunk); only the adapters are drawn here.
        import jax.numpy as _jnp
        kB, kA = jax.random.split(key, 2)
        return dict(W0=_jnp.asarray(W0),
                    B=ainit * jax.random.normal(kB, (Kk, Cc, rank)),
                    A=ainit * jax.random.normal(kA, (Kk, rank, Dd)))

    epochs, lr = _d(rcp["epochs"], DF["epochs"]), _d(rcp["lr"], DF["lr"])
    wd, clip, T = _d(rcp["wd"], DF["wd"]), _d(rcp["clip"], DF["clip"]), _d(rec["T"], DF["T"])
    _iu = np.triu_indices(Kk, 1)
    all_pairs = tuple((int(a), int(b)) for a, b in zip(*_iu))
    refit_log = []
    pt, st, ft = trn.train(trn.TrainConfig(
        K=Kk, C=Cc, D=Dd, r=rank, seed=seed,
        make_data=_data, build_lik=_lik, refit_log=refit_log,
        family=fam, node_order=_node_order(rec["node_order"]),
        tree_topology=_d(rec.get("topology"), "chain"), init_log=_init,
        estimator=est_impl, n_step_keys=3,
        T=T, tau=_d(rec["tau"], DF["tau"]),
        head_init=_d(rec["head_init_scale"], DF["head_init"]),
        c_bias=_d(rec["c_bias"], DF["c_bias"]),
        field_kind=_d(fld["kind"], DF["field_kind"]),
        field_n_hidden=_d(fld["n_hidden"], DF["field_n_hidden"]),
        field_b2=_d(fld["bias_b2"], DF["field_b2"]), j_init=_d(ebm["j_init"], DF["j_init"]),
        freeze=freeze, anchor_B=aB, anchor_A=aA, b_random=brand, group_target=gtgt,
        wd_anchor=_d(resolved["anchoring"]["wd_anchor"], DF["wd_anchor"]),
        gamma0=_d(rcp["gamma0"], DF["gamma0"]), warm_frac=_d(rcp["warm_frac"], DF["warm_frac"]),
        div=_d(moe["div"], DF["div"]), sparse=_d(moe["sparse"], DF["sparse"]),
        loadbal=_d(moe["loadbal"], DF["loadbal"]), pairs=all_pairs,
        label_dropout_p=_d(rcp["label_dropout_p"], DF["label_dropout_p"]),
        epochs=epochs, lr=lr, beta_max=_d(rcp["beta_max"], DF["beta_max"]),
        anneal_frac=_d(rcp["anneal_frac"], DF["anneal_frac"]),
        warmup=_d(rcp["warmup"], DF["warmup"]),
        free_bits=_d(rcp["free_bits"], DF["free_bits"]),
        clip=clip, wd=wd, S=_d(ebm["gibbs_S"], DF["S"]), **_refit_kw(rec),
        **_ebm_kw(rec, _d(ebm["gibbs_S"], DF["S"]), zero=True)))

    # ---- step 4: measure
    # T* is fitted once on the validation pool and carried unchanged to every reported set.
    # Everything else comes from the test pool, clean and corrupted, through the same sampler,
    # so accuracy, ECE and entropy describe the same images.  Hence not _calibrate, which
    # reports on a separate pool.
    import ood_metrics as _ood
    Xte, Yte = task.make_data(n_te, jax.random.key(123), group="test")
    # The whole validation pool, in file order.  It exists only to fit T*.
    Xva, Yva = task.make_data(None, jax.random.key(7), group="val")
    T_star = _ood.fit_T(pt, Xva, Yva, n_samp=n_samp)
    clean = _ood.score(pt, Xte, Yte, n_samp=n_samp, T=T_star, n_class=Cc)
    accp, per, bald = clean["acc"], clean["per_class"], clean["h_epi"]
    # Exact enumeration: the next two build an (n_test, 2^K) table, so above the limit they
    # are skipped and their columns stay empty.  Diagnostics only; the reported numbers come
    # from ood_metrics.score, which samples the gates and does not care about K.
    _ENUM_K_MAX = 12                    # 2^12 = 4096 columns = 328 MB at n_test = 10000
    _enum_ok = Kk <= _ENUM_K_MAX
    accjm = ro.joint_map_acc(pt, st, ft, fam, Xte, Yte) if _enum_ok else None
    Mrt = ro.routing_matrix(pt, Xte, np.asarray(Yte), Cc)
    oh = ro.onehotness_stats(pt, Xte) if _enum_ok else {}

    # Parameter budget.  n_pre is the head width, K or 2K-1, taken from the spec.
    n_pre = int(getattr(st, "n_pre", Kk))
    H = _d(fld["n_hidden"], DF["field_n_hidden"])
    n_trained = (Kk * rank * Dd + Kk * Cc * rank + n_pre * (Dd + Cc + 1)
                 + H * (Dd + 1) + Kk * (H + 1) + Kk * (Kk - 1) // 2)
    # Frozen = W0 plus the trunk without the dense layer the mixture replaces.
    n_frozen = Cc * Dd + (60856 if task.TRUNK != "none" else 0)

    flat = {"task": "fashion", "run_id": resolved["meta"].get("run_id"), "family": fam,
            "estimator": _est_label(est_impl), "seed": seed,
            "n_classes": Cc, "input_dim": Dd,
            "acc_prior": float(accp), "bald": float(bald),
            "chance": float(task.chance()),
            "j_offdiag": _joff(pt["J"]),
            "acc_w0": float(task.acc_w0(W0)),
            "n_trained": int(n_trained), "n_frozen": int(n_frozen)}
    # Present only below _ENUM_K_MAX.
    if _enum_ok:
        flat.update({"acc_q_map": float(accjm),
                     "p_alloff": float(oh["p_alloff"]),
                     "p_onehot": float(oh["p_onehot"]),
                     "p_multi": float(oh["p_multi"])})
    flat.update({f"acc_class_{c}": float(v) for c, v in per.items()})
    flat.update(_j_rows(pt["J"]))
    if Kk == Cc:
        diag = float(np.mean([Mrt[k, k] for k in range(Kk)]))
        off = float((Mrt.sum() - np.trace(Mrt)) / (Kk * Cc - Kk)) if Kk * Cc > Kk else 0.0
        flat["routing_diag"], flat["routing_off"] = diag, off
        flat["routing_diag_minus_off"] = diag - off
    _sp = getattr(st, "parents", None)
    if _sp is not None:
        _par = tuple(int(x) for x in _sp)
        flat["tree_parents"] = ",".join(str(x) for x in _par)
        flat["tree_is_chain"] = float(_par == (-1,) + tuple(range(Kk - 1)))
    flat["n_refits"] = len(refit_log)

    # ---- step 5: the reported blocks and the baselines
    # Clean: every metric out of `clean`, so one accuracy and one T* per run.
    flat.update({
        "T_star": float(T_star),
        "ece_raw": clean["ece"], "nll_raw": clean["nll"],
            "brier_raw": clean["brier"], "brier_cal": clean["brier_cal"],
        "ece_cal": clean["ece_cal"], "nll_cal": clean["nll_cal"],
        "h_total": clean["h_total"], "h_alea": clean["h_alea"],
        "gate_rate": float(cal.mean_gate_rate(pt, Xte[:1000])),
    })
    flat.update({k: v for k, v in clean.items() if k.startswith("sel_at_")})

    # Corrupted: the same metrics on the selected probe and fraction, same sampler and T*,
    # so the degradation is one column minus another.  The blend seed comes from the
    # fraction, which is the standalone study's convention.
    _pseed = int(round(task.OOD_FRACTION * 100))
    Xp, Yp = task.probe(task.PROBE, task.OOD_FRACTION, n=ood_n, seed=_pseed)
    ood = _ood.score(pt, Xp, Yp, n_samp=n_samp, T=T_star)
    flat.update({
        "acc_ood": ood["acc"],
        "ece_raw_ood": ood["ece"], "nll_raw_ood": ood["nll"],
            "brier_raw_ood": ood["brier"],
        "ece_cal_ood": ood["ece_cal"], "nll_cal_ood": ood["nll_cal"],
        "h_total_ood": ood["h_total"], "h_alea_ood": ood["h_alea"],
        "bald_ood": ood["h_epi"],
    })
    # The ranking between methods depends on the corruption, so two more probes are scored.
    for kind in ("noise", "rotate"):
        if kind != task.PROBE:
            Xq, Yq = task.probe(kind, task.OOD_FRACTION, n=ood_n, seed=_pseed)
            flat[f"ece_ood_{kind}"] = _ood.score(pt, Xq, Yq, n_samp=n_samp,
                                                 T=T_star)["ece"]

    # The full fraction ladder is off by default: 30 evaluations instead of 4.
    if str(_d(p.get("fashion_ood_ladder"), DF["ood_ladder"])).lower() in ("yes", "true", "1"):
        rows = []
        for kind in ("letters", "noise", "rotate"):
            for f in task.FRACTIONS:
                Xq, Yq = task.probe(kind, f, n=ood_n, seed=int(round(f * 100)))
                r = _ood.score(pt, Xq, Yq, n_samp=n_samp, T=T_star)
                r.update(probe=kind, fraction=float(f))
                rows.append(r)
        _p = os.path.join(out_dir, f"ood_seed{seed}.csv")
        cols = ["probe", "fraction", "acc", "ece", "nll", "ece_cal", "nll_cal",
                "h_total", "h_alea", "h_epi"]
        with open(_p, "w", encoding="utf-8") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                                  for c in cols) + "\n")
        flat["ood_csv"] = os.path.basename(_p)

    keep = _d(_num(resolved.get("dropout_keep")), DF["keep"])
    if "dropout" in resolved["baselines"]:
        pd_ = bl.train_masked(seed, _lik, _data, keep, epochs, lr, wd, clip, T)
        flat["acc_dropout"] = float(ro.acc_masked(pd_, Xte, Yte))
    if "allon" in resolved["baselines"]:
        pa = bl.train_masked(seed, _lik, _data, 1.0, epochs, lr, wd, clip, T)
        flat["acc_allon"] = float(ro.acc_masked(pa, Xte, Yte))

    _figure("fashion", pt, out_dir, seed, flat,
            extra={"gate_matrix": Mrt, "img_shape": task.IMG_SHAPE, "n_ctx": 0,
                   "col_label": "true class"})
    _tree_figure(fam, pt, st, _init, Xte, Yte, Cc, out_dir, seed, flat)
    return flat


def _run_cifar10_resnet(resolved, out_dir, seed):
    # CIFAR-10 behind a frozen ResNet-13 trunk.  Derived from _run_fashion and kept in step
    # with it: same composition and w0_alpha interpolation on a different front end.  At its
    # defaults with c10r_w0_alpha = 1.0 the row reproduces the standalone study's
    # "ResNet features, random W0" entry.  The trunk was trained on labels.
    import jax
    import cifar10_resnet_task as task
    DF = DEFAULTS["cifar10_resnet"]
    _init = {}

    # ---- step 1: the row
    rec, fld, ebm, dims, w0 = (resolved["recognition"], resolved["field"],
                               resolved["ebm"], resolved["dims"], resolved["w0"])
    fam, est = rec["family"], rec["estimator"]
    if not _est_ok(fam, est):
        raise EngineNotImplemented(
            f"cifar10_resnet: unsupported family/estimator {fam}+{est} -- family must be "
            f"tree | tree_max_span | mf | ebm | ebm-gibbs, estimator sfe-loo | gumbel-concrete (ebm: sfe-loo only)")
    est_impl = _impl_estimator(rec)

    # ---- step 2: rebuild the task constants before anything reads them
    # C, D and the feature scale follow from the trunk, which is trained on first use.
    p = resolved["task"]["params"]
    n_per = _d(resolved["task"]["n_per"], DF["n_per"])       # per class
    n_te = _d(resolved["task"]["n_test"], DF["n_te"])        # per class
    # n_per goes to configure(): the feature scale is fitted on the images the run trains on.
    task.configure(trunk=_d(p.get("c10r_trunk"), DF["trunk"]),
                   schedule=_d(p.get("c10r_trunk_schedule"), DF["trunk_schedule"]),
                   trunk_epochs=_d(p.get("c10r_trunk_epochs"), DF["trunk_epochs"]),
                   classes=_d(p.get("c10r_classes"), DF["classes"]),
                   probe=_d(p.get("c10r_probe"), DF["probe"]),
                   ood_fraction=_d(_num(p.get("c10r_ood_fraction")), DF["ood_fraction"]),
                   seed=seed, n_per=n_per)
    alpha = float(_d(_num(p.get("c10r_w0_alpha")), DF["w0_alpha"]))
    ood_n = int(_d(_num(p.get("c10r_ood_n")), DF["ood_n"]))
    n_samp = int(_d(_num(p.get("c10r_n_samples")), DF["n_samples"]))

    # ---- step 3: compose the TrainConfig
    Kk, Cc, Dd = _d(dims["K"], task.K_DEFAULT), task.C, task.D
    rank = _d(dims["rank"], DF["rank"])
    w0s = _d(w0["scale"], DF["w0_scale"])
    ainit = _d(resolved["adapters"]["init_scale"], DF["adapter_init"])
    aB, aA, brand, gtgt, freeze = _anchor("cifar10_resnet", resolved, gain=None)
    rcp, moe = resolved["recipe"], resolved["moe"]
    W0 = task.w0_matrix(alpha, w0s, seed=seed)               # (C, D), frozen

    def _data(kd):
        return task.make_data(n_per, kd, group="train")

    def _lik(key):
        # W0 comes from the task (it owns the trunk); only the adapters are drawn here.
        import jax.numpy as _jnp
        kB, kA = jax.random.split(key, 2)
        return dict(W0=_jnp.asarray(W0),
                    B=ainit * jax.random.normal(kB, (Kk, Cc, rank)),
                    A=ainit * jax.random.normal(kA, (Kk, rank, Dd)))

    epochs, lr = _d(rcp["epochs"], DF["epochs"]), _d(rcp["lr"], DF["lr"])
    wd, clip, T = _d(rcp["wd"], DF["wd"]), _d(rcp["clip"], DF["clip"]), _d(rec["T"], DF["T"])
    _iu = np.triu_indices(Kk, 1)
    all_pairs = tuple((int(a), int(b)) for a, b in zip(*_iu))
    refit_log = []
    pt, st, ft = trn.train(trn.TrainConfig(
        K=Kk, C=Cc, D=Dd, r=rank, seed=seed,
        make_data=_data, build_lik=_lik, refit_log=refit_log,
        family=fam, node_order=_node_order(rec["node_order"]),
        tree_topology=_d(rec.get("topology"), "chain"), init_log=_init,
        estimator=est_impl, n_step_keys=3,
        T=T, tau=_d(rec["tau"], DF["tau"]),
        head_init=_d(rec["head_init_scale"], DF["head_init"]),
        c_bias=_d(rec["c_bias"], DF["c_bias"]),
        field_kind=_d(fld["kind"], DF["field_kind"]),
        field_n_hidden=_d(fld["n_hidden"], DF["field_n_hidden"]),
        field_b2=_d(fld["bias_b2"], DF["field_b2"]), j_init=_d(ebm["j_init"], DF["j_init"]),
        freeze=freeze, anchor_B=aB, anchor_A=aA, b_random=brand, group_target=gtgt,
        wd_anchor=_d(resolved["anchoring"]["wd_anchor"], DF["wd_anchor"]),
        gamma0=_d(rcp["gamma0"], DF["gamma0"]), warm_frac=_d(rcp["warm_frac"], DF["warm_frac"]),
        div=_d(moe["div"], DF["div"]), sparse=_d(moe["sparse"], DF["sparse"]),
        loadbal=_d(moe["loadbal"], DF["loadbal"]), pairs=all_pairs,
        label_dropout_p=_d(rcp["label_dropout_p"], DF["label_dropout_p"]),
        epochs=epochs, lr=lr, beta_max=_d(rcp["beta_max"], DF["beta_max"]),
        anneal_frac=_d(rcp["anneal_frac"], DF["anneal_frac"]),
        warmup=_d(rcp["warmup"], DF["warmup"]),
        free_bits=_d(rcp["free_bits"], DF["free_bits"]),
        clip=clip, wd=wd, S=_d(ebm["gibbs_S"], DF["S"]), **_refit_kw(rec),
        **_ebm_kw(rec, _d(ebm["gibbs_S"], DF["S"]), zero=True)))

    # ---- step 4: measure
    # T* is fitted once on the validation pool and carried unchanged to every reported set.
    # Everything else comes from the test pool, clean and corrupted, through the same sampler,
    # so accuracy, ECE and entropy describe the same images.  Hence not _calibrate, which
    # reports on a separate pool.
    import ood_metrics as _ood
    Xte, Yte = task.make_data(n_te, jax.random.key(123), group="test")
    # The whole validation pool, in file order.  It exists only to fit T*.
    Xva, Yva = task.make_data(None, jax.random.key(7), group="val")
    T_star = _ood.fit_T(pt, Xva, Yva, n_samp=n_samp)
    clean = _ood.score(pt, Xte, Yte, n_samp=n_samp, T=T_star, n_class=Cc)
    accp, per, bald = clean["acc"], clean["per_class"], clean["h_epi"]
    # Exact enumeration: the next two build an (n_test, 2^K) table, so above the limit they
    # are skipped and their columns stay empty.  Diagnostics only; the reported numbers come
    # from ood_metrics.score, which samples the gates and does not care about K.
    _ENUM_K_MAX = 12                    # 2^12 = 4096 columns = 328 MB at n_test = 10000
    _enum_ok = Kk <= _ENUM_K_MAX
    accjm = ro.joint_map_acc(pt, st, ft, fam, Xte, Yte) if _enum_ok else None
    Mrt = ro.routing_matrix(pt, Xte, np.asarray(Yte), Cc)
    oh = ro.onehotness_stats(pt, Xte) if _enum_ok else {}

    # Parameter budget.  n_pre is the head width, K or 2K-1, taken from the spec.
    n_pre = int(getattr(st, "n_pre", Kk))
    H = _d(fld["n_hidden"], DF["field_n_hidden"])
    n_trained = (Kk * rank * Dd + Kk * Cc * rank + n_pre * (Dd + Cc + 1)
                 + H * (Dd + 1) + Kk * (H + 1) + Kk * (Kk - 1) // 2)
    # Frozen = W0 plus the trunk without the dense layer the mixture replaces (that is W0).
    import resnet as _rn
    _trunk_par = _rn.n_params(n_class=task.N_CLASS_TRUNK)[3] - (task.C * _rn.N_FEAT
                                                               + task.C)
    n_frozen = Cc * Dd + (_trunk_par if task.TRUNK != "none" else 0)

    flat = {"task": "cifar10_resnet", "run_id": resolved["meta"].get("run_id"), "family": fam,
            "estimator": _est_label(est_impl), "seed": seed,
            "n_classes": Cc, "input_dim": Dd,
            "acc_prior": float(accp), "bald": float(bald),
            "chance": float(task.chance()),
            "j_offdiag": _joff(pt["J"]),
            "acc_w0": float(task.acc_w0(W0)),
            "n_trained": int(n_trained), "n_frozen": int(n_frozen)}
    # Present only below _ENUM_K_MAX.
    if _enum_ok:
        flat.update({"acc_q_map": float(accjm),
                     "p_alloff": float(oh["p_alloff"]),
                     "p_onehot": float(oh["p_onehot"]),
                     "p_multi": float(oh["p_multi"])})
    flat.update({f"acc_class_{c}": float(v) for c, v in per.items()})
    flat.update(_j_rows(pt["J"]))
    if Kk == Cc:
        diag = float(np.mean([Mrt[k, k] for k in range(Kk)]))
        off = float((Mrt.sum() - np.trace(Mrt)) / (Kk * Cc - Kk)) if Kk * Cc > Kk else 0.0
        flat["routing_diag"], flat["routing_off"] = diag, off
        flat["routing_diag_minus_off"] = diag - off
    _sp = getattr(st, "parents", None)
    if _sp is not None:
        _par = tuple(int(x) for x in _sp)
        flat["tree_parents"] = ",".join(str(x) for x in _par)
        flat["tree_is_chain"] = float(_par == (-1,) + tuple(range(Kk - 1)))
    flat["n_refits"] = len(refit_log)

    # ---- step 5: the reported blocks and the baselines
    # Clean: every metric out of `clean`, so one accuracy and one T* per run.
    flat.update({
        "T_star": float(T_star),
        "ece_raw": clean["ece"], "nll_raw": clean["nll"],
            "brier_raw": clean["brier"], "brier_cal": clean["brier_cal"],
        "ece_cal": clean["ece_cal"], "nll_cal": clean["nll_cal"],
        "h_total": clean["h_total"], "h_alea": clean["h_alea"],
        "gate_rate": float(cal.mean_gate_rate(pt, Xte[:1000])),
    })
    flat.update({k: v for k, v in clean.items() if k.startswith("sel_at_")})

    # Corrupted: the same metrics on the selected probe and fraction, same sampler and T*,
    # so the degradation is one column minus another.  The blend seed comes from the
    # fraction, which is the standalone study's convention.
    _pseed = int(round(task.OOD_FRACTION * 100))
    Xp, Yp = task.probe(task.PROBE, task.OOD_FRACTION, n=ood_n, seed=_pseed)
    ood = _ood.score(pt, Xp, Yp, n_samp=n_samp, T=T_star)
    flat.update({
        "acc_ood": ood["acc"],
        "ece_raw_ood": ood["ece"], "nll_raw_ood": ood["nll"],
            "brier_raw_ood": ood["brier"],
        "ece_cal_ood": ood["ece_cal"], "nll_cal_ood": ood["nll_cal"],
        "h_total_ood": ood["h_total"], "h_alea_ood": ood["h_alea"],
        "bald_ood": ood["h_epi"],
    })
    # The ranking between methods depends on the corruption, so two more probes are scored.
    for kind in ("noise", "rotate"):
        if kind != task.PROBE:
            Xq, Yq = task.probe(kind, task.OOD_FRACTION, n=ood_n, seed=_pseed)
            flat[f"ece_ood_{kind}"] = _ood.score(pt, Xq, Yq, n_samp=n_samp,
                                                 T=T_star)["ece"]

    # The full fraction ladder is off by default: 30 evaluations instead of 4.
    if str(_d(p.get("c10r_ood_ladder"), DF["ood_ladder"])).lower() in ("yes", "true", "1"):
        rows = []
        for kind in ("svhn", "noise", "rotate"):
            for f in task.FRACTIONS:
                Xq, Yq = task.probe(kind, f, n=ood_n, seed=int(round(f * 100)))
                r = _ood.score(pt, Xq, Yq, n_samp=n_samp, T=T_star)
                r.update(probe=kind, fraction=float(f))
                rows.append(r)
        _p = os.path.join(out_dir, f"ood_seed{seed}.csv")
        cols = ["probe", "fraction", "acc", "ece", "nll", "ece_cal", "nll_cal",
                "h_total", "h_alea", "h_epi"]
        with open(_p, "w", encoding="utf-8") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                                  for c in cols) + "\n")
        flat["ood_csv"] = os.path.basename(_p)

    keep = _d(_num(resolved.get("dropout_keep")), DF["keep"])
    if "dropout" in resolved["baselines"]:
        pd_ = bl.train_masked(seed, _lik, _data, keep, epochs, lr, wd, clip, T)
        flat["acc_dropout"] = float(ro.acc_masked(pd_, Xte, Yte))
    if "allon" in resolved["baselines"]:
        pa = bl.train_masked(seed, _lik, _data, 1.0, epochs, lr, wd, clip, T)
        flat["acc_allon"] = float(ro.acc_masked(pa, Xte, Yte))

    _figure("cifar10_resnet", pt, out_dir, seed, flat,
            extra={"gate_matrix": Mrt, "img_shape": task.IMG_SHAPE, "n_ctx": 0,
                   "col_label": "true class"})
    _tree_figure(fam, pt, st, _init, Xte, Yte, Cc, out_dir, seed, flat)
    return flat


# ===== dispatch
# Task name from the workbook cell -> the path function above.
_DISPATCH = {"blob": _run_blob, "routing": _run_routing, "fashion": _run_fashion,
             "cifar10_resnet": _run_cifar10_resnet}


def train_and_eval(resolved, out_dir, seed):
    """Train and evaluate one run for one seed; return a flat metrics dict.
       Flat means one level deep, keys = metric names, so metrics.py can write the row
       without knowing anything about the task."""
    task = resolved["task"]["name"]
    if task not in _DISPATCH:
        raise EngineNotImplemented(f"unknown task {task!r}")
    flat = _DISPATCH[task](resolved, out_dir, int(seed))
    flat.setdefault("run_id", resolved["meta"].get("run_id"))
    return flat
