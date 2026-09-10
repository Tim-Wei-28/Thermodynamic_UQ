"""E1 baseline runs: blob and routing tasks, families mf/tree/ebm plus the two gate
ablations, exact-KL refit ladder, designed loopy target probe, gibbs sweep and
blob entropy maps.  Writes artifacts/e1_baselines.json and artifacts/e1_maps.npz.
Options: --seeds, --tasks, --smoke, --skip-maps, --skip-ladder, --probe-only.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_H, _os.path.join(_H, "lib"),
                 _os.path.join(_H, "..", "..", "pipeline", "model")]

import ebm_recognition as ebm           # noqa: E402  (x64 side effect first)
import ebm_recognition_gibbs as ebg     # noqa: E402
import gap_tools as gt                  # noqa: E402
import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402
import blob_task                        # noqa: E402
import routing_task                     # noqa: E402
import baselines as bl                  # noqa: E402
import calibration as cal               # noqa: E402
import field as fieldmod                # noqa: E402
import likelihood as likmod             # noqa: E402
import model as modelmod                # noqa: E402
import readouts as ro                   # noqa: E402
import recognition as recog             # noqa: E402
import trainer as trmod                 # noqa: E402
from trainer import TrainConfig, train  # noqa: E402

import tree_recognition as trec         # noqa: E402  (probe: arbitrary tree shapes)

ART = os.path.join(_H, "artifacts")
FAMILIES = ("mf", "tree", "ebm")
SGRID = (1, 2, 3, 6, 12)
J0GRID = (0.25, 0.5, 1.0, 1.5, 2.0)
# the six non-isomorphic trees on 6 vertices (parents arrays, root 0)
TREE_TOPOS = {
    "path":       [-1, 0, 1, 2, 3, 4],
    "spider311":  [-1, 0, 1, 2, 3, 1],
    "spider221":  [-1, 0, 1, 2, 3, 2],
    "dstar":      [-1, 0, 0, 0, 1, 1],
    "spider2111": [-1, 0, 0, 0, 0, 4],
    "star":       [-1, 0, 0, 0, 0, 0],
}
N_SAMP, SWEEPS_PRED = 200, 30
MAP_SEED = 0


def task_blob(smoke):
    """Blob task per pipeline/defaults.py, but with free random adapters and a
       frozen random W0."""
    C, D, K, r = blob_task.C, blob_task.D, 6, 2
    n_per = 200
    tc_kw = dict(epochs=120 if smoke else 1500, lr=0.05, T=8, S=12, tau=1.0,
                 beta_max=0.5, anneal_frac=0.4, warmup=0, free_bits=0.02,
                 clip=1.0, wd=0.005, head_init=0.3, j_init=-0.5,
                 field_kind="mlp", field_n_hidden=24,
                 gamma0=1.0, warm_frac=0.15)
    return dict(name="blob", C=C, D=D, K=K, r=r, tc_kw=tc_kw,
                make_data=lambda kd: blob_task.make_data(n_per, kd, group="train"),
                build_lik=lambda key: likmod.build_random_free(
                    key, K=K, C=C, D=D, rank=r, w0_scale=0.1, adapter_init=0.1),
                test=lambda: blob_task.make_data(1000, jax.random.key(1234),
                                                 group="test") + (None,),
                ceiling=blob_task.bayes_accuracy())


def task_routing(smoke):
    """Routing task with free adapters, K = R, strong inhibition and gates-on
       bootstrap.  The raised lr keeps the free adapters from undertraining."""
    R, C, D, r = routing_task.R, routing_task.C, routing_task.D, 4
    K, n_tr = R, 2400
    tc_kw = dict(epochs=120 if smoke else 2000, lr=0.40, T=8, S=12, tau=1.0,
                 beta_max=1.0, anneal_frac=0.4, warmup=0, free_bits=0.02,
                 clip=1.0, wd=0.01, head_init=0.1, j_init=-1.5,
                 field_kind="mlp", field_n_hidden=24,
                 c_bias=0.5, field_b2=0.5, gamma0=0.0)
    return dict(name="routing", C=C, D=D, K=K, r=r, tc_kw=tc_kw,
                make_data=lambda kd: routing_task.make_data(n_tr, kd)[:2],
                build_lik=lambda key: likmod.build_random_free(
                    key, K=K, C=C, D=D, rank=r, w0_scale=0.1, adapter_init=0.1),
                test=lambda: routing_task.make_data(3000, jax.random.key(1234)),
                ceiling=None, chance=routing_task.chance())


def predict_prior(params, Xte, K, D, C, seed=5):
    """Deployable predictor over a split: (pm (N,C), bald (N,))."""
    keys = jax.random.split(jax.random.key(seed), Xte.shape[0])
    f = jax.jit(jax.vmap(lambda k, x: modelmod.predict_bald(
        k, params["W0"], params["B"], params["A"], params["field"], params["J"],
        x, K, D, C, N_SAMP, SWEEPS_PRED)))
    pm, mi = f(keys, Xte)
    return np.asarray(pm), np.asarray(mi)


def predict_masked(params, keep, Xte, C, seed=5):
    """Same averaging as predict_prior, but gates ~ Bernoulli(keep) or all on."""
    W0, B, A = params["W0"], params["B"], params["A"]
    K, D = B.shape[0], W0.shape[1]

    def one(key, x):
        z = (jnp.ones((N_SAMP, K)) if keep >= 1.0
             else (jax.random.uniform(key, (N_SAMP, K)) < keep).astype(jnp.float64))
        probs = jax.nn.softmax(modelmod.batch_logits(
            W0, B, A, jnp.broadcast_to(x, (N_SAMP, D)), z), -1)
        pm = probs.mean(0)
        H = lambda q: -(q * jnp.log(q + 1e-12)).sum(-1)      # noqa: E731
        return pm, H(pm) - H(probs).mean()
    pm, mi = jax.jit(jax.vmap(one))(
        jax.random.split(jax.random.key(seed), Xte.shape[0]), Xte)
    return np.asarray(pm), np.asarray(mi)


def downstream(pm, mi, Yte):
    Y = np.asarray(Yte)
    return dict(acc=float((pm.argmax(1) == Y).mean()),
                ece=float(cal.ece(jnp.asarray(pm), Y)),
                nll=float(cal.nll(jnp.asarray(pm), Y)),
                brier=float(cal.brier(jnp.asarray(pm), Y)),
                bald=float(mi.mean()))


def exact_gap(params, spec, family, X, Y, C, K):
    """Exact in-run KL(q||p) and posterior correlation by enumeration."""
    R = recog.REGISTRY[family]
    feat = jnp.concatenate([X, jax.nn.one_hot(Y, C)], 1)
    pre = feat @ params["U"].T + params["c"]
    h = fieldmod.apply(params["field"][0], params["field"][1], X)
    logp = gt.true_posterior_logp(params["W0"], params["B"], params["A"],
                                  X, Y, h, params["J"], ebm.make_ebm_spec(K))
    # the gibbs family shares the enum parameterization, so its logq is enumerable
    enum_spec = ebm.make_ebm_spec(K) if family.startswith("ebm") else spec
    enum_fam = "ebm" if family.startswith("ebm") else family
    logq = recog.REGISTRY[enum_fam]["enumerate_logq"](pre, enum_spec)
    met = gt.dist_metrics(logq, logp, K)
    return dict(kl_qp=met["kl_mean"], cov_err=met["cov_err"],
                corr_post=gt.posterior_corr(logp, K)), logp, feat


def designed_logp(J0, K):
    """Task-free loopy target: h = 0 and full mutual inhibition J = -J0."""
    Z = ((np.arange(2 ** K)[:, None] >> np.arange(K)[None, :]) & 1).astype(float)
    J = -J0 * (np.ones((K, K)) - np.eye(K))
    E = 0.5 * np.einsum("nk,kl,nl->n", Z, J, Z)
    E = E - E.max()
    logp = E - np.log(np.exp(E).sum())
    return jnp.asarray(logp)[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", default="blob,routing",
                    help="subset re-run (e.g. 'routing'): the other task's rows "
                         "are PRESERVED from the existing artifact")
    ap.add_argument("--skip-maps", action="store_true")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="recompute ONLY the designed-target probe; every trained "
                         "row is carried over from the existing artifact")
    ap.add_argument("--n-ladder", type=int, default=500)
    ap.add_argument("--ladder-steps", type=int, default=2500)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        seeds = seeds[:1]
    # probe-only runs no task, so the merge below keeps every trained row
    run_tasks = [] if args.probe_only else args.tasks.split(",")
    os.makedirs(ART, exist_ok=True)
    sfx = "_smoke" if args.smoke else ""
    out_path = os.path.join(ART, f"e1_baselines{sfx}.json")
    maps_path = os.path.join(ART, f"e1_maps{sfx}.npz")

    rows, ladder, probe, route, meta = [], [], [], [], {}
    if args.probe_only and not os.path.exists(out_path):
        raise SystemExit(f"--probe-only needs an existing {out_path}")
    # partial re-run: keep the (task, seed) pairs this run does not recompute
    old =(json.load(open(out_path, encoding="utf-8"))
           if os.path.exists(out_path) else None)
    if old is not None:
        old_pairs = {(r["task"], r["seed"]) for r in old.get("rows", [])}
        redone = {(t, s) for t in run_tasks for s in seeds}
        if not old_pairs <= redone:
            keep = lambda r: (r["task"], r["seed"]) not in redone   # noqa: E731
            rows = [r for r in old.get("rows", []) if keep(r)]
            ladder = [r for r in old.get("ladder", []) if keep(r)]
            route = [r for r in old.get("route", [])
                     if ("routing", r["seed"]) not in redone]
            meta = dict(old.get("meta", {}))
            # probe is task-free and rebuilt on every run, so it is not merged
            print(f"partial re-run of tasks={run_tasks} seeds={seeds}: kept "
                  f"{len(rows)} rows / {len(ladder)} ladder / {len(route)} "
                  "route from the existing artifact", flush=True)
    t0 = time.time()

    def _dump():
        with open(out_path, "w") as f:
            json.dump(dict(settings=vars(args), n_samp=N_SAMP,
                           sweeps_pred=SWEEPS_PRED, meta=meta, rows=rows,
                           ladder=ladder, probe=probe, route=route),
                      f, indent=1, default=float)

    keep_maps = {}
    for task_fn in (task_blob, task_routing):
        task = task_fn(args.smoke)
        if task["name"] not in run_tasks:
            continue
        name, K, C, D = task["name"], task["K"], task["C"], task["D"]
        Xte, Yte, rte = task["test"]()
        # the test set is ordered by class, so sliced readouts (gate rate, ladder
        # probe) go through a fixed permutation; full-set consumers keep the
        # original order so their per-row RNG keys stay put
        probe_idx = np.asarray(
            jax.random.permutation(jax.random.key(4242), Xte.shape[0]))
        meta[name] = dict(K=K, C=C, D=D, ceiling=task.get("ceiling"),
                          chance=task.get("chance"))
        print(f"\n=== {name}: K={K} C={C} D={D}, ceiling="
              f"{task.get('ceiling')}, {len(seeds)} seeds ===", flush=True)

        for seed in seeds:
            thetas = {}
            for fam in FAMILIES:
                tc = TrainConfig(K=K, C=C, D=D, r=task["r"], seed=seed,
                                 make_data=task["make_data"],
                                 build_lik=task["build_lik"], family=fam,
                                 estimator="sfe-loo", n_step_keys=3,
                                 **task["tc_kw"])
                ts = time.time()
                params, spec, forward, X, Y = train(tc, return_data=True)
                secs = time.time() - ts
                pm, mi = predict_prior(params, Xte, K, D, C)
                row = downstream(pm, mi, np.asarray(Yte))
                gap, logp, feat = exact_gap(params, spec, fam, X, Y, C, K)
                row.update(gap)
                row.update(task=name, arm=fam, family=fam, seed=seed,
                           gate_rate=ro.mean_gate_rate(
                               params, Xte[probe_idx[:400]]),
                           train_s=secs)
                rows.append(row)
                thetas[fam] = (params, X, Y)
                if name == "routing" and rte is not None:
                    M = ro.routing_matrix(params, Xte, rte, routing_task.R)
                    route.append(dict(arm=fam, seed=seed,
                                      M=np.asarray(M).tolist()))
                # MAP_SEED rather than seeds[0], so a seed-extension run does not
                # overwrite the maps with another seed
                if name == "blob" and seed == MAP_SEED:
                    keep_maps[fam] = params
                print(f"  s{seed} {fam:5s} acc {row['acc']:.4f} ECE {row['ece']:.4f}"
                      f" NLL {row['nll']:.4f} Brier {row['brier']:.4f} | KL(q||p)"
                      f" {row['kl_qp']:.4f} corr_p {row['corr_post']:.3f}"
                      f" gates {row['gate_rate']:.2f}  [{secs:.0f}s]", flush=True)
                _dump()

            # gate ablations: dropout keep = the ebm arm's own mean gate rate
            kappa = rows[-1]["gate_rate"] if FAMILIES[-1] == "ebm" else 0.5
            for label, keep in (("dropout", kappa), ("allon", 1.0)):
                tckw = task["tc_kw"]
                mp = bl.train_masked(seed, task["build_lik"], task["make_data"],
                                     keep=keep, epochs=tckw["epochs"],
                                     lr=tckw["lr"], wd=tckw["wd"],
                                     clip=tckw["clip"], T=tckw["T"])
                pm, mi = predict_masked(mp, keep, Xte, C)
                row = downstream(pm, mi, np.asarray(Yte))
                row.update(task=name, arm=label, family="-", seed=seed,
                           gate_rate=float(min(keep, 1.0)), kl_qp=None,
                           corr_post=None, train_s=None)
                rows.append(row)
                if name == "routing" and rte is not None:
                    Mflat = np.full((K, routing_task.R), float(min(keep, 1.0)))
                    route.append(dict(arm=label, seed=seed, M=Mflat.tolist()))
                print(f"  s{seed} {label:7s} (keep={keep:.2f}) acc {row['acc']:.4f}"
                      f" ECE {row['ece']:.4f} NLL {row['nll']:.4f}", flush=True)
                _dump()

            # refit ladder on the frozen thetas
            if not args.skip_ladder:
                steps = 300 if args.smoke else args.ladder_steps
                nl = min(args.n_ladder, Xte.shape[0])
                sel = probe_idx[:nl]
                Xl, Yl = jnp.asarray(Xte)[sel], jnp.asarray(Yte)[sel]
                featl = jnp.concatenate([Xl, jax.nn.one_hot(Yl, C)], 1)
                for i_s, (src, (params, _X, _Y)) in enumerate(thetas.items()):
                    h = fieldmod.apply(params["field"][0], params["field"][1], Xl)
                    logp = gt.true_posterior_logp(
                        params["W0"], params["B"], params["A"], Xl, Yl, h,
                        params["J"], ebm.make_ebm_spec(K))
                    for i_f, fam in enumerate(FAMILIES):
                        for i_m, mode in enumerate(("amortized", "free")):
                            key = jax.random.fold_in(
                                jax.random.key(31),
                                seed * 1000 + i_s * 100 + i_f * 10 + i_m)
                            met, _ = gt.fit_family(key, fam, K, featl, logp,
                                                   amortized=(mode == "amortized"),
                                                   steps=steps)
                            ladder.append(dict(task=name, seed=seed, src=src,
                                               fam=fam, mode=mode, **met))
                            _dump()
                print(f"  s{seed} ladder done", flush=True)

        # gibbs sweep ladder (blob only)
        if name == "blob":
            grid = SGRID[:2] if args.smoke else SGRID
            for seed in seeds:
                for S in grid:
                    ebg.SWEEPS, ebg.N_CHAINS = S, 64
                    tc = TrainConfig(K=K, C=C, D=D, r=task["r"], seed=seed,
                                     make_data=task["make_data"],
                                     build_lik=task["build_lik"],
                                     family="ebm-gibbs", estimator="sfe-loo",
                                     n_step_keys=3, **task["tc_kw"])
                    ts = time.time()
                    params, spec, forward, X, Y = train(tc, return_data=True)
                    pm, mi = predict_prior(params, Xte, K, D, C)
                    row = downstream(pm, mi, np.asarray(Yte))
                    gap, _lp, _ft = exact_gap(params, spec, "ebm-gibbs", X, Y, C, K)
                    row.update(gap)
                    row.update(task=name, arm=f"ebm-g{S}", family="ebm-gibbs",
                               seed=seed, sweeps=S,
                               gate_rate=ro.mean_gate_rate(
                                   params, Xte[probe_idx[:400]]),
                               train_s=time.time() - ts)
                    rows.append(row)
                    print(f"  s{seed} ebm-g{S:<2d} acc {row['acc']:.4f} NLL"
                          f" {row['nll']:.4f} KL {row['kl_qp']:.4f}", flush=True)
                    _dump()

    # designed loopy target, task-free
    Kp = 6
    featd = jnp.ones((1, 2))                          # dummy feature row, free mode
    for J0 in (J0GRID[:2] if args.smoke else J0GRID):
        logp_d = designed_logp(J0, Kp)
        corr = gt.posterior_corr(logp_d, Kp)
        steps = 300 if args.smoke else 4000
        for i_f, fam in enumerate(FAMILIES):
            key = jax.random.fold_in(jax.random.key(7),
                                     int(J0 * 100) * 10 + i_f)
            met, _ = gt.fit_family(key, fam, Kp, featd, logp_d, amortized=False,
                                   steps=steps)
            probe.append(dict(J0=J0, fam=fam, corr_target=float(corr), **met))
            print(f"  probe J0={J0} {fam:5s} KL {met['kl_mean']:.4f} "
                  f"(|corr|_target {corr:.3f})", flush=True)
            _dump()
        # every tree shape, then the Chow-Liu fit; separate key streams (11, 13)
        # keep the family fits above reproducible
        for i_t, (tname, par) in enumerate(TREE_TOPOS.items()):
            key = jax.random.fold_in(jax.random.key(11),
                                     int(J0 * 100) * 10 + i_t)
            spec_R = (trec.make_tree_from_parents(par), recog.REGISTRY["tree"])
            met, _ = gt.fit_family(key, "tree", Kp, featd, logp_d,
                                   amortized=False, steps=steps, spec_R=spec_R)
            probe.append(dict(J0=J0, fam=f"tree:{tname}", parents=list(par),
                              corr_target=float(corr), **met))
            print(f"  probe J0={J0} tree:{tname:10s} KL {met['kl_mean']:.4f}",
                  flush=True)
            _dump()
        spec_mst = gt.mst_spec_from_logp(logp_d, Kp)
        key = jax.random.fold_in(jax.random.key(13), int(J0 * 100))
        met, _ = gt.fit_family(key, "tree", Kp, featd, logp_d, amortized=False,
                               steps=steps, spec_R=spec_mst)
        # on this exchangeable target all Chow-Liu edge weights tie, so the
        # chosen topology is recorded
        mst_par = [int(p) for p in spec_mst[0].parents]
        probe.append(dict(J0=J0, fam="tree_mst", parents=mst_par,
                          corr_target=float(corr), **met))
        print(f"  probe J0={J0} tree_mst (parents {mst_par}) "
              f"KL {met['kl_mean']:.4f}", flush=True)
        _dump()

    # entropy maps over the blob plane
    if not args.skip_maps and "blob" in run_tasks and keep_maps:
        n_grid = 41 if args.smoke else 101
        lim = 7.0
        gx = np.linspace(-lim, lim, n_grid)
        XX, YY = np.meshgrid(gx, gx)
        G = jnp.asarray(np.stack([XX.ravel(), YY.ravel()], 1))
        maps = dict(grid_x=gx, extent=np.array([-lim, lim, -lim, lim]))
        Xtr, Ytr = blob_task.make_data(200, jax.random.key(0), group="train")
        Xo, _ = blob_task.make_data(60, jax.random.key(3), group="ood")
        maps.update(train_X=np.asarray(Xtr), train_Y=np.asarray(Ytr),
                    ood_X=np.asarray(Xo), centers=np.asarray(blob_task.CENTERS))
        for fam, params in keep_maps.items():
            pm, mi = predict_prior(params, G, 6, 2, 3)
            H_tot = -(pm * np.log(pm + 1e-12)).sum(1)
            maps[f"{fam}_pm"] = pm.reshape(n_grid, n_grid, -1)
            maps[f"{fam}_h_total"] = H_tot.reshape(n_grid, n_grid)
            maps[f"{fam}_h_epi"] = np.asarray(mi).reshape(n_grid, n_grid)
            print(f"  maps: {fam} done", flush=True)
        np.savez(maps_path[:-4], **maps)
        print(f"maps -> {maps_path}", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
