"""Post-hoc temperature scaling for the E1 arms.  Retrains the same five arms as
run_baselines.py, fits one temperature per arm and seed on an independent
validation draw, and re-scores the test set.  Writes artifacts/e1_temp.json.
Options: --seeds, --tasks, --smoke.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

import run_baselines as rb
import jax
import jax.numpy as jnp

import baselines as bl
import calibration as cal
import readouts as ro
import routing_task
import blob_task
from trainer import TrainConfig, train

ARMS = ("mf", "tree", "ebm", "dropout", "allon")
VAL_KEY = 4321                                   


def fit_temperature(pm_val, Y_val):
    """Scalar T minimizing validation NLL of softmax(log pm / T); log-grid then
       a fine local pass."""
    logp = np.log(np.asarray(pm_val) + 1e-12)
    Y = np.asarray(Y_val)

    def nll_at(T):
        z = logp / T
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(Y)), Y] - np.log(np.exp(z).sum(1))).mean())

    Ts = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    i = int(np.argmin([nll_at(t) for t in Ts]))
    fine = np.linspace(Ts[max(i - 1, 0)], Ts[min(i + 1, len(Ts) - 1)], 200)
    j = int(np.argmin([nll_at(t) for t in fine]))
    return float(fine[j])


def temper(pm, T):
    z = np.log(np.asarray(pm) + 1e-12) / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def score(pm, Y):
    Y = np.asarray(Y)
    return dict(acc=float((pm.argmax(1) == Y).mean()),
                ece=float(cal.ece(jnp.asarray(pm), Y)),
                nll=float(cal.nll(jnp.asarray(pm), Y)),
                brier=float(cal.brier(jnp.asarray(pm), Y)))


def val_split(name):
    """Validation split: same distribution as the test split, independent key."""
    if name == "blob":
        X, Y = blob_task.make_data(1000, jax.random.key(VAL_KEY), group="test")
    else:
        X, Y = routing_task.make_data(3000, jax.random.key(VAL_KEY))[:2]
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", default="blob,routing",
                    help="subset re-run; the other task's rows are preserved")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        seeds = seeds[:1]
    run_tasks = args.tasks.split(",")
    os.makedirs(rb.ART, exist_ok=True)
    sfx = "_smoke" if args.smoke else ""
    out_path = os.path.join(rb.ART, f"e1_temp{sfx}.json")

    rows, meta = [], {}
    partial = (set(run_tasks) != {"blob", "routing"}
               or set(seeds) != {0, 1, 2, 3, 4})
    if partial and os.path.exists(out_path):
        # partial re-run: keep every (task, seed) not being recomputed
        old = json.load(open(out_path, encoding="utf-8"))
        rows = [r for r in old.get("rows", [])
                if not (r["task"] in run_tasks and r["seed"] in seeds)]
        meta = old.get("meta", {})
        print(f"partial re-run of tasks={run_tasks} seeds={seeds}: kept "
              f"{len(rows)} rows", flush=True)
    t0 = time.time()

    def _dump():
        with open(out_path, "w") as f:
            json.dump(dict(settings=vars(args), val_key=VAL_KEY,
                           n_samp=rb.N_SAMP, sweeps_pred=rb.SWEEPS_PRED,
                           meta=meta, rows=rows), f, indent=1, default=float)

    def finish(pm_va, pm_te, Yva, Yte, base):
        """Fit T on val, score raw + tempered test predictive, append the row."""
        T = fit_temperature(pm_va, Yva)
        raw = score(np.asarray(pm_te), Yte)
        ts = score(temper(pm_te, T), Yte)
        base.update(T=T, **raw,
                    **{f"{k}_ts": v for k, v in ts.items() if k != "acc"})
        rows.append(base)
        print(f"  s{base['seed']} {base['arm']:7s} T={T:5.2f}  ECE "
              f"{raw['ece']:.4f}->{ts['ece']:.4f}  NLL {raw['nll']:.4f}->"
              f"{ts['nll']:.4f}  Brier {raw['brier']:.4f}->{ts['brier']:.4f}",
              flush=True)
        _dump()

    for task_fn in (rb.task_blob, rb.task_routing):
        task = task_fn(args.smoke)
        if task["name"] not in run_tasks:
            continue
        name, K, C, D = task["name"], task["K"], task["C"], task["D"]
        Xte, Yte, _rte = task["test"]()
        Xva, Yva = val_split(name)
        meta[name] = dict(K=K, C=C, D=D, ceiling=task.get("ceiling"),
                          chance=task.get("chance"))
        print(f"\n=== {name}: K={K} C={C} D={D}, {len(seeds)} seeds ===",
              flush=True)

        for seed in seeds:
            # same training calls as run_baselines.py
            ebm_rate = 0.5
            for fam in rb.FAMILIES:
                tc = TrainConfig(K=K, C=C, D=D, r=task["r"], seed=seed,
                                 make_data=task["make_data"],
                                 build_lik=task["build_lik"], family=fam,
                                 estimator="sfe-loo", n_step_keys=3,
                                 **task["tc_kw"])
                ts0 = time.time()
                params, _spec, _fw, _X, _Y = train(tc, return_data=True)
                secs = time.time() - ts0
                if fam == "ebm":
                    ebm_rate = ro.mean_gate_rate(params, Xte[:400])
                pm_te, _ = rb.predict_prior(params, Xte, K, D, C)
                pm_va, _ = rb.predict_prior(params, Xva, K, D, C)
                finish(pm_va, pm_te, Yva, Yte,
                       dict(task=name, arm=fam, seed=seed, train_s=secs))

            # gate ablations, dropout keep = the ebm arm's mean gate rate
            for label, keep in (("dropout", float(ebm_rate)), ("allon", 1.0)):
                tckw = task["tc_kw"]
                mp = bl.train_masked(seed, task["build_lik"], task["make_data"],
                                     keep=keep, epochs=tckw["epochs"],
                                     lr=tckw["lr"], wd=tckw["wd"],
                                     clip=tckw["clip"], T=tckw["T"])
                pm_te, _ = rb.predict_masked(mp, keep, Xte, C)
                pm_va, _ = rb.predict_masked(mp, keep, Xva, C)
                finish(pm_va, pm_te, Yva, Yte,
                       dict(task=name, arm=label, seed=seed, train_s=None))

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
