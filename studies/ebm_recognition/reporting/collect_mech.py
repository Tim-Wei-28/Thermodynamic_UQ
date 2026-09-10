"""Validate and aggregate the cifar10 JSONs of run_mech_check.py: arm table, refit
ladder with the ordering and residual gates, Gibbs saturation gate, head times.
Prints to stdout.
"""
from __future__ import annotations
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
SEEDS = (0, 1, 2)
SRCS = ("mf", "tree-chain", "ebm")
FAMS = ("mf", "tree", "ebm")
SWEEPS = (2, 6, 12, 24)

rows, ladder = [], []
for s in SEEDS:
    d = json.load(open(os.path.join(ART, f"cifar10_mech_n4500_s{s}.json"),
                       encoding="utf-8"))
    assert d["base"]["estimator"] == "sfe-loo", "estimator not pinned!"
    assert len(d["rows"]) == 12, f"s{s}: {len(d['rows'])} arm rows (expect 12)"
    assert len(d["ladder"]) == 18, f"s{s}: {len(d['ladder'])} ladder rows (expect 18)"
    rows += d["rows"]
    ladder += d["ladder"]

A = {(r["arm"], r["r"], r["sweeps"], r["seed"]): r for r in rows}
assert len(A) == 36, "duplicate arm rows"
L = {(r["src"], r["fam"], r["mode"], r["seed"]): r["kl_mean"] for r in ladder}
print("3 JSONs x (12 arm + 18 ladder) rows, estimator pinned -- ALL CHECKS PASS\n")


def m(vals):
    mu = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mu, se


print("=== PART A: arms at K=10 r=8 (mean +- SE, 3 seeds) ===")
print(f"{'arm':11s} {'acc':>7s} {'ece':>7s} {'nll':>7s} {'brier':>7s} "
      f"{'acc_ood':>8s} {'KL(q||p)':>12s} {'corr_post':>9s} {'corr_lost':>9s}")
for arm in SRCS:
    rs = [A[(arm, 8, 0, s)] for s in SEEDS]
    kl, kse = m([r["kl_qp"] for r in rs])
    print(f"{arm:11s} {np.mean([r['acc'] for r in rs]):7.4f} "
          f"{np.mean([r['ece'] for r in rs]):7.4f} "
          f"{np.mean([r['nll'] for r in rs]):7.4f} "
          f"{np.mean([r['brier'] for r in rs]):7.4f} "
          f"{np.mean([r.get('acc_ood50', np.nan) for r in rs]):8.4f} "
          f"{kl:7.4f}+-{kse:.4f} "
          f"{np.mean([r['corr_post'] for r in rs]):9.4f} "
          f"{np.mean([r['corr_lost'] for r in rs]):9.4f}")
paired = [A[("ebm", 8, 0, s)]["kl_qp"] - A[("tree-chain", 8, 0, s)]["kl_qp"]
          for s in SEEDS]
print(f"paired ebm - tree KL: mean {np.mean(paired):+.4f}, "
      f"range [{min(paired):+.4f}, {max(paired):+.4f}]")
jq = [A[("ebm", 8, 0, s)] for s in SEEDS]
if "j_mean" in jq[0] or "jq_mean" in jq[0]:
    keys = [k for k in jq[0] if k.startswith(("j_", "jq_", "align"))]
    print("Jq stats:", {k: round(float(np.mean([r[k] for r in jq])), 3)
                        for k in sorted(keys)})

print("\n=== LADDER (KL, mean over seeds; free / amortized) ===")
print(f"{'fitted':6s}" + "".join(f"{'theta_' + s:>16s}" for s in SRCS))
for fam in FAMS:
    line = f"{fam:6s}"
    for src in SRCS:
        fr = np.mean([L[(src, fam, "free", s)] for s in SEEDS])
        am = np.mean([L[(src, fam, "amortized", s)] for s in SEEDS])
        line += f"  {fr:6.4f}/{am:6.4f}"
    print(line)

g1 = g2 = True
for s in SEEDS:
    for src in SRCS:
        v = {f: L[(src, f, "free", s)] for f in FAMS}
        if not (v["ebm"] <= v["tree"] + 1e-9 and v["tree"] <= v["mf"] + 1e-9):
            g1 = False
            print(f"  G1 violation: seed {s} theta_{src}: {v}")
        if v["ebm"] >= 0.01:
            g2 = False
            print(f"  G2 violation: seed {s} theta_{src}: ebm free {v['ebm']:.4f}")
print(f"G1 ladder ordering ebm<=tree<=mf everywhere: {'PASS' if g1 else 'FAIL'}")
print(f"G2 free-EBM residual < 0.01 nats everywhere: {'PASS' if g2 else 'FAIL'}")

print("\n=== GIBBS: paired gibbs(S) - enum ceiling (mean over 3 seeds) ===")
print(f"{'r':>2s} {'S':>3s} {'d_acc':>8s} {'d_ece':>8s} {'d_nll':>8s} "
      f"{'d_brier':>8s} {'d_KL':>8s} {'corr_post':>9s}")
g3 = True
for rr in (1, 8):
    ceil = {s: A[("ebm", rr, 0, s)] for s in SEEDS}
    for S_ in SWEEPS:
        g = {s: A[(f"ebm-g{S_}", rr, S_, s)] for s in SEEDS}
        ds = {k: np.mean([g[s][k] - ceil[s][k] for s in SEEDS])
              for k in ("acc", "ece", "nll", "brier", "kl_qp")}
        cp = np.mean([g[s]["corr_post"] for s in SEEDS])
        print(f"{rr:>2d} {S_:>3d} {ds['acc']:+8.4f} {ds['ece']:+8.4f} "
              f"{ds['nll']:+8.4f} {ds['brier']:+8.4f} {ds['kl_qp']:+8.4f} {cp:9.4f}")
        if S_ >= 6 and (abs(ds["acc"]) > 0.005 or abs(ds["nll"]) > 0.01):
            g3 = False
            print(f"    G3 violation at r={rr} S={S_}")
    print(f"   (enum ceiling r={rr}: acc "
          f"{np.mean([ceil[s]['acc'] for s in SEEDS]):.4f}, corr_post "
          f"{np.mean([ceil[s]['corr_post'] for s in SEEDS]):.4f})")
print(f"G3 saturation flat from S=6 (|d_acc|<=0.005, |d_nll|<=0.01): "
      f"{'PASS' if g3 else 'FAIL'}")

print("\n=== train_s by arm (fresh heads only, seconds) ===")
for key in sorted({(r["arm"], r["r"]) for r in rows}):
    v = [r["train_s"] for r in rows if (r["arm"], r["r"]) == key and r["train_s"] > 1]
    if v:
        print(f"  {key[0]:9s} r={key[1]}: mean {np.mean(v):6.0f}s  "
              f"(n={len(v)}, max {max(v):.0f}s)")
print(f"\ntotal train time: {sum(r['train_s'] for r in rows) / 3600:.1f} h")
