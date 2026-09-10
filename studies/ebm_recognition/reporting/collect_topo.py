"""Validate and aggregate fashion_topo_n1000.json from run_fashion_topologies.py:
arm table, ladder per mode, free-mode ordering check.  Prints to stdout.
"""
from __future__ import annotations
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "artifacts",
                   "fashion_topo_n1000.json")
SEEDS = (0, 1, 2)
ARMS = ("mf", "tree-chain", "tree-binary", "tree-star", "mst-warm-mf", "ebm")
FAMS = ("mf", "tree", "tree-binary", "tree-star", "tree-mst", "ebm")
SRCS = ("mf", "tree-chain", "mst-warm-mf", "ebm")

d = json.load(open(SRC, encoding="utf-8"))
rows, ladder = d["rows"], d["ladder"]
assert d["base"]["estimator"] == "sfe-loo", "estimator not pinned!"
assert len(rows) == 18, f"{len(rows)} arm rows (expect 18)"
assert len(ladder) == 144, f"{len(ladder)} ladder rows (expect 144)"
idx = {(r["arm"], r["seed"]): r for r in rows}
assert len(idx) == 18
print("18 arm rows + 144 ladder rows, estimator pinned -- ALL CHECKS PASS")


def m(arm, key):
    v = [idx[(arm, s)][key] for s in SEEDS]
    n = len(v)
    mu = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mu, se


print("\n=== ARM TABLE (mean +- SE over 3 paired seeds) ===")
print(f"{'arm':13s} {'KL(q||p)':>14s} {'corr_post':>10s} {'corr_lost':>10s}")
for a in ARMS:
    kl, kse = m(a, "kl_qp")
    cp, _ = m(a, "corr_post")
    cl, _ = m(a, "corr_lost")
    print(f"{a:13s} {kl:9.4f}+-{kse:.4f} {cp:10.4f} {cl:10.4f}")

lad = {}
for r in ladder:
    lad.setdefault((r["src"], r["fam"], r["mode"]), []).append(r["kl_mean"])

for mode in ("free", "amortized"):
    print(f"\n=== LADDER [{mode}]: mean KL over seeds (rows: fitted family; "
          f"cols: theta) ===")
    print(f"{'fitted':13s}" + "".join(f"{s:>14s}" for s in SRCS))
    for fam in FAMS:
        cells = [np.mean(lad[(s, fam, mode)]) for s in SRCS]
        print(f"{fam:13s}" + "".join(f"{c:14.4f}" for c in cells))

# free-mode ordering: ebm best, mf worst per column
viol = 0
for s_ in SEEDS:
    for src in SRCS:
        v = {f: next(r["kl_mean"] for r in ladder if r["seed"] == s_
                     and r["src"] == src and r["fam"] == f and r["mode"] == "free")
             for f in FAMS}
        if not (v["ebm"] <= min(v[f] for f in FAMS if f != "ebm") + 1e-9):
            viol += 1
            print(f"  EBM not best: seed {s_} theta_{src}: {v}")
        if not (v["mf"] >= max(v[f] for f in ("tree", "tree-binary", "tree-star",
                                              "tree-mst")) - 1e-9):
            viol += 1
            print(f"  mf not worst among trees: seed {s_} theta_{src}")
print(f"\nfree-mode hierarchy violations: {viol}/24 checks")

print("\n=== per-seed kl_qp (in-run) ===")
for a in ARMS:
    v = {s: idx[(a, s)]["kl_qp"] for s in SEEDS}
    print(f"  {a:13s}" + "".join(f"  s{s}={v[s]:.4f}" for s in SEEDS))
