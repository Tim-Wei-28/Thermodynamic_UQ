"""Validate and aggregate the K x r grid JSONs of run_fashion_kr_gibbs.py (ebm-gibbs
and its tree twin): completeness checks, then paired ebm minus tree differences per
metric and condition.  Prints to stdout.
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
KS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
RS = (1, 2, 3, 4, 6, 8, 10, 16, 24, 36)
SEEDS = (0, 1, 2)
CHAINS = {30: 32, 50: 16}          # 64 elsewhere
METRICS = ("acc", "ece", "nll", "brier")
CONDS = ("", "_ood50")


def load(study, pattern, want_files):
    files = sorted(glob.glob(os.path.join(ART, study, pattern)))
    assert len(files) == want_files, f"{study}: {len(files)} JSONs (expect {want_files})"
    rows = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        assert d["base"]["estimator"] == "sfe-loo", f"{f}: estimator not pinned!"
        rows += d["rows"]
    return rows


ebm = load("", "fashion_krg_n1000_s*.json", 21)
# the tree JSONs also match s*.json, so the glob demands the _tree suffix
tree = load("", "fashion_krg_n1000_s*_tree.json", 3)
assert len(ebm) == 360, f"{len(ebm)} ebm rows"
assert len(tree) == 360, f"{len(tree)} tree rows"

E = {(r["K"], r["r"], r["seed"]): r for r in ebm}
T = {(r["K"], r["r"], r["seed"]): r for r in tree}
assert len(E) == 360 and len(T) == 360, "duplicate (K,r,seed) rows"
missing = [(k, r, s) for k in KS for r in RS for s in SEEDS
           if (k, r, s) not in E or (k, r, s) not in T]
assert not missing, f"unpaired cells: {missing[:5]}"

bad = [k for k, r in E.items()
       if r["family"] != "ebm-gibbs" or r["sweeps"] != 12
       or r["n_chains"] != CHAINS.get(k[0], 64)]
assert not bad, f"ebm rows off-design: {bad[:5]}"
bad = [k for k, r in T.items() if r["family"] != "tree" or r["sweeps"] != 0]
assert not bad, f"tree rows off-design: {bad[:5]}"
print("21 + 3 JSONs, 360 paired cells x 3 seeds, estimator pinned, "
      "sweeps=12 + chain ladder 64/32/16 verified -- ALL CHECKS PASS\n")

for cond, cname in zip(CONDS, ("clean f=0", "letters blend f=0.5")):
    print(f"=== {cname}: EBM - tree, paired over 3 seeds ===")
    print(f"{'metric':7s} {'mean diff':>10s} {'EBM wins':>9s} {'sign-stable':>12s} "
          f"{'  best cell (diff)':<24s}")
    for m in METRICS:
        key = m + cond
        diffs = {}                               # (K, r) -> [per-seed diff]
        for k in KS:
            for r in RS:
                diffs[(k, r)] = [E[(k, r, s)][key] - T[(k, r, s)][key] for s in SEEDS]
        mu = {c: float(np.mean(v)) for c, v in diffs.items()}
        good = (lambda d: d > 0) if m == "acc" else (lambda d: d < 0)
        wins = sum(good(v) for v in mu.values())
        stable = sum(1 for v in diffs.values()
                     if all(good(x) for x in v) or all(not good(x) for x in v))
        stable_win = sum(1 for v in diffs.values() if all(good(x) for x in v))
        best = (max if m == "acc" else min)(mu, key=lambda c: mu[c])
        print(f"{m:7s} {np.mean(list(mu.values())):+10.4f} {wins:>6d}/120 "
              f"{stable_win:>5d}+{stable - stable_win:<3d}   "
              f"K={best[0]:<2d} r={best[1]:<2d} ({mu[best]:+.4f})")
    print()

print("=== mean diff by rank (pooled over K and seeds; acc/ece in points x100) ===")
hdr = f"{'r':>3s}" + "".join(f"{m:>10s}{m + '_b':>10s}" for m in METRICS)
print(hdr + "   (clean, then blend)")
for r in RS:
    line = f"{r:>3d}"
    for m in METRICS:
        sc = 100.0 if m in ("acc", "ece") else 1.0
        for cond in CONDS:
            v = [E[(k, r, s)][m + cond] - T[(k, r, s)][m + cond]
                 for k in KS for s in SEEDS]
            line += f"{np.mean(v) * sc:>+10.3f}"
    print(line)

print("\n=== mean diff by K (pooled over r and seeds) ===")
for k in KS:
    line = f"{k:>3d}"
    for m in METRICS:
        sc = 100.0 if m in ("acc", "ece") else 1.0
        for cond in CONDS:
            v = [E[(k, r, s)][m + cond] - T[(k, r, s)][m + cond]
                 for r in RS for s in SEEDS]
            line += f"{np.mean(v) * sc:>+10.3f}"
    print(line)

print("\n=== absolute clean acc at anchor cells (ebm | tree, seed mean) ===")
for k, r in ((1, 1), (10, 8), (10, 36), (50, 8), (50, 36)):
    ea = np.mean([E[(k, r, s)]["acc"] for s in SEEDS])
    ta = np.mean([T[(k, r, s)]["acc"] for s in SEEDS])
    print(f"  K={k:<2d} r={r:<2d}: {ea:.4f} | {ta:.4f}")

secs_e = sum(r.get("train_s", 0) for r in ebm)
secs_t = sum(r.get("train_s", 0) for r in tree)
print(f"\ntrain time: ebm {secs_e / 3600:.1f} h, tree {secs_t / 3600:.1f} h "
      f"(cached rows count 0)")
print("pairing caveat: s0 shares the shipped trunk exactly; s1/s2 trained fresh "
      "trunks per study (subset + W0 draws identical).")
