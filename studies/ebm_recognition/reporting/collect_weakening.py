"""Validate and aggregate the backbone-weakening grids (w=0.25 trunk and raw pixels,
ebm-gibbs and tree): plateau, knee and paired ebm minus tree differences per level.
Prints to stdout.
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
CHAINS = {30: 32, 50: 16}
METRICS = ("acc", "ece", "nll", "brier")


def load(study, pattern, want, fam, sweeps):
    files = sorted(glob.glob(os.path.join(ART, study, pattern)))
    assert len(files) == want, f"{study}: {len(files)} JSONs (expect {want})"
    R = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        assert d["base"]["estimator"] == "sfe-loo", f"{f}: estimator!"
        for r in d["rows"]:
            key = (r["K"], r["r"], r["seed"])
            assert key not in R, f"{study}: duplicate {key}"
            assert r["family"] == fam and r["sweeps"] == sweeps, f"{study}: {key} off-design"
            if fam == "ebm-gibbs":
                assert r["n_chains"] == CHAINS.get(r["K"], 64), f"{study}: chains {key}"
            R[key] = r
    missing = [(k, r, s) for k in KS for r in RS for s in SEEDS if (k, r, s) not in R]
    assert not missing, f"{study}: missing {missing[:4]}"
    return R


LEVELS = {
    "w025": (load("", "fashion_w025_krg_n1000_s*.json", 21,
                  "ebm-gibbs", 12),
             load("", "fashion_w025_krg_n1000_s*_tree.json", 3,
                  "tree", 0)),
    "a1": (load("", "fashion_a1_krg_n1000_s*.json", 24,
                "ebm-gibbs", 12),
           load("", "fashion_a1_krg_n1000_s*_tree.json", 9,
                "tree", 0)),
}
print("11: 21 / 12: 3 / 13: 24 / 14: 9 JSONs, each level a complete paired "
      "360+360 grid, design verified -- ALL CHECKS PASS\n")

for lvl, (E, T) in LEVELS.items():
    print(f"===== level {lvl} =====")
    # plateau: mean acc over K>=8, r>=8
    for name, R in (("ebm", E), ("tree", T)):
        plat = np.mean([R[(k, r, s)]["acc"] for k in KS if k >= 8
                        for r in RS if r >= 8 for s in SEEDS])
        lo = np.mean([R[(1, 1, s)]["acc"] for s in SEEDS])
        print(f"  {name:4s}: plateau (K>=8, r>=8) {plat:.4f} | K=1,r=1 corner {lo:.4f}")
    # knee along r at K=10: first r within 0.5 pts of the K=10 row max
    for name, R in (("ebm", E), ("tree", T)):
        accs = {r: np.mean([R[(10, r, s)]["acc"] for s in SEEDS]) for r in RS}
        top = max(accs.values())
        knee_r = min(r for r in RS if accs[r] >= top - 0.005)
        accK = {k: np.mean([R[(k, 8, s)]["acc"] for s in SEEDS]) for k in KS}
        topK = max(accK.values())
        knee_k = min(k for k in KS if accK[k] >= topK - 0.005)
        print(f"  {name:4s}: r-knee@K=10 -> r={knee_r} (row max {top:.4f}); "
              f"K-knee@r=8 -> K={knee_k} (col max {topK:.4f})")
    for cond, cname in (("", "clean"), ("_ood50", "blend f=0.5")):
        line = f"  ebm-tree ({cname}):"
        for m in METRICS:
            diffs, stable = [], 0
            good = (lambda d: d > 0) if m == "acc" else (lambda d: d < 0)
            wins = 0
            for k in KS:
                for r in RS:
                    v = [E[(k, r, s)][m + cond] - T[(k, r, s)][m + cond]
                         for s in SEEDS]
                    diffs.append(np.mean(v))
                    wins += good(np.mean(v))
                    stable += all(good(x) for x in v)
            line += (f"  {m} {np.mean(diffs):+.4f} ({wins}/120 wins, "
                     f"{stable} stable)")
        print(line)
    print()

# value ranges for shared colour limits in the figures
print("=== ranges (min/max over rows) per level, for the shared z limits ===")
for lvl, (E, T) in LEVELS.items():
    for m in METRICS:
        for cond in ("", "_ood50"):
            v = [R[(k, r, s)][m + cond] for R in (E, T) for k in KS for r in RS
                 for s in SEEDS]
            print(f"  {lvl:4s} {m + cond:12s} [{min(v):.4f}, {max(v):.4f}]")
