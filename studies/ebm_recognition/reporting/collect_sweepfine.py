"""Validate and aggregate the JSONs of run_fashion_sweepfine.py: paired gibbs(S)
minus enum-ceiling curves per cell and the saturation sweep count, plus an optional
audit of heads reused from the coarse Gibbs study.  Prints to stdout.
"""
from __future__ import annotations
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
SEEDS = (0, 1, 2)
CELLS = ((10, 1), (5, 1), (10, 2), (10, 8))
SGRID = (1, 2, 3, 4, 5, 6, 8, 10, 16, 24)
REUSED = tuple((10, r, s) for r in (1, 8) for s in (2, 6, 24))   # (K, r, S)
METRICS = ("acc", "ece", "nll", "brier", "kl_qp")

rows = []
for s in SEEDS:
    d = json.load(open(os.path.join(ART,
                                    f"fashion_sweepfine_n1000_s{s}.json"),
                       encoding="utf-8"))
    assert d["base"]["estimator"] == "sfe-loo", "estimator not pinned!"
    assert len(d["rows"]) == 44, f"s{s}: {len(d['rows'])} rows (expect 44)"
    rows += d["rows"]

R = {(r["K"], r["r"], r["sweeps"], r["seed"]): r for r in rows}
assert len(R) == 132, "duplicate (K,r,S,seed) rows"
missing = [(k, r, s, sd) for k, r in CELLS for s in (0,) + SGRID for sd in SEEDS
           if (k, r, s, sd) not in R]
assert not missing, f"missing: {missing[:5]}"
bad = [key for key, r in R.items()
       if (r["family"], r["n_chains"]) != (("ebm", 0) if key[2] == 0
                                           else ("ebm-gibbs", 64))]
assert not bad, f"off-design rows: {bad[:5]}"
print("3 JSONs x 44 rows, 4 cells x (10 S + enum) x 3 seeds complete, estimator "
      "pinned, 64 chains -- ALL CHECKS PASS\n")

# reused-head audit, only when the coarse Gibbs study's JSONs are present
G = {}
for s in SEEDS:
    p = os.path.join(ART, f"fashion_gibbs_n1000_s{s}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    for r in d["rows"]:
        if r.get("approach") == "trunk":
            G[(r["K"], r["r"], r["sweeps"], r["seed"])] = r

if G:
    print("=== reused-head audit: sweep-fine minus the coarse Gibbs study, same head ===")
    print(f"{'cell':10s} {'S':>3s}" + "".join(f"{m:>10s}" for m in METRICS))
    worst = 0.0
    for (k, rr, s) in REUSED:
        for sd in SEEDS:
            a, b = R.get((k, rr, s, sd)), G.get((k, rr, s, sd))
            if not b:
                continue
            ds = [a[m] - b[m] for m in METRICS]
            if sd > 0:
                worst = max(worst, *(abs(x) for x in ds[:4]))
            print(f"K={k} r={rr:<3d} {s:>3d}  s{sd}"
                  + "".join(f"{x:>+10.4f}" for x in ds))
    print(f"worst |downstream| mismatch on s1/s2: {worst:.4f}\n")

for m in METRICS:
    print(f"=== {m}: gibbs(S) - enum ceiling, mean over 3 paired seeds ===")
    print(f"{'cell':10s}" + "".join(f"{s:>9d}" for s in SGRID) + f"{'ceil':>9s}")
    for k, rr in CELLS:
        line = f"K={k} r={rr:<3d}"
        for s in SGRID:
            v = [R[(k, rr, s, sd)][m] - R[(k, rr, 0, sd)][m] for sd in SEEDS]
            line += f"{np.mean(v):>+9.4f}"
        ceil = np.mean([R[(k, rr, 0, sd)][m] for sd in SEEDS])
        print(line + f"{ceil:>9.4f}")
    print()

print("=== saturation: first S from which |delta| stays <= 0.002 (acc/ece/brier) "
      "resp. 0.005 (nll/kl) ===")
for k, rr in CELLS:
    line = f"K={k} r={rr:<3d}: "
    for m in METRICS:
        tol = 0.005 if m in ("nll", "kl_qp") else 0.002
        sat = None
        for i, s in enumerate(SGRID):
            ok = all(abs(np.mean([R[(k, rr, t, sd)][m] - R[(k, rr, 0, sd)][m]
                                  for sd in SEEDS])) <= tol for t in SGRID[i:])
            if ok:
                sat = s
                break
        line += f"{m}:S>={sat}  "
    print(line)

secs = sum(r.get("train_s", 0) for r in rows)
print(f"\ntotal fresh train time {secs / 3600:.1f} h")
