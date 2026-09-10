"""
Least-squares refit of the total-FLOPs formula's monomial coefficients against the
measured totals (tree, sfe-loo, mlp cells), per backend.  Reads
artifacts/flops_measurements.json plus the GPU grid if present; writes
artifacts/flops_formula_fitted.md.  Valid only at the grid's fixed C and H_f.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, "..", "lib"))
ART = os.path.join(_H, "..", "artifacts")

# monomial basis, in presentation order
BASIS = [
    ("N",         lambda r: r["N"]),
    ("N*K",       lambda r: r["N"] * r["K"]),
    ("N*K^2",     lambda r: r["N"] * r["K"] ** 2),
    ("N*D",       lambda r: r["N"] * r["D"]),
    ("N*K*D",     lambda r: r["N"] * r["K"] * r["D"]),
    ("N*K*r*D",   lambda r: r["N"] * r["K"] * r["r"] * r["D"]),
    ("S*N*K",     lambda r: r["S"] * r["N"] * r["K"]),
    ("S*N*K^2",   lambda r: r["S"] * r["N"] * r["K"] ** 2),
    ("T*N",       lambda r: r["T"] * r["N"]),
    ("T*N*K",     lambda r: r["T"] * r["N"] * r["K"]),
    ("T*N*K*r",   lambda r: r["T"] * r["N"] * r["K"] * r["r"]),
]


def _cells(rows):
    """tree + sfe-loo + mlp + gamma0=0 cells, deduplicated by config."""
    seen, out = set(), []
    for r in rows:
        if (r["family"], r["estimator"], r["field_kind"], r["gamma0"]) != \
                ("tree", "sfe-loo", "mlp", 0.0):
            continue
        key = (r["N"], r["K"], r["T"], r["S"], r["D"], r["r"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def fit(rows, tag):
    cells = _cells(rows)
    A = np.array([[f(r) for _, f in BASIS] for r in cells], dtype=np.float64)
    y = np.array([r["flops"] for r in cells], dtype=np.float64)
    # row-normalised least squares: relative residuals, so small cells count
    w = 1.0 / y
    coef, _, rank, _ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    resid = (A @ coef - y) / y
    lines = [f"### {tag}: {len(cells)} cells, basis rank {rank}/{len(BASIS)}",
             "",
             "| monomial | fitted coefficient |", "|---|---|"]
    for (name, _), c in zip(BASIS, coef):
        lines.append(f"| {name} | {c:10.2f} |")
    lines.append("")
    worst = cells[int(np.abs(resid).argmax())]
    lines.append(f"fit quality: max |rel residual| {np.abs(resid).max()*100:.1f} % "
                 f"(at N={worst['N']} K={worst['K']} T={worst['T']} S={worst['S']} "
                 f"D={worst['D']} r={worst['r']}), mean {np.abs(resid).mean()*100:.1f} %")
    if rank < len(BASIS):
        lines.append("")
        lines.append(f"**WARNING — rank {rank}/{len(BASIS)}: the one-knob-at-a-time "
                     "grid leaves the basis collinear, so INDIVIDUAL coefficients are "
                     "NOT interpretable (only their well-determined combinations are; "
                     "negative values are the symptom, not a finding).  Use this "
                     "formula to PREDICT totals inside the swept ranges; for "
                     "interpretable coefficients measure the cross-sweeps "
                     "(KxS, KxD, KxT; r and C variation) first.**")
    terms = "  +  ".join(f"{c:.1f}*{name}" for (name, _), c in zip(BASIS, coef)
                         if abs(c) > 0.5)
    lines.append("")
    lines.append(f"```\nFLOPs_{tag} ~= {terms}\n```")
    return "\n".join(lines), coef, np.abs(resid).max()


def main():
    with open(os.path.join(ART, "flops_measurements.json"), encoding="utf-8") as f:
        cpu = json.load(f)["rows"]
    out = ["# Empirical total-FLOPs formulas (fitted on measured XLA counts)",
           "",
           "Valid at C=3, H_f=24 (constant in every fitted cell -- their factors are",
           "folded into the coefficients).  For extrapolation in C or H_f use the L0",
           "grand formula's structure.  Family tree, estimator sfe-loo, mlp field.",
           ""]
    txt, ccpu, _ = fit(cpu, "CPU")
    out.append(txt)
    gpath = os.path.join(ART, "..", "artifacts_cluster", "flops_01",
                         "flops_measurements_gpu.json")
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as f:
            gpu = json.load(f)["rows"]
        txt, cgpu, _ = fit(gpu, "GPU")
        out.append("")
        out.append(txt)
        out.append("")
        out.append("| monomial | CPU | GPU | GPU/CPU |")
        out.append("|---|---|---|---|")
        for (name, _), a, b in zip(BASIS, ccpu, cgpu):
            rt = f"{b/a:.2f}" if abs(a) > 1e-9 else "--"
            out.append(f"| {name} | {a:10.2f} | {b:10.2f} | {rt} |")
    txt_all = "\n".join(out)
    with open(os.path.join(ART, "flops_formula_fitted.md"), "w", encoding="utf-8") as f:
        f.write(txt_all + "\n")
    print(txt_all)
    print("\nwritten: artifacts/flops_formula_fitted.md")


if __name__ == "__main__":
    main()
