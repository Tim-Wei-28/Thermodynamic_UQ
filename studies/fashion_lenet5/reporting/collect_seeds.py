"""Merge the per-seed result JSONs into one table with error bars, printed to stdout.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np


# Path bootstrap; must stay inline because it is what makes `import paths` work.
import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import HERE, ART                                          # noqa: E402

# Liu et al. (2022), Table 3, Fashion-MNIST; no NLL is reported for classification.
PAPER = {"Liu DNN (sw)": dict(acc=0.9009, ece=0.0328, ece_ood50=0.3435),
         "Liu BNN (sw)": dict(acc=0.9015, ece=0.0156, ece_ood50=0.0918),
         "Liu BNN (spin)": dict(acc=0.8970, ece=0.0135, ece_ood50=0.1066)}

COLS = [("acc", "Acc"), ("ece", "ECE"), ("nll", "NLL"),
        ("acc_ood50", "Acc@.5"), ("ece_ood50", "ECE@.5"), ("nll_ood50", "NLL@.5"),
        ("ece_noise50", "ECE noise"), ("ece_rotate50", "ECE rot"),
        ("h_epi_ood50", "H_epi@.5")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(ART, "seed_*_n1000.json"),
                    help="seed_*_n1000.json = the standard budget; _n5000 = the "
                         "data-budget arm.  Never mix the two in one table.")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no seed files matched {args.glob}")
    print(f"{len(files)} seed file(s): {[os.path.basename(f) for f in files]}\n")

    # name -> metric -> list over seeds
    acc = {}
    for f in files:
        for r in json.load(open(f))["rows"]:
            acc.setdefault(r["name"], {}).setdefault("_n", 0)
            acc[r["name"]]["_n"] += 1
            for k, _ in COLS:
                if r.get(k) is not None:
                    acc[r["name"]].setdefault(k, []).append(float(r[k]))

    w = 26
    head = "Model".ljust(w) + "n  " + "  ".join(h.rjust(15) for _, h in COLS)
    print(head)
    print("-" * len(head))
    for name, d in acc.items():
        line = name[:w].ljust(w) + f"{d['_n']:<3d}"
        for k, _ in COLS:
            v = d.get(k)
            line += "  " + (f"{np.mean(v):.4f}+-{np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"
                            if v else "-").rjust(15)
        print(line)

    print("\n" + "=" * 78)
    print("Does the OOD-ECE gap to Liu's software BNN (0.0918) survive the seed spread?")
    print("=" * 78)
    for name, d in acc.items():
        v = d.get("ece_ood50")
        if not v or len(v) < 2:
            continue
        m, s = float(np.mean(v)), float(np.std(v, ddof=1))
        se = s / np.sqrt(len(v))
        gap = PAPER["Liu BNN (sw)"]["ece_ood50"] - m
        # One-sample distance to a fixed published constant, in standard errors of the
        # seed mean.  Not a two-sample test: the paper reports no spread.
        z = gap / se if se > 0 else float("inf")
        verdict = ("clear" if abs(z) > 3 else "suggestive" if abs(z) > 2 else "within noise")
        print(f"  {name[:30]:30s} {m:.4f} +- {s:.4f} (n={len(v)})  "
              f"gap {gap:+.4f}  = {z:+.1f} SE  -> {verdict}")
    print("\nNOTE: Liu et al. report no seed spread, so this compares our mean against a")
    print("fixed published constant.  It bounds OUR variance, not theirs.")


if __name__ == "__main__":
    main()
