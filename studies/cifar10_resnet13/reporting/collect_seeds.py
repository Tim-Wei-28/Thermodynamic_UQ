"""Merge the per-seed JSONs into the base table, as mean +/- sample standard deviation, and
say whether a gap survives the seed spread.  Reads artifacts/seed_*.json (--glob).

The `n` column is not the same for every row: the two ResNet rows are functions of the trunk
seed, of which there are few, while the EBM rows vary the head seed.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np

import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import ART                                                # noqa: E402

# Empty: the paper reports Fashion-MNIST and CIFAR-100, not CIFAR-10, so there is no
# published block.  The sibling study's 100-class numbers must not be copied in here.
PAPER = {}

COLS = [("acc", "Acc"), ("ece", "ECE"), ("nll", "NLL"),
        ("acc_ood50", "Acc@.5"), ("ece_ood50", "ECE@.5"), ("nll_ood50", "NLL@.5"),
        ("ece_noise50", "ECE noise"), ("ece_rotate50", "ECE rot"),
        ("h_epi_ood50", "H_epi@.5"), ("acc_w0", "acc(W0)")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(ART, "seed_*_n4500.json"),
                    help="n4500 is the standard budget.  Never mix two budgets in one table: "
                         "the feature scale is fitted on the images the run trains on, so a "
                         "different n_per is a different normalisation, not just more data.")
    args = ap.parse_args(argv)
    files = sorted(glob.glob(args.glob),
                   key=lambda p: int(os.path.basename(p).split("_")[1]))
    if not files:
        raise SystemExit(f"no seed files matched {args.glob}")
    print(f"{len(files)} seed file(s): {[os.path.basename(f) for f in files]}\n")

    acc = {}                          # name -> metric -> list over seeds
    for f in files:
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            d = acc.setdefault(r["name"], {"_n": 0})
            d["_n"] += 1
            for k, _ in COLS:
                if r.get(k) is not None:
                    d.setdefault(k, []).append(float(r[k]))

    w = 22
    head = "Model".ljust(w) + "n   " + "  ".join(h.rjust(15) for _, h in COLS)
    print(head)
    print("-" * len(head))
    for name, d in acc.items():
        line = name[:w].ljust(w) + f"{d['_n']:<4d}"
        for k, _ in COLS:
            v = d.get(k)
            line += "  " + (f"{np.mean(v):.4f}+-{np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"
                            if v else "-").rjust(15)
        print(line)

    print("\n" + "-" * len(head))
    print("Liu et al. (published, Table 3 CIFAR-10 block)".ljust(w + 4)
          + "  ".join(("-" if h not in ("Acc", "ECE", "ECE@.5") else "").rjust(15)
                      for _, h in COLS))
    for nm, p in PAPER.items():
        line = nm[:w].ljust(w) + "-   "
        for k, _ in COLS:
            v = p.get(k)
            line += "  " + (f"{v:.4f}" if v is not None else "-").rjust(15)
        print(line)

    # ---- the questions the seeds were run to answer ---------------------------------
    for target, label in ((("ece", "ece"), "in-distribution ECE vs their BNN (0.0220)"),
                          (("ece_ood50", "ece_ood50"),
                           "OOD ECE at 50 % SVHN vs their BNN (0.1101)")):
        key = target[0]
        ref = PAPER["Liu BNN (sw)"][key]
        print("\n" + "=" * 78)
        print(f"Does the gap in {label} survive the seed spread?")
        print("=" * 78)
        for name, d in acc.items():
            v = d.get(key)
            if not v or len(v) < 2:
                if v:
                    print(f"  {name[:30]:30s} {np.mean(v):.4f} (n={len(v)}) "
                          f"-- too few seeds to judge")
                continue
            m, s = float(np.mean(v)), float(np.std(v, ddof=1))
            se = s / np.sqrt(len(v))
            gap = ref - m
            # One-sample check against a fixed published constant, in standard errors of the
            # mean measured here; the reference has no published spread.  Zero spread means
            # the row is degenerate, so no SE is formed from it.
            if s == 0.0:
                print(f"  {name[:30]:30s} {m:.4f} +- 0 (n={len(v)})  gap {gap:+.4f}  "
                      f"-> DEGENERATE: zero spread over {len(v)} seeds, so no SE exists. "
                      f"The seeds are not disagreeing because there is nothing left "
                      f"stochastic in this row.")
                continue
            z = gap / se
            verdict = "clear" if abs(z) > 3 else "suggestive" if abs(z) > 2 else "within noise"
            print(f"  {name[:30]:30s} {m:.4f} +- {s:.4f} (n={len(v)})  "
                  f"gap {gap:+.4f}  = {z:+.1f} SE  -> {verdict}")
    print("\nNOTE: Liu et al. report no seed spread, so this compares our mean against a")
    print("fixed published constant.  It bounds OUR variance, not theirs.")


if __name__ == "__main__":
    main()
