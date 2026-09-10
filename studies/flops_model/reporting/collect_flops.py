"""
Compare the analytic FLOPs model with the measured XLA counts.
Reads artifacts/flops_measurements.json and prints the model's block attribution
at the base point, measured vs model slopes per swept knob, and kappa
(measured / model) per cell.  No options.
"""
from __future__ import annotations
import json
import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, "..", "lib"))

import flops_model as fm                              # noqa: E402

ART = os.path.join(_H, "..", "artifacts")


def main():
    with open(os.path.join(ART, "flops_measurements.json"), encoding="utf-8") as f:
        data = json.load(f)
    base, rows = data["base"], data["rows"]

    pred = fm.predict(**fm.pick(base))
    total = pred["total_flops"]
    print("=" * 78)
    print(f"L0 BLOCK ATTRIBUTION at base {fm.pick(base)}")
    print("-" * 78)
    for name, v in sorted(pred["blocks"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:16s} {v:14.3e} FLOPs   {100*v/total:5.1f} %")
    print(f"  {'TOTAL':16s} {total:14.3e} FLOPs   100.0 %")
    print(f"  sampling share (offloadable, as digitally executed): "
          f"{100*pred['sampling_share']:.1f} %")
    print(f"  transcendentals: { {k: int(v) for k, v in pred['transcendentals'].items()} }")
    print(f"  sampler ops:     { {k: int(v) for k, v in pred['sampler_ops'].items()} }")

    NUMERIC = ("S", "T", "K", "N", "D")
    print("\n" + "=" * 78)
    print("SLOPES  d(flops)/d(knob): measured (XLA) vs model  [consecutive differences]")
    print("-" * 78)
    for knob in NUMERIC:
        sw = sorted((r for r in rows if r["sweep"] == knob), key=lambda r: r[knob])
        if len(sw) < 2:
            continue
        print(f"  {knob}-sweep:")
        for a, b in zip(sw, sw[1:]):
            dk = b[knob] - a[knob]
            ms = (b["flops"] - a["flops"]) / dk
            # model recomputed from the row's symbols, so recalibrating constants needs no re-measuring
            md = (fm.predict(**fm.pick(b))["total_flops"]
                  - fm.predict(**fm.pick(a))["total_flops"]) / dk
            print(f"    {knob}={a[knob]:>5} -> {b[knob]:>5}:  measured {ms:12.4e}"
                  f"   model {md:12.4e}   ratio {ms/md if md else float('nan'):6.2f}")

    print("\n" + "=" * 78)
    print("KAPPA = measured / model, per cell")
    print("-" * 78)
    for r in rows:
        mf_ = fm.predict(**fm.pick(r))["total_flops"]
        kap = r["flops"] / mf_ if mf_ else float("nan")
        tag = (f"{r['family']}/{r['estimator']}"
               f" N={r['N']} K={r['K']} T={r['T']} S={r['S']} D={r['D']}"
               f" field={r['field_kind']} g0={r['gamma0']}")
        print(f"  {r['sweep']:10s} {tag:60s} kappa {kap:5.2f}")


if __name__ == "__main__":
    main()
