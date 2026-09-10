"""
Render the FLOPs model into the thesis tables (thesis E5): symbolic per-block
formulas, the baseline evaluation with measured values where a knob isolates a
block, and the per-driver slopes.  Reads artifacts/flops_measurements.json plus
the GPU grid if present; writes artifacts/flops_table_{symbolic,base,drivers}.md.
"""
from __future__ import annotations
import json
import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, "..", "lib"))

import flops_model as fm                              # noqa: E402

ART = os.path.join(_H, "..", "artifacts")

# Measurement group per block: the knob whose slope measures it jointly.
GROUP = {"gibbs": "direct",
         "writers": "T&r", "softmax": "T&r", "q_sample": "T&r", "q_logq": "T&r",
         "sfe_glue": "T&r",
         "readers": "D&r", "field": "D&r", "base_logits": "D&r"}


def _slope(rows, knob):
    """Mean consecutive-difference slope of measured flops along one sweep."""
    sw = sorted((r for r in rows if r["sweep"] == knob), key=lambda r: r[knob])
    if len(sw) < 2:
        return None
    ds = [(b["flops"] - a["flops"]) / (b[knob] - a[knob]) for a, b in zip(sw, sw[1:])]
    return sum(ds[-2:]) / len(ds[-2:])                # tail pairs: past small-size effects


def main():
    with open(os.path.join(ART, "flops_measurements.json"), encoding="utf-8") as f:
        data = json.load(f)
    base, rows = data["base"], data["rows"]
    brow = next(r for r in rows if r["sweep"] == "base")
    # GPU companion grid, if fetched; it has no r cells.
    gpath = os.path.join(ART, "..", "artifacts_cluster", "flops_01",
                         "flops_measurements_gpu.json")
    grows, gbrow = [], None
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as f:
            grows = json.load(f)["rows"]
        gbrow = next((r for r in grows if r["sweep"] == "base"), None)
    pred = fm.predict(**fm.pick(base))
    total = pred["total_flops"]

    sym = ["| block | drivers | hardware | formula (FLOPs, MAC=2; tau=45 XLA-CPU) |",
           "|---|---|---|---|"]
    for name in pred["blocks"]:
        m = fm.META[name]
        sym.append(f"| {name} | {m['drivers']} | {m['hw']} | `{m['formula']}` |")
    sym.append("")
    sym.append("Sampler ops (hardware-native unit, not FLOPs): site updates = `S*N*K` "
               "(+ `n_chains*S_q*N*K` q-side under ebm-gibbs); one RNG draw per site "
               "update plus `T*N*K` for the estimator samples.")

    # Only the Gibbs block is measured directly (S-companion body); the rest is grouped or model.
    meas = {"gibbs": (brow["gibbs_body"] * base["S"], "direct")} \
        if "gibbs_body" in brow else {}
    gmeas = {"gibbs": gbrow["gibbs_body"] * base["S"]} \
        if gbrow is not None and "gibbs_body" in gbrow else {}
    bl = [f"| block | drivers | hardware | analytic (L0) | % | measured CPU | measured GPU | how |",
          "|---|---|---|---|---|---|---|---|"]
    for name, v in sorted(pred["blocks"].items(), key=lambda kv: -kv[1]):
        m = fm.META[name]
        if name in meas:
            mv, how = meas[name]
            mtxt = f"{mv:.3e}"
        else:
            how = GROUP.get(name, "model")
            mtxt = "(group)" if how != "model" else "--"
        gtxt = f"{gmeas[name]:.3e}" if name in gmeas else ("(group)" if mtxt == "(group)" else "--")
        bl.append(f"| {name} | {m['drivers']} | {m['hw']} | {v:.3e} | "
                  f"{100*v/total:.1f} | {mtxt} | {gtxt} | {how} |")
    gtot = (f"**{gbrow['flops']:.3e}** (kappa {gbrow['flops']/total:.2f})"
            if gbrow is not None else "--")
    bl.append(f"| **TOTAL** |  |  | **{total:.3e}** | 100 | "
              f"**{brow['flops']:.3e}** (kappa {brow['flops']/total:.2f}) | {gtot} |  |")

    # Model slopes are finite-differenced from predict(), so both tables share one model side.
    p0 = fm.pick(base)
    lines = ["Group slopes (FLOPs per +1 of the knob; measured slope x knob value =",
             "that group's share of the baseline totals above):"]
    for knob, label in (("T", "writers, softmax, q_sample, q_logq, sfe_glue"),
                        ("r", "writers, readers"),
                        ("D", "readers, field, base_logits, head/params D-parts")):
        ms = _slope(rows, knob)
        if ms is None:
            continue
        p1 = dict(p0)
        p1[knob] += 1
        an = fm.predict(**p1)["total_flops"] - total
        gs = _slope(grows, knob) if grows else None
        gtxt = f", GPU {gs:.3e} (kappa {gs/an:.2f})" if gs is not None else ""
        lines.append(f"  d/d{knob} {{{label}}}: measured CPU {ms:.3e} vs model "
                     f"{an:.3e} (kappa {ms/an:.2f}){gtxt}")

    tsu = sum(v for n, v in pred["blocks"].items() if fm.META[n]["hw"] == "TSU")
    lines.append("")
    lines.append(f"Scenario A (all GPU): {total:.3e} FLOPs/epoch.")
    lines.append(f"Scenario B (hybrid):  GPU {total-tsu:.3e} FLOPs/epoch + TSU "
                 f"{int(pred['sampler_ops']['site_updates'])} site updates "
                 f"(digital equivalent {tsu:.3e} FLOPs = "
                 f"{100*tsu/total:.1f}% offloaded; under ebm-gibbs the "
                 f"'GPU>TSU' rows move as well and the share becomes dominant).")
    if gbrow is not None and "gibbs" in gmeas:
        lines.append(f"In the GPU backend's own accounting the offloadable share is "
                     f"LARGER: {gmeas['gibbs']:.3e} / {gbrow['flops']:.3e} = "
                     f"{100*gmeas['gibbs']/gbrow['flops']:.1f}% (the GPU expands the "
                     f"per-site transcendentals to ~2.3x the CPU count).")

    drv_txt = drivers_table(base, rows, grows)

    sym_txt = "\n".join(sym)
    base_txt = "\n".join(bl) + "\n\n" + "\n".join(lines)
    for fn, txt in (("flops_table_symbolic.md", sym_txt),
                    ("flops_table_base.md", base_txt),
                    ("flops_table_drivers.md", drv_txt)):
        with open(os.path.join(ART, fn), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print("SYMBOLIC\n" + "=" * 78 + "\n" + sym_txt)
    print("\nBASELINE  " + str(fm.pick(base)) + "\n" + "=" * 78 + "\n" + base_txt)
    print("\nDRIVERS\n" + "=" * 78 + "\n" + drv_txt)
    print(f"\nwritten: artifacts/flops_table_{{symbolic,base,drivers}}.md")


# Presentation strings of predict() (tree, sfe-loo, mlp); the numeric columns are
# finite-differenced from predict() itself.
GRAND = ("FLOPs_epoch = N * [ (27 + 2*S)*K^2  +  1.3*tau*S*K  +  4*(D+C)*(2K-1)\n"
         "                  +  4*H_f*(D+K)  +  2*D*C  +  4*K*r*D\n"
         "                  +  T*( 6*C*K*r + (10+4*tau)*K + 5*C + 6 )  +  10*K ]\n"
         "              + 8*P_params")

DRIVERS = [
    ("S",   "gibbs",                              "N*(2*K^2 + 1.3*tau*K)"),
    ("T",   "writers, softmax, q_sample, q_logq", "N*(6*C*K*r + (10+4*tau)*K + 5*C + 6)"),
    ("K",   "all K-blocks (leading: gibbs, q_forward, kl)",
                                                  "2*N*(27+2*S)*K + N*(1.3*tau*S + 8*(D+C) + 4*H_f + 4*r*D + T*(6*C*r + 10+4*tau) + 10)"),
    ("N",   "every block (full batch)",           "FLOPs_epoch / N   (all linear in N)"),
    ("D",   "head, field, base_logits, readers",  "N*(4*n_pre + 4*H_f + 2*C + 4*K*r)"),
    ("r",   "readers, writers",                   "N*(4*K*D + 6*T*C*K)"),
    ("C",   "head, base_logits, writers, softmax", "N*(4*n_pre + 2*D + 6*T*K*r + 5*T)"),
    ("H_f", "field (mlp)",                        "4*N*(D+K)"),
]


def drivers_table(base, rows, grows=()):
    """One row per driver: blocks driven, slope formula, analytic slope, measured slope per backend."""
    p0 = fm.pick(base)
    f0 = fm.predict(**p0)["total_flops"]
    out = ["The grand total (tree + sfe-loo + mlp field), collected by monomial:", "",
           "```", GRAND, "```", "",
           "NOTE ON UNITS: this table holds SLOPES -- FLOPs per +1 of the driver at",
           "the base point -- while the baseline block table holds TOTALS.  The two",
           "agree by construction: slope x driver value = the block/group total there",
           "(gibbs: 3.05e5 x S=12 = 3.66e6).",
           "",
           "| driver | drives blocks | d(FLOPs)/d(driver) | analytic | CPU meas. | CPU/an | GPU meas. | GPU/an |",
           "|---|---|---|---|---|---|---|---|"]
    for name, blocks, formula in DRIVERS:
        p1 = dict(p0)
        p1[name] = p1[name] + 1          # every driver is in pick(base) by construction
        an = fm.predict(**p1)["total_flops"] - f0
        cells = []
        for rr in (rows, grows):
            ms = _slope(rr, name) if rr else None
            cells += ([f"{ms:.3e}", f"{ms/an:.2f}"] if ms is not None and an
                      else ["--", "--"])
        out.append(f"| {name} | {blocks} | `{formula}` | {an:.3e} | "
                   + " | ".join(cells) + " |")
    out.append("")
    out.append("Ratios ~1.0 = validated scaling.  S, T, D validate on BOTH backends; "
               "the dense (MAC) parts are backend-invariant to ~3% (D-slope: CPU "
               "1.460e5 vs GPU 1.464e5), the transcendental expansions are backend- "
               "AND op-specific (tau is a CPU calibration; the GPU counts the Gibbs "
               "site ~2.3x higher and the estimator samples ~2x lower).  The CPU "
               "K-anomaly (ratio ~3) largely disappears on GPU -- a CPU-lowering "
               "artifact, not the algorithm.  r stays open (no GPU r-sweep in "
               "flops_01; XLA contracts the adapter einsums cheaper than the naive "
               "count).")
    return "\n".join(out)


if __name__ == "__main__":
    main()
