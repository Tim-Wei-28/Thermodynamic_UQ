"""
Digital FLOPs per epoch vs K with and without thermodynamic sampling (thesis E5).
Panel 1: absolute FLOPs, model lines and measured GPU markers; panel 2: the
offloadable share.  Also writes the sweep-budget figure.  Reads
artifacts/flops_measurements.json and any artifacts_cluster/flops_0{1,2}/*.json;
writes artifacts/gibbs_scaling{,_S}.png.  No options.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, "..", "lib"))
import flops_model as fm                              # noqa: E402

ART = os.path.join(_H, "..", "artifacts")
AC = os.path.join(_H, "..", "artifacts_cluster")

BLUE, ORANGE = "#2a78d6", "#eb6834"                   # GPU-only, hybrid
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

BASE = dict(family="tree", estimator="sfe-loo", N=512, D=32, C=3, r=4, T=8, S=12,
            field_kind="mlp", H_f=24, gamma0=0.0, n_chains=64, S_q=12)


def load_rows():
    """All measurement rows, CPU grid plus any fetched GPU grids."""
    out = []
    with open(os.path.join(ART, "flops_measurements.json"), encoding="utf-8") as f:
        out += [dict(r, backend="cpu") for r in json.load(f)["rows"]]
    for p in (glob.glob(os.path.join(AC, "flops_01", "*.json"))
              + glob.glob(os.path.join(AC, "flops_02", "*.json"))):
        with open(p, encoding="utf-8") as f:
            out += json.load(f)["rows"]               # rows carry their own backend
    return out


def k_curve(rows, family, backend):
    """K -> (with, without) measured, from K-sweep + base cells at base knobs."""
    pts = {}
    for r in rows:
        if (r.get("backend") != backend or r["family"] != family
                or r["estimator"] != "sfe-loo" or r.get("gamma0", 0.0)
                or (r["N"], r["T"], r["S"], r["D"], r["r"]) !=
                (BASE["N"], BASE["T"], BASE["S"], BASE["D"], BASE["r"])
                # pin the q-side budget too, or the S_q sweep's cells overwrite the K-curve points
                or r.get("S_q", BASE["S_q"]) != BASE["S_q"]
                or r.get("n_chains", BASE["n_chains"]) != BASE["n_chains"]
                or "flops_s0" not in r):
            continue
        pts[r["K"]] = (r["flops"], r["flops_s0"])
    ks = sorted(pts)
    return (np.array(ks), np.array([pts[k][0] for k in ks]),
            np.array([pts[k][1] for k in ks]))


# GPU transcendental constants: per Gibbs site and per estimator sample.
TAU_GPU = 104.0
TAU_SAMP_GPU = 25.0


def model_curve(family, ks):
    w, wo = [], []
    for k in ks:
        p = fm.predict(**{**fm.pick(BASE), "family": family, "K": int(k)},
                       tau_transc=TAU_GPU, tau_samp=TAU_SAMP_GPU)
        w.append(p["total_flops"])
        wo.append(p["total_flops"] - p["offload_flops"])
    return np.array(w), np.array(wo)


def s_curve(rows, family, K, knob):
    """Sweep-budget curve at fixed K: knob ('S' | 'S_q') -> (with, without), everything else at base."""
    other = "S_q" if knob == "S" else "S"
    pts = {}
    for r in rows:
        if (r.get("backend") != "gpu" or r["family"] != family
                or r["estimator"] != "sfe-loo" or r.get("gamma0", 0.0)
                or r["K"] != K
                or (r["N"], r["T"], r["D"], r["r"]) !=
                (BASE["N"], BASE["T"], BASE["D"], BASE["r"])
                or r.get(other, BASE[other]) != BASE[other]
                or r.get("n_chains", BASE["n_chains"]) != BASE["n_chains"]
                or "flops_s0" not in r):
            continue
        pts[r.get(knob, BASE[knob])] = (r["flops"], r["flops_s0"])
    ss = sorted(pts)
    return (np.array(ss), np.array([pts[s][0] for s in ss]),
            np.array([pts[s][1] for s in ss]))


def s_figure(rows):
    """Digital FLOPs vs the mixing budget; the hybrid line is flat by construction."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.patch.set_facecolor(SURF)
    panels = [("tree", "S", np.arange(0, 51), axes[0],
               "tree q: prior mixing budget S"),
              ("ebm-gibbs", "S_q", np.arange(0, 27), axes[1],
               "EBM-q (Gibbs): q-side budget S_q  (72 chains)")]
    for fam, knob, sgrid, ax, title in panels:
        for K, dash in ((8, "-"), (16, "--")):
            w, wo = [], []
            for s in sgrid:
                p = fm.predict(**{**fm.pick(BASE), "family": fam, "K": K,
                                  knob: int(s)}, tau_transc=TAU_GPU, tau_samp=TAU_SAMP_GPU)
                w.append(p["total_flops"])
                wo.append(p["total_flops"] - p["offload_flops"])
            ax.plot(sgrid, w, dash, color=BLUE, lw=2,
                    label=f"all sampling on GPU" if K == 8 else None)
            ax.plot(sgrid, wo, dash, color=ORANGE, lw=2,
                    label=f"sampling on TSU (0 FLOPs)" if K == 8 else None)
            ax.annotate(f"K={K}", (sgrid[-1], w[-1]), textcoords="offset points",
                        xytext=(3, -3), fontsize=8.5, color=INK2)
            mk, mw, mwo = s_curve(rows, fam, K, knob)
            if len(mk):
                m = "o" if K == 8 else "s"
                ax.plot(mk, mw, m, color=BLUE, ms=7, mec=SURF, mew=1.2)
                ax.plot(mk, mwo, m, color=ORANGE, ms=7, mec=SURF, mew=1.2)
        ax.set_ylim(0, None)
        ax.set_xlabel(f"{'Gibbs sweeps ' + knob}", color=INK)
        ax.set_ylabel("digital (GPU) FLOPs per epoch", color=INK)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK)
        ax.set_facecolor(SURF)
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#d8d7d0")
    axes[0].annotate("flat: extra mixing is FREE in digital arithmetic\n"
                     "once sampling is physical",
                     xy=(14, 4.0e6), fontsize=8, color=INK2)
    fig.tight_layout()
    out = os.path.join(ART, "gibbs_scaling_S.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print("written:", out)


def main():
    rows = load_rows()
    s_figure(rows)
    kk = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.patch.set_facecolor(SURF)

    families = [("tree", "tree q", "o", 1.0)]
    if any(r["family"] == "ebm-gibbs" for r in rows):
        families.append(("ebm-gibbs", "EBM-q (Gibbs)", "D", 0.85))

    for fam, flabel, mark, alpha in families:
        mw, mwo = model_curve(fam, kk)
        gk, gw, gwo = k_curve(rows, fam, "gpu")
        meas_max = gk.max() if len(gk) else 0
        solid = kk <= max(meas_max, kk[0])
        for vals, col, lab in ((mw, BLUE, "all sampling on GPU (model)"),
                               (mwo, ORANGE, "sampling on TSU, 0 FLOPs (model)")):
            ax1.plot(kk[solid], vals[solid], "-", color=col, lw=2, alpha=alpha,
                     label=lab if fam == "tree" else None)
            ax1.plot(kk[~solid | (kk == meas_max)], vals[~solid | (kk == meas_max)],
                     "--", color=col, lw=1.6, alpha=0.6 * alpha)
        if len(gk):
            ax1.plot(gk, gw, mark, color=BLUE, ms=7, mec=SURF, mew=1.2)
            ax1.plot(gk, gwo, mark, color=ORANGE, ms=7, mec=SURF, mew=1.2)
        ax1.annotate(flabel, (kk[-1], mw[-1]), textcoords="offset points",
                     xytext=(4, 2), fontsize=8.5, color=INK2)

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(kk)
    ax1.set_xticklabels([str(k) for k in kk])
    ax1.set_xlabel("gates K", color=INK)
    ax1.set_ylabel("digital (GPU) FLOPs per epoch", color=INK)
    su = BASE["S"] * BASE["N"] * 16
    ax1.annotate(f"TSU absorbs S·N·K = {su//1000}k site updates/epoch at K=16\n"
                 "(work changes units, it does not vanish)",
                 xy=(4.2, 4.0e9), fontsize=8, color=INK2)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK)
    ax1.set_title("Digital arithmetic with / without thermodynamic sampling",
                  fontsize=10.5, color=INK)

    # panel 2: offloadable share
    for fam, flabel, mark, alpha in families:
        share = []
        for k in kk:
            p = fm.predict(**{**fm.pick(BASE), "family": fam, "K": int(k)},
                           tau_transc=TAU_GPU, tau_samp=TAU_SAMP_GPU)
            share.append(100 * p["sampling_share"])
        ax2.plot(kk, share, "-", color=INK2, lw=1.6, alpha=alpha)
        ax2.annotate(f"{flabel} (model, GPU τ)", (kk[-1], share[-1]),
                     textcoords="offset points", xytext=(4, -2),
                     fontsize=8.5, color=INK2)
        for backend, col in (("gpu", BLUE), ("cpu", ORANGE)):
            mk, mw_, mwo_ = k_curve(rows, fam, backend)
            if len(mk):
                ax2.plot(mk, 100 * (mw_ - mwo_) / mw_, mark, color=col, ms=7,
                         mec=SURF, mew=1.2,
                         label=f"{backend.upper()} measured" if fam == "tree" else None)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(kk)
    ax2.set_xticklabels([str(k) for k in kk])
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("gates K", color=INK)
    ax2.set_ylabel("offloadable share of digital FLOPs  [%]", color=INK)
    ax2.legend(loc="center right", fontsize=8.5, frameon=False, labelcolor=INK)
    ax2.annotate("CPU markers sit low: their K-cells carry the CPU-lowering\n"
                 "anomaly (inflated denominator); model lines use the GPU tau",
                 xy=(4.2, 60), fontsize=7.5, color=INK2)
    ax2.set_title("Share the TSU absorbs (saturates: the residue keeps its own K²)",
                  fontsize=10.5, color=INK)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color("#d8d7d0")
    fig.suptitle("", fontsize=1)
    fig.tight_layout()
    out = os.path.join(ART, "gibbs_scaling.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print("written:", out)
    for fam, flabel, _, _ in families:
        for backend in ("gpu", "cpu"):
            mk, w, wo = k_curve(rows, fam, backend)
            for k, a, b in zip(mk, w, wo):
                print(f"  {flabel:14s} {backend} K={k:3d}  with {a:.3e}  "
                      f"without {b:.3e}  share {100*(a-b)/a:.1f}%")


if __name__ == "__main__":
    main()

