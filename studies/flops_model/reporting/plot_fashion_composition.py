"""Composition of the epoch's digital FLOPs over K at the Fashion-MNIST configuration,
from the cost model with GPU constants, plus measured ticks where a grid exists.
Writes artifacts/flops_composition_fashion.png and _abs.png.  No options."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, "..", "lib"))
import flops_model as fm  # noqa: E402

ART = os.path.join(_H, "..", "artifacts")
AC = os.path.join(_H, "..", "artifacts_cluster")


def measured_rows():
    """GPU rows of the measured Fashion grid, from either artifacts directory."""
    files = (glob.glob(os.path.join(ART, "flops_fashion.json"))
             + glob.glob(os.path.join(AC, "flops_03", "*.json")))
    rows = []
    for f in files:
        rows += [r for r in json.load(open(f, encoding="utf-8"))["rows"]
                 if r.get("backend") == "gpu" and "flops_s0" in r]
    return rows


def measured_row(fam, K, rows):
    """(flops, flops_s0) for the cell at (family, K), or None."""
    for r in rows:
        if (r["family"] == fam and r["K"] == K
                and all(r.get(k) == FASH[k] for k in ("N", "D", "C", "r", "T", "S"))):
            return r["flops"], r["flops_s0"]
    return None

GROUPS = [("sampling", "#eb6834"), ("moments/KL", "#1baf7a"),
          ("estimator", "#2a78d6"), ("input-dim", "#eda100"),
          ("other", "#b5b4ac")]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
TAU, TAU_S = 104.0, 25.0
FASH = dict(family="tree", estimator="sfe-loo", N=10000, D=85, C=10, r=8, T=8, S=12,
            field_kind="mlp", H_f=24, gamma0=1.0, n_chains=64, S_q=12)
KS_TREE, KS_EBM = (4, 8, 10, 16, 32, 64), (4, 8, 10, 16, 32, 48)


def groups_of(cfg):
    p = fm.predict(**fm.pick({**FASH, **cfg}), tau_transc=TAU, tau_samp=TAU_S)
    fl, off = p["blocks"], p["offload_flops"]
    est = fl["writers"] + fl["softmax"] + fl.get("sfe_glue", 0.0) + fl.get("q_logq", 0.0)
    if cfg.get("family", "tree") != "ebm-gibbs":
        est += fl["q_sample"]
        mom = fl["q_forward"] + fl["kl_terms"] + fl["zz_neg"] + fl["coupling_update"]
    else:
        mom = (fl["kl_terms"] + fl["zz_neg"] + fl["coupling_update"]
               + fl["q_forward"] + fl["q_sample"] + fl["gibbs"] - off)
    idm = fl["field"] + fl["readers"] + fl["base_logits"] + fl["head"]
    tot = p["total_flops"]
    oth = tot - off - est - mom - idm
    return dict(zip([g for g, _ in GROUPS], [off, mom, est, idm, oth])), tot


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ax.spines.values():
        s.set_color("#d8d7d0")


def panels(relative, out):
    rows = measured_rows()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=relative)
    fig.patch.set_facecolor(SURF)
    for ax, ks, fam, title in ((axes[0], KS_TREE, "tree", "tree q  (chain)"),
                               (axes[1], KS_EBM, "ebm-gibbs", "EBM-q (Gibbs)")):
        for x, k in enumerate(ks):
            g, tot = groups_of(dict(family=fam, K=k))
            y = 0.0
            for name, col in GROUPS:
                h = 100.0 * g[name] / tot if relative else g[name]
                ax.bar(x, h, bottom=y, width=0.72, color=col, edgecolor=SURF, linewidth=1.5)
                if relative and name == "sampling" and h > 6:
                    ax.text(x, y + h / 2, f"{h:.0f}%", ha="center", va="center",
                            fontsize=8, color="#ffffff")
                y += h
            m = measured_row(fam, k, rows)
            if m is not None:                          # black ticks = measured
                f, f0 = m
                if relative:
                    ax.plot([x - 0.36, x + 0.36], [100 * (f - f0) / f] * 2, "-",
                            color=INK, lw=1.8)
                else:
                    ax.plot([x - 0.36, x + 0.36], [f - f0] * 2, "-", color=INK, lw=1.8)
                    ax.plot([x - 0.36, x + 0.36], [f] * 2, "-", color=INK, lw=1.8)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"K={k}" + ("\n(E2/E4)" if k == 10 else "") for k in ks], fontsize=8.5)
        ax.set_title(title, fontsize=10.5, color=INK)
        style(ax)
    if relative:
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("share of digital FLOPs per epoch  [%]", color=INK)
        sup = "Where the epoch's arithmetic goes at the Fashion-MNIST configuration — and what a TSU absorbs"
    else:
        for ax in axes:
            ax.set_ylabel("digital FLOPs per epoch", color=INK)
        sup = "Absolute digital arithmetic per epoch at the Fashion-MNIST configuration  (note the panels' different scales)"
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in GROUPS]
    if relative:
        axes[1].legend(handles, [g for g, _ in GROUPS], loc="center left",
                       bbox_to_anchor=(1.005, 0.5), fontsize=8.5, frameon=False, labelcolor=INK)
    else:
        axes[1].legend(handles, [g for g, _ in GROUPS], loc="upper left", fontsize=8.5,
                       frameon=False, labelcolor=INK)
    note = "N = 10 000, D = 85, C = 10, r = 8, T = 8, S = 12   (model, GPU τ" + \
        ("; black ticks = measured" if rows else "") + ")"
    axes[0].annotate(note,
                     xy=(0.02, 0.95) if not relative else (0.02, -0.2), xycoords="axes fraction",
                     fontsize=8, color=INK2, annotation_clip=False)
    fig.suptitle(sup, fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURF, bbox_inches="tight")
    print("written:", out)


if __name__ == "__main__":
    panels(True, os.path.join(ART, "flops_composition_fashion.png"))
    panels(False, os.path.join(ART, "flops_composition_fashion_abs.png"))
