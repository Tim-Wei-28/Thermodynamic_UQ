"""
Stacked composition of the epoch's digital FLOPs per config, from the model with
GPU transcendental constants: share, absolute and Mekko variants.  Sampling sits
at the bottom of each stack; black ticks mark measured GPU values where they
exist.  Writes artifacts/flops_composition{,_abs,_mekko}.png.  No options.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, "..", "lib"))
import flops_model as fm                              # noqa: E402

ART = os.path.join(_H, "..", "artifacts")
AC = os.path.join(_H, "..", "artifacts_cluster")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GROUPS = [("sampling", "#f43f5e"), ("moments/KL", "#14b8a6"),
          ("estimator", "#6366f1"), ("input-dim", "#f59e0b"),
          ("other", "#94a3b8")]
STAMP = "FLOPS MODEL"


def tag(fig):
    fig.text(0.996, 0.942, " ".join(STAMP), color=MUTED, fontsize=15,
             ha="right", va="center", fontweight="bold")


def style(ax, ylabel=None, title=None):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=14, length=3)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=15)
    if title:
        ax.set_title(title, color=INK, fontsize=17, loc="center", pad=10)


def header(fig, title, subtitle=None):
    fig.suptitle(title, color=INK, fontsize=22, x=0.008, ha="left", y=0.965)
    if subtitle:
        fig.text(0.008, 0.895, subtitle, color=INK2, fontsize=15, ha="left")
    tag(fig)


TAU, TAU_S = 104.0, 25.0                               # GPU transcendental constants

BASE = dict(family="tree", estimator="sfe-loo", N=512, D=32, C=3, r=4, T=8, S=12,
            field_kind="mlp", H_f=24, gamma0=0.0, n_chains=64, S_q=12)


def groups_of(cfg):
    """(group -> FLOPs) for one config, from the model with GPU constants."""
    p = fm.predict(**fm.pick({**BASE, **cfg}), tau_transc=TAU, tau_samp=TAU_S)
    fl, off = p["blocks"], p["offload_flops"]
    est = fl["writers"] + fl["softmax"] + fl.get("sfe_glue", 0.0) + fl.get("q_logq", 0.0)
    if cfg.get("family", "tree") != "ebm-gibbs":
        est += fl["q_sample"]
        mom = fl["q_forward"] + fl["kl_terms"] + fl["zz_neg"] + fl["coupling_update"]
    else:
        # under ebm-gibbs the non-offloaded chain machinery (suff-stats, DiCE) counts as moments
        mom = (fl["kl_terms"] + fl["zz_neg"] + fl["coupling_update"]
               + fl["q_forward"] + fl["q_sample"] + fl["gibbs"] - off)
    idm = fl["field"] + fl["readers"] + fl["base_logits"] + fl["head"]
    tot = p["total_flops"]
    oth = tot - off - est - mom - idm
    return dict(zip([g for g, _ in GROUPS], [off, mom, est, idm, oth])), tot


def measured_row(cfg):
    """(flops, flops_s0) from the fetched GPU rows, or None."""
    files = (glob.glob(os.path.join(AC, "flops_01", "*.json"))
             + glob.glob(os.path.join(AC, "flops_02", "*.json")))
    want = {**BASE, **cfg}
    for f in files:
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            if (r.get("backend") == "gpu" and "flops_s0" in r
                    and all(r.get(k, BASE.get(k)) == want[k] for k in
                            ("family", "N", "K", "T", "S", "D", "r", "S_q",
                             "n_chains")) and not r.get("gamma0", 0.0)):
                return r["flops"], r["flops_s0"]
    return None


def measured_share(cfg):
    m = measured_row(cfg)
    return (m[0] - m[1]) / m[0] if m else None


def legend(ax, ticks, loc="center left", anchor=(1.005, 0.5)):
    """Swatches in stack order (top first) plus the measured-line entries (label, linestyle)."""
    from matplotlib.lines import Line2D
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in reversed(GROUPS)]
    labels = [g for g, _ in reversed(GROUPS)]
    for lab, ls in ticks:
        handles.append(Line2D([0], [0], color=INK, lw=3.0, linestyle=ls))
        labels.append(lab)
    ax.legend(handles, labels, loc=loc, bbox_to_anchor=anchor,
              fontsize=14, frameon=False, labelcolor=INK)


def share_figure():
    bars = ([(f"$K$={k}", dict(K=k)) for k in (4, 8, 16, 32, 64)]
            + [(f"$K$={k}", dict(family="ebm-gibbs", K=k)) for k in (4, 8, 16, 32, 48)])
    xs = list(range(5)) + [6.2 + i for i in range(5)]

    fig, ax = plt.subplots(figsize=(14.6, 6.2), facecolor="white")
    for x, (lab, cfg) in zip(xs, bars):
        g, tot = groups_of(cfg)
        y = 0.0
        for name, col in GROUPS:
            h = 100.0 * g[name] / tot
            ax.bar(x, h, bottom=y, width=0.72, color=col,
                   edgecolor="white", linewidth=1.5)
            if name == "sampling" and h > 6:
                ax.text(x, y + h / 2, f"{h:.0f}%", ha="center", va="center",
                        fontsize=16, fontweight="bold", color="#ffffff")
            y += h
        ms = measured_share(cfg)
        if ms is not None:
            ax.plot([x - 0.36, x + 0.36], [100 * ms] * 2, ":", color=INK, lw=3.0)

    style(ax, ylabel="share of digital FLOPs per epoch  (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _ in bars], fontsize=14)
    ax.set_ylim(0, 100)
    ax.annotate("Tree as recognition model", xy=(2.0, -17.5),
                ha="center", fontsize=15, color=INK2, annotation_clip=False)
    ax.annotate("EBM as recognition model", xy=(8.2, -17.5), ha="center",
                fontsize=15, color=INK2, annotation_clip=False)
    legend(ax, [("measured offloadable\nshare (GPU)", ":")])
    header(fig,
           "Composition of FLOPS per Training Epoch under Number of Gates $K$")
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.065, right=0.85)
    out = os.path.join(ART, "flops_composition.png")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print("written:", out)
    for x, (lab, cfg) in zip(xs, bars):
        g, tot = groups_of(cfg)
        parts = "  ".join(f"{n} {100*g[n]/tot:4.1f}%" for n, _ in GROUPS)
        lab_ = lab.replace("$", "").replace(chr(10), " ")
        print(f"  {cfg.get('family','tree'):9s} {lab_:11s} {parts}")


def abs_figure():
    """Absolute FLOPs, one linear scale per panel (stacked bars on a log axis would falsify segment heights)."""
    left = [(f"$K$={k}", dict(K=k)) for k in (4, 8, 16, 32, 64)]
    right = [(f"$K$={k}", dict(family="ebm-gibbs", K=k)) for k in (4, 8, 16, 32, 48)]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.22))
    for ax, bars, title in ((axes[0], left, "Tree as recognition model"),
                            (axes[1], right, "EBM as recognition model")):
        cells = [groups_of(cfg) for _, cfg in bars]
        panel_max = max(t for _, t in cells)
        for x, ((lab, cfg), (g, tot)) in enumerate(zip(bars, cells)):
            y = 0.0
            for name, col in GROUPS:
                ax.bar(x, g[name], bottom=y, width=0.72, color=col,
                       edgecolor="white", linewidth=1.5)
                y += g[name]
            m = measured_row(cfg)
            if m is not None:
                ax.plot([x - 0.36, x + 0.36], [m[0] - m[1]] * 2, ":",
                        color=INK, lw=3.0)
                ax.plot([x - 0.36, x + 0.36], [m[0]] * 2, "-", color=INK, lw=3.0)
            # share label inside the sampling segment when tall enough, else above the bar
            pct = f"{100 * g['sampling'] / tot:.0f}%"
            if g["sampling"] > 0.10 * panel_max:
                ax.text(x, g["sampling"] / 2, pct, ha="center", va="center",
                        fontsize=16, fontweight="bold", color="#ffffff")
            else:
                ytop = max(tot, m[0] if m else tot)
                ax.text(x, ytop + 0.025 * panel_max, pct, ha="center",
                        va="bottom", fontsize=12.5, color=INK2)
        style(ax, ylabel="digital FLOPs per epoch", title=title)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([lab for lab, _ in bars], fontsize=14)
        ax.yaxis.get_offset_text().set_color(MUTED)
        ax.yaxis.get_offset_text().set_fontsize(13)
    legend(axes[1], [("measured offloadable\nFLOPs (GPU)", ":"),
                     ("measured total (GPU)", "-")],
           loc="upper left", anchor=(0.02, 0.98))
    header(fig,
           "Absolute FLOPS per Training Epoch under Number of Gates $K$")
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.065, right=0.985)
    out = os.path.join(ART, "flops_composition_abs.png")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print("written:", out)


def mekko_figure():
    """Mekko chart: bar height = composition, bar width = absolute total, width scaled per panel."""
    left = [(k, dict(K=k)) for k in (4, 8, 16, 32, 64)]
    right = [(k, dict(family="ebm-gibbs", K=k)) for k in (4, 8, 16, 32, 48)]
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.0), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.18))
    for ax, bars, title, unit in (
            (axes[0], left, "Tree as recognition model", 1e8),
            (axes[1], right, "EBM as recognition model", 1e9)):
        tots = []
        for _, cfg in bars:
            g, tot = groups_of(cfg)
            tots.append((g, tot))
        gap = 0.02 * sum(t for _, t in tots)
        x0 = 0.0
        for i, ((k, cfg), (g, tot)) in enumerate(zip(bars, tots)):
            y = 0.0
            for name, col in GROUPS:
                h = 100.0 * g[name] / tot
                ax.bar(x0, h, width=tot, bottom=y, align="edge", color=col,
                       edgecolor="white", linewidth=1.5)
                if name == "sampling" and h > 6 and tot > 0.10 * sum(t for _, t in tots):
                    ax.text(x0 + tot / 2, y + h / 2, f"{h:.0f}%", ha="center",
                            va="center", fontsize=16, fontweight="bold",
                            color="#ffffff")
                y += h
            ms = measured_share(cfg)
            if ms is not None:
                ax.plot([x0, x0 + tot], [100 * ms] * 2, ":", color=INK, lw=3.0)
            # labels staggered so the narrow bars' labels clear each other
            ax.annotate(f"$K$={k}\n{tot / unit:.2f}", (x0 + tot / 2, 100),
                        xytext=(0, 8 + 26 * ((i + 1) % 2 if i < 3 else 0)),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=12, color=INK2, annotation_clip=False)
            x0 += tot + gap
        style(ax, ylabel="share of digital FLOPs per epoch  (%)" if ax is axes[0]
              else None, title=None)
        ax.set_title(title, color=INK, fontsize=17, loc="center", pad=78)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, x0 - gap)
        ax.set_xticks([])
        e = int(f"{unit:.0e}".split("e")[1])
        ax.set_xlabel(f"bar width  =  absolute FLOPs per epoch  (labels in "
                      f"$10^{{{e}}}$)", color=INK2, fontsize=15)
    legend(axes[1], [("measured offloadable\nshare (GPU)", ":")],
           loc="center left", anchor=(1.02, 0.5))
    header(fig, "Composition and Volume of FLOPS per Training Epoch under "
                "Number of Gates $K$")
    fig.subplots_adjust(top=0.70, bottom=0.10, left=0.06, right=0.87)
    out = os.path.join(ART, "flops_composition_mekko.png")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print("written:", out)


def main():
    share_figure()
    abs_figure()
    mekko_figure()


if __name__ == "__main__":
    main()
