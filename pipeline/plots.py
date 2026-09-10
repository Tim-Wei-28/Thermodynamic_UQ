"""Per-run figures: the blob triptych (decision map, entropy, BALD from prior-sampled gates,
the gate-activation / coupling maps for routing and the image tasks,
the uncertainty figure and the recognition-tree figure.
"""
from __future__ import annotations
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")               # headless backend: write PNGs, never open a window
import matplotlib.pyplot as plt

import model

NS, SW = 100, 20                    # prior samples / Gibbs sweeps per grid point
NG = 140                            # grid resolution for the raw-coordinate tasks
COLORS = ["#1f78ff", "#a1d99b", "#fdae6b"]   


def smooth(Z, k=2):
    """Box blur (window 2k+1) against MC speckle before contourf; cosmetic only."""
    out = np.zeros_like(Z, dtype=float); n = 0
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            out += np.roll(np.roll(Z, di, axis=0), dj, axis=1); n += 1
    return out / n


# ----------------------------------------------------------------- grids
def _raw_grid(X, margin=1.5):
    lo = np.asarray(X).min(0) - margin
    hi = np.asarray(X).max(0) + margin
    xs = np.linspace(lo[0], hi[0], NG)
    ys = np.linspace(lo[1], hi[1], NG)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], 1)
    return XX, YY, pts, XX.shape


def _predict_grid(params, pts, shape):
    """Prior-sampled per-class probs (H,W,C) and BALD (H,W) over the grid."""
    W0, B, A, J = (params[k] for k in ("W0", "B", "A", "J"))
    fs = params["field"]
    K, D, C = J.shape[0], W0.shape[1], W0.shape[0]
    keys = jax.random.split(jax.random.key(0), pts.shape[0])
    fn = jax.jit(jax.vmap(
        lambda k, x: model.predict_bald(k, W0, B, A, fs, J, x, K, D, C, NS, SW)))
    pms, mis = fn(keys, jnp.asarray(pts))
    return (np.asarray(pms).reshape(*shape, -1), np.asarray(mis).reshape(shape))


# ----------------------------------------------------------------- training overlay
def _train_points(task_name):
    if task_name == "blob":
        import blob_task
        X, Y = blob_task.make_data(80, jax.random.key(7), group="train")
        X = np.asarray(X)
        return X[:, 0], X[:, 1], np.asarray(Y)
    raise ValueError(task_name)


# ----------------------------------------------------------------- triptych
def triptych(params, task_name, out_path, subtitle=""):
    """Write the 3-panel predictive map for a 2-D task.  Returns the path."""
    px, py, _ = _train_points(task_name)
    XX, YY, pts, shape = _raw_grid(np.stack([px, py], 1))
    xlab, ylab = "x1", "x2"

    probs, bald = _predict_grid(params, pts, shape)
    ps = smooth(probs)
    dec = ps.argmax(-1)                         # panel 1
    ent = -(ps * np.log(ps + 1e-12)).sum(-1)    # panel 2
    bal = smooth(bald)                          # panel 3

    tx, ty, tY = _train_points(task_name)
    # figure height follows the data aspect (set_aspect('equal') below)
    xr, yr = XX.max() - XX.min(), YY.max() - YY.min()
    fig_h = max(3.0, 4.6 * (yr / xr) + 1.9)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, fig_h))
    cf0 = ax[0].contourf(XX, YY, dec, levels=[-0.5, 0.5, 1.5, 2.5], colors=COLORS)
    cf1 = ax[1].contourf(XX, YY, ent, levels=np.linspace(0, max(ent.max(), 1e-9), 21), cmap="magma")
    cf2 = ax[2].contourf(XX, YY, bal, levels=np.linspace(0, max(bal.max(), 1e-9), 21), cmap="viridis")
    titles = ["Decision regions  argmax p(y|x)",
              "Predictive entropy  H[p(y|x)]",
              "Epistemic  BALD = H[mean] - mean H"]
    for c, (a, cf) in enumerate(zip(ax, [cf0, cf1, cf2])):
        a.scatter(tx, ty, c=[COLORS[y] for y in tY], s=5, edgecolor="k", linewidth=0.2)
        a.set_aspect("equal")
        a.set_xlim(XX.min(), XX.max()); a.set_ylim(YY.min(), YY.max())
        a.set_xlabel(xlab); a.set_title(titles[c], fontsize=11)
        cb = fig.colorbar(cf, ax=a, fraction=0.046, pad=0.04,
                          ticks=[0, 1, 2] if c == 0 else None)
        if c == 0:
            cb.set_label("class")
    ax[0].set_ylabel(ylab)
    fig.suptitle(f"{subtitle}\nPrediction by sampling gates from the learned prior "
                 f"(Eq. 4.10)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- routing analogue
def routing_maps(M, J, out_path, subtitle="", W=None, gate_group=None, gate_level=None):
    """Routing matrix E[z_k | rule r] next to the learned coupling J; a third panel with
       the Chow-Liu edge weights W for learned trees.  gate_group / gate_level serve tasks
       with grouped or two-level gates; the routing task passes neither."""
    M = np.asarray(M); J = np.asarray(J)
    K, R = M.shape
    lvl = None if gate_level is None else np.asarray(gate_level)
    npan = 3 if W is not None else 2
    fig, ax = plt.subplots(1, npan, figsize=(5.5 * npan, 4.4))
    # panel 1: routing matrix, row k = expert, column r = true rule
    im0 = ax[0].imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax[0].set_xlabel("true active rule"); ax[0].set_ylabel("expert k gate prob")
    ax[0].set_xticks(range(R)); ax[0].set_yticks(range(K))
    if lvl is None:
        ax[0].set_title("Routing matrix  E[z_k | rule r]\n(diagonal = correct routing)",
                        fontsize=10)
    else:
        ax[0].set_yticklabels([("sup g%d" % k) if lvl[k] == 0 else ("sub %d" % k)
                               for k in range(K)], fontsize=8)
        nsup = int((lvl == 0).sum())
        ax[0].axhline(nsup - 0.5, color="w", lw=2)
        ax[0].set_title("Routing matrix  E[z_k | leaf rule r]\n"
                        "(correct = TWO bright cells per column: one superior + one sub)",
                        fontsize=10)
    for i in range(K):
        for j in range(R):
            ax[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       color="white" if M[i, j] < 0.5 else "black", fontsize=9)
    # panel 2: J on a diverging map centred at 0
    v = max(abs(J).max(), 1e-9)
    im1 = ax[1].imshow(J, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[1].set_xticks(range(K)); ax[1].set_yticks(range(K))
    if lvl is None:
        ax[1].set_title("Learned coupling J\n(off-diag < 0 = one-rule-at-a-time)", fontsize=10)
    else:
        # two-level gates: parent-child pairs must be excitatory
        nsup = int((lvl == 0).sum())
        for s in (ax[1].axhline, ax[1].axvline):
            s(nsup - 0.5, color="k", lw=1.5)
        ax[1].set_title("Learned coupling J\n"
                        "(want: superior-sub RED / excite, sub-sub BLUE / inhibit)",
                        fontsize=10)
    for i in range(K):
        for j in range(K):
            ax[1].text(j, i, f"{J[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    ims = [im0, im1]
    if W is not None:
        # panel 3: Chow-Liu edge weights; tick labels carry each gate's group
        Wn = np.asarray(W)
        vmax = max(abs(Wn).max(), 1e-9)
        im2 = ax[2].imshow(Wn, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        labs = [(f"{k}" if gate_group is None else f"{k}\n(g{int(gate_group[k])})")
                for k in range(K)]
        ax[2].set_xticks(range(K)); ax[2].set_yticks(range(K))
        ax[2].set_xticklabels(labs, fontsize=7); ax[2].set_yticklabels(labs, fontsize=7)
        ax[2].set_title("Chow-Liu edge weights W (last refit)\n"
                        "(bright block within a group = recoverable structure)", fontsize=10)
        for i in range(K):
            for j in range(K):
                ax[2].text(j, i, f"{Wn[i, j]:.2f}", ha="center", va="center",
                           color="white" if Wn[i, j] < 0.5 * vmax else "black", fontsize=7)
        ims.append(im2)
    for a, im in zip(ax, ims):
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- image datasets
def mnist_maps(M, J, params, out_path, subtitle="", img_shape=None, n_ctx=0,
               col_label="true class"):
    """Gate x class activation and J for the image tasks, plus one image per gate of its
       effective edit B_k A_k (for the class it boosts most) when the input is still pixels.
       n_ctx skips a leading context block."""
    M = np.asarray(M); J = np.asarray(J)
    K, R = M.shape
    B, A = np.asarray(params["B"]), np.asarray(params["A"])
    # B and A are only meaningful as the product W_k = B_k A_k, shape (C, D)
    Weff = np.einsum("kcr,krd->kcd", B, A)[:, :, n_ctx:] if img_shape else None
    show_imgs = img_shape is not None and Weff is not None \
        and Weff.shape[2] == img_shape[0] * img_shape[1]

    nrow = 2 if show_imgs else 1
    fig = plt.figure(figsize=(11, 4.6 * nrow if not show_imgs else 4.6 + 1.5 * ((K + 4) // 5)))
    gs = fig.add_gridspec(nrow, 1, height_ratios=[1, 0.75] if show_imgs else [1])
    top = gs[0].subgridspec(1, 2)
    a0, a1 = fig.add_subplot(top[0]), fig.add_subplot(top[1])

    im0 = a0.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    a0.set_xlabel(col_label); a0.set_ylabel("expert k gate prob")
    a0.set_xticks(range(R)); a0.set_yticks(range(K))
    a0.set_title(f"Gate activation  E[z_k | {col_label}]\n"
                 f"(diagonal = specialised experts)", fontsize=10)
    if K * R <= 120:                              # still legible
        for i in range(K):
            for j in range(R):
                a0.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.5 else "black", fontsize=6.5)
    v = max(abs(J).max(), 1e-9)
    im1 = a1.imshow(J, cmap="RdBu_r", vmin=-v, vmax=v)
    a1.set_xticks(range(K)); a1.set_yticks(range(K))
    a1.set_title("Learned coupling J\n(off-diag < 0 = one-expert-at-a-time)", fontsize=10)
    if K <= 10:
        for i in range(K):
            for j in range(K):
                a1.text(j, i, f"{J[i, j]:+.1f}", ha="center", va="center", fontsize=6.5)
    for a, im in ((a0, im0), (a1, im1)):
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    if show_imgs:
        ncol = min(5, K)
        nr = (K + ncol - 1) // ncol
        bot = gs[1].subgridspec(nr, ncol)
        for k in range(K):
            ax = fig.add_subplot(bot[k // ncol, k % ncol])
            # the class this expert pushes hardest: the row of W_k with the largest norm
            cstar = int(np.argmax(np.linalg.norm(Weff[k], axis=1)))
            img = Weff[k, cstar].reshape(img_shape)
            m = max(abs(img).max(), 1e-12)
            ax.imshow(img, cmap="RdBu_r", vmin=-m, vmax=m)
            ax.set_title(f"gate {k} -> class {cstar}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.text(0.5, 0.015, "effective adapter edit  B_k A_k  for the class each expert "
                             "boosts most  (red = evidence for, blue = against)",
                 ha="center", fontsize=8.5)
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0.03 if show_imgs else 0, 1, 0.92])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- uncertainty figure
def _reliability(ax, pm, Y, nbins, label, color, marker):
    """One reliability curve, on the same bin partition calibration.ece uses."""
    conf, pred = pm.max(1), pm.argmax(1)
    corr = (pred == Y).astype(float)
    xs, ys, ns = [], [], []
    for b in range(nbins):
        lo, hi = b / nbins, (b + 1) / nbins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(corr[m].mean()); ns.append(int(m.sum()))
    ax.plot(xs, ys, marker + "-", color=color, label=label, lw=1.6, ms=5)
    return np.array(xs), np.array(ys), np.array(ns)


def mnist_uq(out_path, subtitle="", rel=None, gallery=None, nbins=15):
    """Reliability diagram (raw and temperature-scaled, with per-bin counts) plus, when
       gallery holds images, the highest-BALD and the misclassified test images."""
    has_g = gallery is not None and gallery.get("images") is not None
    nrow = 2 if has_g else 1
    fig = plt.figure(figsize=(12, 5.2 + (3.4 if has_g else 0)))
    gs = fig.add_gridspec(nrow, 1, height_ratios=[1, 0.75] if has_g else [1])

    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfectly calibrated")
    _reliability(ax, rel["pm_raw"], rel["Y"], nbins, f"raw  (ECE {rel['ece_raw']:.3f})",
                 "#c0392b", "o")
    xs, ys, ns = _reliability(ax, rel["pm_cal"], rel["Y"], nbins,
                              f"T* = {rel['T']:.2f}  (ECE {rel['ece_cal']:.3f})",
                              "#2471a3", "s")
    ax2 = ax.twinx()                                   # per-bin counts
    ax2.bar(xs, ns, width=1.0 / nbins * 0.8, color="0.85", zorder=0)
    ax2.set_ylabel("test points per bin", color="0.5", fontsize=9)
    ax2.tick_params(axis="y", colors="0.5", labelsize=8)
    ax2.set_zorder(0); ax.set_zorder(1); ax.patch.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("claimed confidence  max p(y|x)")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability -- above the diagonal = UNDER-confident "
                 "(the mixture over gate draws smooths more than the data warrants)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    if has_g:
        imgs, bald = gallery["images"], gallery["bald"]
        pred, true = gallery["pred"], gallery["true"]
        names = gallery.get("class_names")
        n = min(12, len(imgs) // 2)
        order = np.argsort(bald)[::-1]                 # most uncertain first
        wrong = np.array([i for i in order if pred[i] != true[i]], dtype=int)
        top = order[:n]
        n_bad_in_top = int((pred[top] != true[top]).sum())
        row1 = (f"MOST uncertain\n(highest BALD)\n{n_bad_in_top}/{len(top)} wrong", top)
        if len(wrong):
            # median percentile of the errors in the BALD ranking, 100 = top
            rank = {int(i): r for r, i in enumerate(order)}
            pct = 100 * (1 - np.median([rank[int(i)] for i in wrong]) / max(len(order) - 1, 1))
            row2 = (f"MISCLASSIFIED\n(by BALD, {len(wrong)} total)\n"
                    f"median BALD rank: top {pct:.0f}%", wrong[:n])
        else:
            row2 = ("LEAST uncertain\n(no errors to show)", order[::-1][:n])
        picks = [row1, row2]
        sub = gs[1].subgridspec(2, n, hspace=0.65)
        for row, (title, idx) in enumerate(picks):
            for j, i in enumerate(idx):
                a = fig.add_subplot(sub[row, j])
                a.imshow(imgs[i], cmap="gray_r")
                a.set_xticks([]); a.set_yticks([])
                p = names[pred[i]] if names else pred[i]
                t = names[true[i]] if names else true[i]
                ok = pred[i] == true[i]
                a.set_title(f"{p}/{t}\n{bald[i]:.2f}", fontsize=7.5,
                            color=("#1e8449" if ok else "#c0392b"))
                for s in a.spines.values():
                    s.set_color("#1e8449" if ok else "#c0392b")
                    s.set_linewidth(1.4)
                if j == 0:
                    a.set_ylabel(title, fontsize=7.5, rotation=0, ha="right", va="center")
        fig.text(0.5, 0.012, "each tile: predicted / true, and its BALD.  green = correct, "
                             "red = wrong.  A useful uncertainty puts the errors HIGH in the "
                             "BALD ranking -- that is exactly what selective prediction "
                             "exploits.", ha="center", fontsize=8.5)
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0.03 if has_g else 0, 1, 0.93])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- recognition tree figure
def _tree_layout(parents):
    """(positions, depth).  A path is laid out along a line (depth 0); a branching tree
       layered top-down, leaves spread evenly, parents centred over their children."""
    K = len(parents)
    root = list(parents).index(-1)
    kids = {k: [c for c in range(K) if parents[c] == k] for k in range(K)}
    depth = {root: 0}
    frontier = [root]
    while frontier:
        n = frontier.pop(0)
        for c in kids[n]:
            depth[c] = depth[n] + 1
            frontier.append(c)
    if all(len(v) <= 1 for v in kids.values()):         # a path
        return {k: (float(depth[k]), 0.0) for k in range(K)}, 0
    xs, nxt = {}, [0.0]

    def place(n):
        if not kids[n]:
            xs[n] = nxt[0]
            nxt[0] += 1.0
            return xs[n]
        cx = [place(c) for c in kids[n]]
        xs[n] = float(np.mean(cx))
        return xs[n]
    place(root)
    return {k: (xs[k], -depth[k]) for k in range(K)}, max(depth.values())


def _draw_tree(ax, parents, node_val, edge_val, title, node_fmt="{:.2f}"):
    """One tree panel: nodes show the mean marginal, edges the recognition covariance
       (colour = sign, width = magnitude)."""
    if parents is None:
        ax.text(0.5, 0.5, "no tree\n(mean-field warm-up:\nthe run starts without edges)",
                ha="center", va="center", fontsize=9, color="0.4")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        return
    pos, dep = _tree_layout(parents)
    K = len(parents)
    is_path = dep == 0                     # path: edge labels go above the line
    vmax = max(np.abs(list(edge_val.values())).max() if edge_val else 0.0, 1e-9)
    # edge width is scaled per panel; only the printed numbers are comparable across panels
    title = f"{title}\nedge width scaled to this panel (max |Cov| = {vmax:.4f})"
    ns = 430 if K <= 6 else max(150, int(430 - 28 * (K - 6)))     # shrink when crowded
    fs = 7 if K <= 6 else 6
    for k, p in enumerate(parents):
        if p == -1:
            continue
        (x0, y0), (x1, y1) = pos[k], pos[p]
        v = edge_val.get((min(k, p), max(k, p)), 0.0)
        ax.plot([x0, x1], [y0, y1], "-", lw=1 + 4 * abs(v) / vmax,
                color=("#c0392b" if v < 0 else "#2471a3"), zorder=1, alpha=0.85)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.30 if is_path else 0.0), f"{v:+.3f}",
                fontsize=6.5, ha="center", va="center", zorder=3, rotation=90 if is_path else 0,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for k in range(K):
        x, y = pos[k]
        ax.scatter([x], [y], s=ns, c=[[0.93, 0.93, 0.93]], edgecolors="k",
                   linewidths=0.8, zorder=2)
        ax.text(x, y, f"{k}\n{node_fmt.format(node_val[k])}", ha="center", va="center",
                fontsize=fs, zorder=4)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    # head-room for the rotated edge labels of a path
    ax.set_ylim(min(ys) - 0.7, max(ys) + (1.1 if is_path else 0.7))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9)


def _heat(ax, M, title, xlabel, ylabel, cmap, vmin=None, vmax=None, fmt="{:.2f}", fig=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.tick_params(labelsize=7)
    if M.size <= 121:                                  # still legible
        thr = (np.nanmax(np.abs(M)) or 1.0) * 0.55
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if abs(M[i, j]) > thr else "black")
    if fig is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def tree_maps(out_path, subtitle="", panels=None):
    """Recognition-tree figure (tree / tree_max_span): P(gate on | class) for q and for the
       prior (left), the tree with edge covariances before and after training (top), and
       the prior coupling J before and after (bottom).  q is label-conditioned, the prior is not."""
    p = panels
    K = p["J_after"].shape[0]
    fig, ax = plt.subplots(2, 3, figsize=(17.5, 9.4))

    _heat(ax[0, 0], p["cls_recog"], "Recognition q(z|x,y):  P(gate on | class)\n"
                                    "LABEL-CONDITIONED -- q is fed onehot(y)",
          "true class", "gate k", "viridis", 0, 1, fig=fig)
    _heat(ax[1, 0], p["cls_prior"], "Prior p(z|x):  P(gate on | class)\n"
                                    "the deployable one -- the label is NOT used",
          "true class", "gate k", "viridis", 0, 1, fig=fig)

    _draw_tree(ax[0, 1], p["parents_before"], p["mu_before"], p["cov_before"],
               "Recognition tree BEFORE training\nnode = gate / mean marginal, "
               "edge = Cov_q(child, parent)")
    _draw_tree(ax[0, 2], p["parents_after"], p["mu_after"], p["cov_after"],
               "Recognition tree AFTER training\n"
               + ("topology LEARNED (Chow-Liu)" if p.get("learned")
                  else "topology fixed -- only the values moved"))

    jm = max(np.abs(p["J_before"]).max(), np.abs(p["J_after"]).max(), 1e-9)
    _heat(ax[1, 1], p["J_before"], "Prior coupling J BEFORE training\n"
                                   "(complete graph: every gate pair, not a tree)",
          "gate j", "gate k", "RdBu_r", -jm, jm, "{:+.2f}", fig=fig)
    _heat(ax[1, 2], p["J_after"], "Prior coupling J AFTER training\n"
                                  "negative off-diagonal = gates inhibit each other",
          "gate j", "gate k", "RdBu_r", -jm, jm, "{:+.2f}", fig=fig)

    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- dispatch
TWO_D = ("blob",)                               # the drawable tasks (D = 2)


def make_figure(task_name, params, out_dir, seed, subtitle="", extra=None):
    """Write the per-run figure; returns its path or None (no figure for this task)."""
    if task_name in TWO_D:
        return triptych(params, task_name,
                        os.path.join(out_dir, f"triptych_seed{seed}.png"), subtitle)
    if task_name == "routing" and extra is not None \
            and extra.get("routing_matrix") is not None:
        return routing_maps(extra["routing_matrix"], params["J"],
                            os.path.join(out_dir, f"routing_maps_seed{seed}.png"), subtitle,
                            W=extra.get("W"), gate_group=extra.get("gate_group"),
                            gate_level=extra.get("gate_level"))
    # fashion and cifar10_resnet share the figure; mnist_maps drops the expert-image row
    # by itself when the input is not pixels (any row with a trunk)
    if task_name in ("fashion", "cifar10_resnet") and extra is not None \
            and extra.get("gate_matrix") is not None:
        return mnist_maps(extra["gate_matrix"], params["J"], params,
                          os.path.join(out_dir, f"{task_name}_maps_seed{seed}.png"), subtitle,
                          img_shape=extra.get("img_shape"), n_ctx=extra.get("n_ctx", 0),
                          col_label=extra.get("col_label", "true class"))
    return None
