"""Raw-versus-calibrated overview table, in-distribution and at the 50 % blend.
Reads results.json (and results_ansatz4.json if present) from ART, writes overview.txt.
Liu et al. (2022) reference rows carry accuracy and ECE only; they do not
temperature-scale, so their T*, NLL_cal and ECE_cal cells stay empty.
"""
from __future__ import annotations
import json
import os


# Path bootstrap; must stay inline because it is what makes `import paths` work.
import os as _os, sys as _sys
_H = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_H, "..", "lib"),
                 _os.path.join(_H, "..", "..", "..", "pipeline", "model")]
from paths import HERE, ART                                          # noqa: E402

COLS = [("name", "Approach", 34, "s"), ("acc", "Acc", 7, "f"),
        ("nll", "NLL_raw", 8, "f"), ("ece", "ECE_raw", 8, "f"),
        ("T", "T*", 6, "f"), ("nll_cal", "NLL_cal", 8, "f"),
        ("ece_cal", "ECE_cal", 8, "f")]

LABEL = {
    "l2_55k": "2  LeNet-5 det. (55k)",
    "l2_10k": "2  LeNet-5 det. (10k)",
    "a1_10k": "1  pixels + EBM (10k)",
    "a3_10k": "3  LeNet+EBM, trained W0 (10k)",
    "a3_55k": "3  LeNet+EBM, trained W0 (55k)",
}
ORDER = ["l2_55k", "l2_10k", "a1_10k", "a3_10k", "a3_55k"]


def _fmt(v, w, kind):
    if v is None:
        return "-".rjust(w)
    if kind == "f":
        return f"{v:.4f}".rjust(w)
    return str(v)[:w].ljust(w)


def _table(rows, title, note):
    out = [title, "=" * 84, note, "",
           "  ".join(h.rjust(w) if k != "name" else h.ljust(w)
                     for k, h, w, _ in COLS),
           "  ".join("-" * w for _, _, w, _ in COLS)]
    for r in rows:
        out.append("  ".join(_fmt(r.get(k), w, kind) for k, _, w, kind in COLS))
    return "\n".join(out)


def _ood_row(row, fraction=0.5):
    """The blended-probe entry of one approach, carrying its in-distribution T* along."""
    for o in row.get("ood", []):
        if abs(o["fraction"] - fraction) < 1e-9:
            d = dict(o)
            d["name"] = row["name"]
            d["T"] = row.get("T")          # in-distribution T*, not refitted on the blend
            return d
    return None


def collect():
    """Return (own rows, reference rows) for the table."""
    with open(os.path.join(ART, "results.json")) as f:
        res = json.load(f)
    by_key = {r["key"]: r for r in res["rows"]}
    rows = []
    for k in ORDER:
        if k in by_key:
            r = dict(by_key[k])
            r["name"] = LABEL[k]
            rows.append(r)
    p4 = os.path.join(ART, "results_ansatz4.json")
    if os.path.exists(p4):
        with open(p4) as f:
            a4 = json.load(f)
        for r in a4["rows"]:
            r = dict(r)
            r["name"] = (f"4  LeNet+EBM, W0 noise a={r['alpha']:.2f} "
                         f"(base acc {r['acc_w0']:.3f})")
            rows.append(r)
    # Liu et al. (2022), raw accuracy and ECE only.
    paper = [dict(name="   Liu DNN (software)", acc=0.9009, ece=0.0328,
                  ece_ood50=0.3435),
             dict(name="   Liu BNN (software)", acc=0.9015, ece=0.0156,
                  ece_ood50=0.0918),
             dict(name="   Liu BNN (spintronic)", acc=0.8970, ece=0.0135,
                  ece_ood50=0.1066)]
    return rows, paper


def main():
    rows, paper = collect()
    ind = rows + [dict(p) for p in paper]
    blocks = [_table(ind, "IN-DISTRIBUTION  (Fashion-MNIST test, 10 000 images)",
                     "ECE_raw is the only column comparable to Liu et al. -- they do not "
                     "temperature-scale.")]
    ood = [r for r in (_ood_row(r) for r in rows) if r is not None]
    for p in paper:
        ood.append(dict(name=p["name"], ece=p.get("ece_ood50")))
    blocks.append(_table(
        ood, "50 % OOD MIX  (Fashion-MNIST blended with EMNIST-Letters, 1 000 pairs)",
        "T* is the IN-DISTRIBUTION temperature, carried over unchanged -- at deployment "
        "there are no labels for the shifted data."))
    txt = "\n\n\n".join(blocks)
    print(txt)
    with open(os.path.join(ART, "overview.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nwritten: {os.path.join(ART, 'overview.txt')}")


if __name__ == "__main__":
    main()
