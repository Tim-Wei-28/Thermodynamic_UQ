"""Parse / normalise / validate / resolve run rows from run_config.xlsx.

The workbook defines the schema: sheet "Runs" gives the columns, sheet "Optionen" the
allowed values.  A blank cell means "use the path default" (defaults.py).
"""
from __future__ import annotations
import os
import re

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XLSX = os.path.join(HERE, "run_config.xlsx")

_INT_RE   = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

# --------------------------------------------------------------- schema tables
# filled by _bind_schema() from the workbook
KEYS: list = []
OPTIONS: dict = {}
TYPE: dict = {}
ENUM_KEYS: list = []

# columns whose option list holds examples of a comma list, not an enumeration
_LIST_VALUED = ("node_order", "fashion_classes", "c10r_classes")


def _find_header_row(ws):
    # searched, so the sheet may carry title rows above the header
    for r in range(1, 8):
        if str(ws.cell(r, 1).value).strip() == "run_id":
            return r
    raise ValueError("could not find the 'run_id' header row in sheet 'Runs'")


def _read_schema(path):
    """(keys, options) from the workbook: keys = the header row of 'Runs', options = the
       columns of the 'Optionen' sheet (name in row 2, values below it)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Runs"]
    head_row = _find_header_row(ws)
    keys = [c.value for c in next(ws.iter_rows(min_row=head_row, max_row=head_row))]
    keys = [str(k).strip() for k in keys if k is not None and str(k).strip() != ""]
    options = {k: [] for k in keys}
    opt_sheet = next((n for n in ("Optionen", "Options") if n in wb.sheetnames), None)
    if opt_sheet is not None:
        rows = list(wb[opt_sheet].iter_rows(min_row=2, values_only=True))
        names = rows[0] if rows else ()
        for c, name in enumerate(names):
            if name is None or str(name).strip() not in options:
                continue
            vals = [row[c] for row in rows[1:] if c < len(row) and row[c] is not None
                    and str(row[c]).strip() != ""]
            options[str(name).strip()] = [str(v).strip() for v in vals]
    wb.close()
    return keys, options


def _infer_type(key):
    if key == "seed":
        return "seedlist"                              # "0,1,2" -> [0,1,2]
    opts = OPTIONS.get(key, [])
    s = set(opts)
    if s and s <= {"yes", "no"}:
        return "bool"
    if opts and all(_INT_RE.match(o) for o in opts):
        return "int"
    if opts and all(_FLOAT_RE.match(o) for o in opts):
        return "float"
    return "str"


def _bind_schema(path):
    """(Re)build the schema tables from the workbook at `path`."""
    global KEYS, OPTIONS, TYPE, ENUM_KEYS
    KEYS, OPTIONS = _read_schema(path)
    TYPE = {k: _infer_type(k) for k in KEYS}
    ENUM_KEYS = [k for k in KEYS
                 if TYPE[k] == "str" and OPTIONS[k] and k not in _LIST_VALUED]


if os.path.exists(DEFAULT_XLSX):
    _bind_schema(DEFAULT_XLSX)

# task-specific columns; a value set on the wrong task gives a warning
TASK_PARAMS = {
    "blob":       ["blob_sigma", "blob_R_C"],
    "routing":    ["routing_R", "routing_DC", "ctx", "routing_sigma_c"],
    "fashion":    ["fashion_classes", "fashion_trunk", "fashion_trunk_epochs",
                   "fashion_w0_alpha", "fashion_probe", "fashion_ood_fraction",
                   "fashion_ood_n", "fashion_n_samples", "fashion_ood_ladder"],
    "cifar10_resnet": ["c10r_classes", "c10r_trunk", "c10r_trunk_epochs",
                       "c10r_trunk_schedule", "c10r_w0_alpha", "c10r_probe",
                       "c10r_ood_fraction", "c10r_ood_n", "c10r_n_samples",
                       "c10r_ood_ladder"],
}
ALL_TASK_PARAMS = {k for v in TASK_PARAMS.values() for k in v}
TASKS = tuple(TASK_PARAMS)
# recognition families (thesis Ch. 4.4); ebm = exact enumeration, ebm-gibbs = Gibbs chains
FAMILIES = ("tree", "mf", "tree_max_span", "ebm", "ebm-gibbs")
EBM_FAMILIES = ("ebm", "ebm-gibbs")

# ---- effectiveness guard --------------------------------------------------------
# Columns that take effect per task; validate() flags a non-blank cell outside the set.
_ALWAYS_OK = {"run_id", "enabled", "seed", "notes", "family", "task"}
# knobs the engine feeds into TrainConfig on every path
_COMMON = {"T_samples", "head_init_scale", "epochs", "lr", "beta_max", "gibbs_sweeps_S",
           "tau", "node_order", "tree_topology",
           "anneal_frac", "free_bits", "wd", "clip", "warmup",
           "anchor_type", "wd_anchor", "field_kind",
           "estimator",
           # effective only for family=tree_max_span
           "tree_edge_weight", "tree_refit", "tree_struct_warmup",
           "tree_struct_warmup_family"}
# per path: _COMMON plus what that path's engine function reads
EFFECTIVE = {
    "routing": _COMMON | {
        "c_bias", "n_test", "w0_scale", "K_gates", "rank", "adapter_init_scale",
        "j_init", "field_n_hidden", "field_bias_b2", "baselines", "dropout_keep",
        "gamma0", "warm_frac", "ctx", "routing_sigma_c", "routing_R", "routing_DC"},
    "blob": _COMMON | {
        "n_per", "gamma0", "warm_frac", "j_init", "field_n_hidden", "K_gates", "rank",
        "gain", "w0_scale", "blob_sigma", "blob_R_C", "baselines",
        "div", "sparse", "loadbal"},
    "fashion": _COMMON | {
        "c_bias", "n_per", "n_test", "w0_scale", "K_gates", "rank", "adapter_init_scale",
        "j_init", "field_n_hidden", "field_bias_b2", "baselines", "dropout_keep",
        "gamma0", "warm_frac", "div", "sparse", "loadbal", "label_dropout_p",
        "fashion_classes", "fashion_trunk", "fashion_trunk_epochs", "fashion_w0_alpha",
        "fashion_probe", "fashion_ood_fraction", "fashion_ood_n", "fashion_n_samples", "fashion_ood_ladder"},
    "cifar10_resnet": _COMMON | {
        "c_bias", "n_per", "n_test", "w0_scale", "K_gates", "rank", "adapter_init_scale",
        "j_init", "field_n_hidden", "field_bias_b2", "baselines", "dropout_keep",
        "gamma0", "warm_frac", "div", "sparse", "loadbal", "label_dropout_p",
        "c10r_classes", "c10r_trunk", "c10r_trunk_epochs", "c10r_trunk_schedule",
        "c10r_w0_alpha", "c10r_probe", "c10r_ood_fraction", "c10r_ood_n",
        "c10r_n_samples", "c10r_ood_ladder"},
}


def _path_of(norm):
    return norm.get("task")


# --------------------------------------------------------------- 1. load
def load_runs(path):
    """Read sheet 'Runs' -> (headers, rows); rows are {header: value} dicts.
       Also binds the schema tables to this workbook."""
    _bind_schema(path)
    wb = openpyxl.load_workbook(path, data_only=True)   # data_only: values, not formulas
    ws = wb["Runs"]
    head_row = _find_header_row(ws)
    headers = [ws.cell(head_row, c).value for c in range(1, ws.max_column + 1)]
    headers = [h for h in headers if h is not None]
    rows = []
    for r in range(head_row + 1, ws.max_row + 1):
        vals = {headers[c]: ws.cell(r, c + 1).value for c in range(len(headers))}
        if any(v is not None and str(v).strip() != "" for v in vals.values()):
            vals["__row__"] = r          # remember the sheet row number for error messages
            rows.append(vals)
    return headers, rows


# --------------------------------------------------------------- 2. normalise
def _blank(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def parse_and_normalise(raw):
    """Coerce each known column to its type; blank -> None.  Returns (norm, errors)."""
    norm, errors = {}, []
    for key in KEYS:
        if key not in raw:
            norm[key] = None                     # column missing from the sheet entirely
            continue
        v = raw[key]
        if _blank(v):
            norm[key] = None
            continue
        sval = str(v).strip()
        t = TYPE[key]
        try:
            if t == "bool":
                norm[key] = sval.lower() in ("yes", "true", "1")
            elif t == "int":
                norm[key] = int(float(sval))          # tolerate "8" and "8.0"
            elif t == "float":
                norm[key] = float(sval)
            elif t == "seedlist":
                norm[key] = [int(x) for x in sval.split(",") if x.strip() != ""]
            else:
                norm[key] = sval
        except (ValueError, TypeError):
            # keep the raw text so config.json shows what was typed
            norm[key] = sval
            errors.append(f"{key!r}: cannot parse {sval!r} as {t}")
    return norm, errors


# --------------------------------------------------------------- 3. validate
def validate(norm):
    """Return (errors, warnings).  Errors block the run; warnings are advisory."""
    errors, warnings = [], []
    task = norm.get("task")

    if _blank(norm.get("family")):
        errors.append("family is required (" + " | ".join(FAMILIES) + ")")
    elif norm.get("family") not in FAMILIES:
        errors.append(f"family={norm.get('family')!r} is not one of " + " | ".join(FAMILIES))
    # the EBM families have no relaxed sampler
    if norm.get("family") in EBM_FAMILIES and norm.get("estimator") == "gumbel":
        errors.append(f"family={norm.get('family')!r} requires estimator=sfe-loo "
                      f"(no relaxed sampler exists for the energy-based family).")
    if norm.get("family") == "ebm" and (norm.get("K_gates") or 0) > 14:
        errors.append("family='ebm' enumerates all 2^K gate patterns and is limited to "
                      "K_gates <= 14; use family='ebm-gibbs' for larger K.")
    if _blank(task):
        errors.append("task is required (" + " | ".join(TASKS) + ")")
    elif task not in TASKS:
        errors.append(f"task={task!r} is not one of " + " | ".join(TASKS))
    # 'allon' (one merged adapter) needs an engine path that trains it
    _ALLON_OK = ("routing", "fashion", "cifar10_resnet")
    if norm.get("baselines") and "allon" in str(norm["baselines"]) \
            and task not in _ALLON_OK:
        warnings.append("baselines 'allon' (merged adapter) is only defined on "
                        + " / ".join(_ALLON_OK)
                        + "; it is ignored here -- use 'dropout'.")

    # enum membership: warn only, so a typo does not block the batch
    for key in ENUM_KEYS:
        v = norm.get(key)
        if v is not None and v not in OPTIONS[key]:
            warnings.append(f"{key}={v!r} is not in the Optionen list "
                            f"({'|'.join(OPTIONS[key])}); check spelling or add it there")

    # effectiveness guard: warn for any non-blank column that does not reach this task
    if task in EFFECTIVE:
        eff = EFFECTIVE[task] | _ALWAYS_OK
        own_params = set(TASK_PARAMS.get(task, []))
        for key in KEYS:
            v = norm.get(key)
            if v is None or key in eff:
                continue
            if key in ALL_TASK_PARAMS and key not in own_params:
                owner = next(t for t, ks in TASK_PARAMS.items() if key in ks)
                warnings.append(f"{key} is set but task={task!r} (a {owner}-only knob; ignored)")
            else:
                warnings.append(f"{key} is set but has NO EFFECT on task={task!r} "
                                "(the path default is used).")
    # a blank/'auto' estimator resolves to the family default; two families then differ
    # in the estimator as well, so a family comparison needs the column set explicitly
    if task in EFFECTIVE and norm.get("estimator") in (None, "auto"):
        warnings.append(
            f"estimator is blank/'auto' -> FAMILY DEFAULT "
            f"({'sfe-loo' if norm.get('family') in ('tree', 'tree_max_span') else 'gumbel'} "
            f"for family={norm.get('family')!r}). A tree-vs-mf comparison built from blank "
            f"cells is ESTIMATOR-CONFOUNDED, not a family comparison -- set the column "
            f"explicitly (same value on both rows) unless you are reproducing a documented run.")

    # family-specific knobs set on another family: say so rather than ignore them
    if norm.get("family") in ("mf", "tree_max_span") + EBM_FAMILIES \
            and norm.get("node_order") not in (None, "auto"):
        warnings.append("node_order only wires the fixed-chain tree family; it is ignored for "
                        f"family={norm.get('family')!r} "
                        + ("(mean-field has no topology)." if norm.get("family") == "mf"
                           else "(tree_max_span LEARNS the topology via Chow-Liu)."
                           if norm.get("family") == "tree_max_span"
                           else "(the energy-based family is fully connected)."))
    # ---- tree_topology: the fixed shape of the recognition tree
    _topo = norm.get("tree_topology")
    if _topo not in (None, "chain"):
        if norm.get("node_order") not in (None, "auto"):
            warnings.append(
                f"node_order is set together with tree_topology={_topo!r} and is IGNORED: a "
                f"node order is a CHAIN wiring, and {_topo!r} is not a chain. Use "
                f"tree_topology=chain if you want the ordering to take effect.")
        if norm.get("family") == "mf":
            warnings.append("tree_topology is set but family='mf' has no edges at all; "
                            "it is ignored.")
        if norm.get("family") in EBM_FAMILIES:
            warnings.append("tree_topology is set but the energy-based family couples "
                            "every gate pair; it is ignored.")
        if norm.get("family") == "tree_max_span":
            if norm.get("tree_struct_warmup_family") == "mf":
                warnings.append(
                    "tree_topology has NO EFFECT here: with tree_struct_warmup_family='mf' "
                    "the warm-up segment is mean-field, so there is no warm-up tree to "
                    "shape (and the refit builds the topology from scratch afterwards).")
            else:
                warnings.append(
                    f"tree_topology={_topo!r} only sets the WARM-UP shape on "
                    f"family='tree_max_span'; the first Chow-Liu refit replaces it. That is "
                    f"a legitimate ablation (does the learned tree depend on where it "
                    f"started?), but the final topology is NOT {_topo!r} -- read tree_parents.")
        # 'hier' needs a task-declared hierarchy; an error, so no chain is used silently
        if _topo == "hier":
            errors.append(
                "tree_topology='hier' needs a task that declares a gate hierarchy; none of "
                "the tasks here does. Use chain | binary | star | random.")
        if _topo == "random":
            warnings.append(
                "tree_topology='random' draws the tree from the RUN SEED, so each seed gets "
                "a different shape. That is what makes it a control -- but it needs several "
                "seeds to mean anything; a single seed is just one more arbitrary tree.")
        _K = norm.get("K_gates")
        if _K is not None and _K <= 3 and _topo in ("binary", "star"):
            warnings.append(f"with K_gates={_K} the shapes degenerate: 'binary' and 'star' "
                            f"are the same tree (and at K<=2 so is 'chain').")
    # the structure-learning knobs only bite when the topology is learned
    if norm.get("family") != "tree_max_span":
        for k in ("tree_edge_weight", "tree_refit", "tree_struct_warmup",
                  "tree_struct_warmup_family"):
            if norm.get(k) is not None:
                warnings.append(f"{k} only affects family=tree_max_span "
                                f"(ignored for family={norm.get('family')!r}).")
    # a blank warm-up family means the chain warm-up, whose topology the refit largely echoes
    elif norm.get("tree_struct_warmup_family") in (None, "tree"):
        warnings.append(
            "tree_struct_warmup_family is blank/'tree' -> the warm-up imposes the CHAIN, "
            "and the learned topology is largely an ECHO of it. A differing tree_parents "
            "is then NOT evidence that structure was found. Set it to 'mf' for a "
            "structure-free warm-up unless you are reproducing a documented run.")

    # fashion guards, so a bad row fails at --dry-run rather than after trunk training
    if task == "fashion":
        tr = norm.get("fashion_trunk") or "lenet55k"
        if tr not in ("none", "lenet55k", "lenet10k"):
            errors.append(f"fashion_trunk must be none | lenet55k | lenet10k (got {tr!r}).")
        pr = norm.get("fashion_probe") or "letters"
        if pr not in ("letters", "noise", "rotate"):
            errors.append(f"fashion_probe must be letters | noise | rotate (got {pr!r}).")
        al = norm.get("fashion_w0_alpha")
        if al is not None and not 0.0 <= float(al) <= 1.0:
            errors.append(f"fashion_w0_alpha must be in [0,1] (got {al}).")
        fr = norm.get("fashion_ood_fraction")
        if fr is not None and not 0.0 <= float(fr) <= 1.0:
            errors.append(f"fashion_ood_fraction must be in [0,1] (got {fr}).")
        # with no trunk there is no trained head to mix with, so alpha does nothing
        if tr == "none" and al is not None and float(al) != 1.0:
            warnings.append(
                f"fashion_w0_alpha={al} has NO EFFECT with fashion_trunk='none': there is "
                f"no trained pixel head to interpolate towards, so W0 is a weak random map "
                f"regardless. Set fashion_trunk to lenet55k/lenet10k if you meant to vary "
                f"the base map's competence.")
    if task == "cifar10_resnet":
        tr = norm.get("c10r_trunk") or "resnet45k"
        if tr not in ("none", "resnet45k", "resnet10k"):
            errors.append(f"c10r_trunk must be none | resnet45k | resnet10k (got {tr!r}).")
        pr = norm.get("c10r_probe") or "svhn"
        if pr not in ("svhn", "noise", "rotate"):
            errors.append(f"c10r_probe must be svhn | noise | rotate (got {pr!r}).")
        al = norm.get("c10r_w0_alpha")
        if al is not None and not 0.0 <= float(al) <= 1.0:
            errors.append(f"c10r_w0_alpha must be in [0,1] (got {al}).")
        fr = norm.get("c10r_ood_fraction")
        if fr is not None and not 0.0 <= float(fr) <= 1.0:
            errors.append(f"c10r_ood_fraction must be in [0,1] (got {fr}).")
    return errors, warnings


# --------------------------------------------------------------- 4. resolve (semantic -> concrete)
# 'frozen-exact' is not an anchor: it freezes the adapters, so no pull is needed
_ANCHOR_MODE = {                      # anchor_type -> (freeze_adapters, anchor_mode)
    "none":          (False, "none"),
    "frozen-exact":  (True,  "none"),
    "per-gate-soft": (False, "per-gate"),
    "group-mean":    (False, "group-mean"),
    "group-sum":     (False, "group-sum"),
}


def _dims(norm):
    """(C, D, K) from task + overrides.  C is 3 on the synthetic tasks; D follows from the
       task's construction; K defaults per task and may be overridden by K_gates."""
    task = norm.get("task")
    C, K, D = 3, norm.get("K_gates"), None
    if task == "blob":
        D = 2                                          # drawable 2-D task
        K = K or 6                                     # 2 experts per class
    elif task == "routing":
        R, DC = norm.get("routing_R") or 3, norm.get("routing_DC") or 6
        D = R + DC                                     # context block + content block
        K = K or R                                     # one expert per rule
    # fashion / cifar10_resnet: C, D and K come from the task module after configure()
    return C, D, K


def resolve(norm):
    """Semantic knobs -> the nested parameter dict the engine consumes.
       Numeric knobs pass through as-is; None means blank and the engine supplies the default."""
    task = norm.get("task")
    C, D, K = _dims(norm)
    fam = norm.get("family")
    freeze, anchor_mode = _ANCHOR_MODE.get(norm.get("anchor_type") or "none", (False, "none"))
    # tree_max_span shares the tree head width 2K-1
    tree_like = fam in ("tree", "tree_max_span")
    n_pre = None
    if K is not None:                                  # recognition-head width
        n_pre = ((2 * K - 1) if tree_like
                 else (K + K * (K - 1) // 2) if fam in EBM_FAMILIES else K)

    baselines = norm.get("baselines")
    baseline_list = [] if baselines in (None, "none") else baselines.split("+")
    # blank/"auto" -> family default; an explicit value lets mf run under sfe-loo and tree under gumbel
    _est_col = norm.get("estimator")
    if _est_col == "sfe-loo":
        estimator = "sfe-loo"
    elif _est_col == "gumbel":
        estimator = "gumbel-concrete"
    else:                                              # None / "auto" / unknown -> family default
        estimator = "gumbel-concrete" if fam == "mf" else "sfe-loo"

    return {
        "meta": {
            "run_id": norm.get("run_id"),
            "seeds": norm.get("seed") if norm.get("seed") is not None else [0],
            "enabled": bool(norm.get("enabled")),
        },
        "task": {
            "name": task,
            "n_per": norm.get("n_per"),
            "n_test": norm.get("n_test"),
            # only this task's own parameters travel on
            "params": {k: norm.get(k) for k in TASK_PARAMS.get(task, [])},
        },
        "dims": {"C": C, "D": D, "K": K, "rank": norm.get("rank")},
        "w0": {"scale": norm.get("w0_scale")},
        "adapters": {
            "init_scale": norm.get("adapter_init_scale"),
            # frozen adapters get no warm-start noise
            "noise": 0.0 if freeze else norm.get("adapter_noise"),
            "gain": norm.get("gain"),
        },
        "anchoring": {"type": norm.get("anchor_type"), "mode": anchor_mode,
                      "freeze": freeze, "wd_anchor": norm.get("wd_anchor")},
        "recognition": {
            "family": fam, "estimator": estimator,
            "node_order": norm.get("node_order"),
            "topology": norm.get("tree_topology"),
            "T": norm.get("T_samples"), "tau": norm.get("tau"),
            "head_init_scale": norm.get("head_init_scale"),
            "c_bias": norm.get("c_bias"), "n_pre": n_pre,
            # tree_max_span structure learning (None for tree/mf)
            "edge_weight": norm.get("tree_edge_weight"),
            "refit": norm.get("tree_refit"),
            "struct_warmup": norm.get("tree_struct_warmup"),
            "struct_warmup_family": norm.get("tree_struct_warmup_family"),
        },
        "ebm": {"j_init": norm.get("j_init"), "gibbs_S": norm.get("gibbs_sweeps_S")},
        "field": {
            "kind": norm.get("field_kind"), "n_hidden": norm.get("field_n_hidden"),
            "bias_b2": norm.get("field_bias_b2"), "v_scale": norm.get("field_v_scale"),
        },
        "baselines": baseline_list,
        "dropout_keep": norm.get("dropout_keep"),
        "recipe": {k: norm.get(k) for k in
                   ("epochs", "lr", "beta_max", "anneal_frac", "warmup", "free_bits",
                    "wd", "clip", "gamma0", "warm_frac", "label_dropout_p")},
        "moe": {"div": norm.get("div"), "sparse": norm.get("sparse"),
                "loadbal": norm.get("loadbal")},
    }


# --------------------------------------------------------------- convenience
def run_id_of(norm, index):
    rid = norm.get("run_id")
    return rid if rid else f"run_{index:03d}"
