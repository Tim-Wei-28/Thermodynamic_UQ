"""Command-line runner: reads run_config.xlsx, executes the enabled rows through engine.py,
writes runs/<run_id>_<timestamp>/ per row and upserts runs/results.xlsx.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfgmod
import metrics as metmod

DEFAULT_XLSX = os.path.join(HERE, "run_config.xlsx")
DEFAULT_OUT = os.path.join(HERE, "runs")


def _now():
    """yymmdd_hh_mm_ss; seconds included so two runs of one row never share a folder."""
    return datetime.datetime.now().strftime("%y%m%d_%H_%M_%S")


def _jd(o):
    return float(o) if hasattr(o, "item") else str(o)


# --------------------------------------------------------------- per-run
def _free_memory():
    """Drop JAX caches between seeds.  Frees device arrays, not host RSS."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()


def process_run(raw, index, out_root, dry_run, locale, free_memory=False):
    """Parse/validate/resolve one row, then run the engine per seed.
       Returns a list of flat records (one per seed, or one for invalid/skipped)."""
    import engine                                   

    # all four steps run even for an invalid row, so config.json documents the attempt
    norm, coerce_errs = cfgmod.parse_and_normalise(raw)
    errors, warnings = cfgmod.validate(norm)
    errors = coerce_errs + errors
    resolved = cfgmod.resolve(norm)
    rid = cfgmod.run_id_of(norm, index)
    task = norm.get("task")

    # one timestamp per run: it names the folder and goes into every seed's record
    ts = _now()
    run_dir = os.path.join(out_root, f"{rid}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    raw_clean = {k: v for k, v in raw.items() if k != "__row__"}
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": rid, "sheet_row": raw.get("__row__"), "raw": raw_clean,
                   "normalised": norm, "resolved": resolved,
                   "errors": errors, "warnings": warnings}, f, indent=2, default=_jd)

    base = {"run_id": rid, "task": task, "family": norm.get("family"),
            "notes": norm.get("notes"),
            "n_warnings": len(warnings), "out_dir": run_dir, "timestamp": ts}

    if errors or dry_run:
        rec = {**base, "seed": None,
               "status": "invalid" if errors else "validated",
               "error": "; ".join(errors)}
        _emit(run_dir, rec, task, None, locale)
        return [rec]

    records = []
    for seed in resolved["meta"]["seeds"]:
        t0 = time.time()
        rec = {**base, "seed": seed, "status": "", "error": ""}
        try:
            flat = engine.train_and_eval(resolved, run_dir, seed)
            rec.update(flat)
            rec.update(base)                        # run_id/task/family stay authoritative
            rec["seed"] = seed
            rec["status"] = "done"
        except engine.EngineNotImplemented as e:
            rec["status"] = "pending-engine"
            rec["error"] = str(e)
        except Exception as e:                      
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            with open(os.path.join(run_dir, f"error_seed{seed}.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        rec["wall_time_s"] = round(time.time() - t0, 1)
        _emit(run_dir, rec, task, seed, locale)
        records.append(rec)
        if free_memory:
            _free_memory()
    return records


def _emit(run_dir, rec, task, seed, locale):
    """Write the per-run long CSV + a JSON twin."""
    tag = "" if seed is None else f"_seed{seed}"
    metmod.write_run_csv(os.path.join(run_dir, f"metrics{tag}.csv"),
                         rec, task or "", locale=locale)
    with open(os.path.join(run_dir, f"metrics{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=_jd)


# --------------------------------------------------------------- isolated execution
def _run_isolated(selected, args):
    """Run every selected row in its own subprocess; return the collected records.
       Rows are addressed by sheet index because run_ids may repeat, and the records
       come back through a temp file so the child's stdout stays visible."""
    import subprocess
    import tempfile

    records = []
    tmpdir = tempfile.mkdtemp(prefix="thrml_iso_")
    print(f"isolate    : one subprocess per row  ({len(selected)} row(s))")
    for n, (i, raw) in enumerate(selected, 1):
        rid = raw.get("run_id") or f"run_{i:03d}"
        out = os.path.join(tmpdir, f"rec_{i}.json")
        cmd = [sys.executable, os.path.abspath(__file__),
               "--config", args.config, "--out-dir", args.out_dir,
               "--row-index", str(i), "--record-out", out]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.csv_en:
            cmd.append("--csv-en")
        if args.free_memory:
            cmd.append("--free-memory")
        print(f"\n[{n}/{len(selected)}] isolated run {rid} (sheet row {i}) ...")
        try:
            # not captured, so the child's progress stays visible
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            print("\ninterrupted -- stopping the batch (the finished rows are kept).")
            break
        if rc == 0 and os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                records.extend(json.load(f))
            continue
        # the child was killed rather than raising: record the row anyway
        print(f"  !! run {rid} did not complete (exit code {rc})")
        records.append({"run_id": rid, "task": raw.get("task"),
                        "family": raw.get("family"), "notes": raw.get("notes"),
                        "seed": None, "status": "error", "n_warnings": 0,
                        "error": f"isolated subprocess exited with code {rc} "
                                 f"(no record file written -- killed rather than raised; "
                                 f"on a cluster this is usually the OOM reaper or a "
                                 f"wall-clock limit)",
                        "timestamp": _now(), "out_dir": args.out_dir})
    return records


# --------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(description="THRML config-driven runner")
    ap.add_argument("--config", default=DEFAULT_XLSX, help="path to run_config.xlsx")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="output root for run folders")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these run_id(s)")
    ap.add_argument("--all", action="store_true", help="include disabled rows too")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + resolve; write config.json; skip the engine")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="list the selected runs and exit")
    ap.add_argument("--csv-en", action="store_true",
                    help="write CSVs comma/dot (pandas standard) instead of de-DE ';'/','")
    ap.add_argument("--free-memory", action="store_true",
                    help="drop JAX caches + gc after every seed. Fixes the ~50 MB/run "
                         "retention that otherwise grows without bound, at the cost of a "
                         "recompile per run (measured +46%% on repeated identical shapes). "
                         "Use on memory-capped nodes; unnecessary for short local batches.")
    ap.add_argument("--isolate", action="store_true",
                    help="run every selected row in its OWN subprocess. The strongest "
                         "memory guarantee -- the OS reclaims everything, including host "
                         "RSS, which --free-memory cannot shrink -- and an OOM-killed row "
                         "no longer takes the batch with it. Costs one interpreter + JAX "
                         "start per row.")
    # internal, used only by the children --isolate spawns
    ap.add_argument("--row-index", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--record-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    locale = "en" if args.csv_en else "de"

    if not os.path.exists(args.config):
        ap.error(f"config workbook not found: {args.config}")
    headers, rows = cfgmod.load_runs(args.config)

    # --only wins over the enabled cell
    selected = []
    for i, raw in enumerate(rows):
        on = str(raw.get("enabled", "")).strip().lower() in ("yes", "true", "1")
        rid = raw.get("run_id") or f"run_{i:03d}"
        if args.only and rid not in args.only:
            continue
        if not args.all and not args.only and not on:
            continue
        selected.append((i, raw))

    print(f"config     : {args.config}")
    print(f"rows total : {len(rows)}   selected : {len(selected)}"
          + ("  (enabled only)" if not args.all and not args.only else ""))
    if not selected:
        print("nothing to do (no enabled rows; use --all or --only <id>).")
        return 0

    if args.list_only:
        print(f"\n  {'run_id':<22} {'enabled':<8} {'family':<6} {'task':<11} row")
        for i, raw in selected:
            print(f"  {(raw.get('run_id') or f'run_{i:03d}'):<22} "
                  f"{str(raw.get('enabled') or ''):<8} {str(raw.get('family') or ''):<6} "
                  f"{str(raw.get('task') or ''):<11} {raw.get('__row__')}")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)

    # child mode (--isolate): one row, no workbook write; the parent owns results.xlsx
    if args.row_index is not None:
        recs = process_run(rows[args.row_index], args.row_index, args.out_dir,
                           args.dry_run, locale, free_memory=args.free_memory)
        if args.record_out:
            with open(args.record_out, "w", encoding="utf-8") as f:
                json.dump(recs, f, default=_jd)
        return 0

    all_records = []
    if args.isolate:
        all_records = _run_isolated(selected, args)
    else:
        for i, raw in selected:
            print("Starting wiht run " + str(i))
            all_records.extend(process_run(raw, i, args.out_dir, args.dry_run, locale,
                                           free_memory=args.free_memory))

    # one workbook write per batch (upsert by run_id)
    path, n_total, n_kept = metmod.write_results_xlsx(
        os.path.join(args.out_dir, "results.xlsx"), all_records)
    _print_summary(all_records, path, n_total, n_kept)
    return 0


def _print_summary(records, results_path, n_total=0, n_kept=0):
    print(f"\n{'run_id':<22} {'seed':<5} {'status':<15} {'acc_prior':<10} {'nll_cal':<9} "
          f"{'sel@0.5':<8} {'time_s':<7} warn")
    for r in records:
        f = lambda k, w: (f"{r[k]:.3f}".ljust(w) if isinstance(r.get(k), float) else "-".ljust(w))
        print(f"{r['run_id']:<22} {str(r.get('seed') if r.get('seed') is not None else '-'):<5} "
              f"{r['status']:<15} {f('acc_prior',10)} {f('nll_cal',9)} {f('sel_at_0p5',8)} "
              f"{str(r.get('wall_time_s','-')):<7} {r.get('n_warnings', 0)}")
    by = {}
    for r in records:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print(f"\n{len(records)} record(s): " + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    print(f"results.xlsx -> {results_path}   ({n_total} row(s): {len(records)} written, "
          f"{n_kept} kept from previous runs)")
    print("per run      -> runs/<run_id>/metrics_seed<n>.csv  (+ .json, config.json, figure)")


if __name__ == "__main__":
    raise SystemExit(main())
