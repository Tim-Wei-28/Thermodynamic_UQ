"""
Measure the FLOPs of the trainer's compiled epoch step with XLA's cost analysis.
Captures the jitted step via TrainConfig.step_out, lowers it with the run's real
shapes and records XLA's counts next to the model prediction; nothing is executed.
Writes artifacts/flops_measurements.json (or flops_<grid>.json).
Options: --grid smoke|full|scaling|smoke2|fashion (default full), --out basename.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

_H = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_H, "..", "lib"),
                os.path.join(_H, "..", "..", "..", "pipeline", "model")]

import jax                                            # noqa: E402
jax.config.update("jax_enable_x64", True)             # match production
import numpy as np                                    # noqa: E402
import flops_model as fm                              # noqa: E402
from trainer import TrainConfig, train                # noqa: E402

ART = os.path.join(_H, "..", "artifacts")

BASE = dict(N=512, D=32, C=3, K=8, r=4, T=8, S=12, family="tree",
            estimator="sfe-loo", field_kind="mlp", H_f=24, gamma0=0.0,
            S_q=12, n_chains=64)   # q-side sampler budget (ebm-gibbs only)

_EBG = None


def _ebm_gibbs():
    """Lazy import of the ebm-gibbs family module; its SWEEPS / N_CHAINS are set per cell."""
    global _EBG
    if _EBG is None:
        import ebm_recognition_gibbs as ebg           # noqa: E402  (pipeline/model)
        _EBG = ebg
    return _EBG


def _mk_calls(N, D, C, K, r):
    """Shape-correct synthetic data and likelihood builders; values do not matter for a static cost."""
    def make_data(kd):
        kx, ky = jax.random.split(kd)
        return (jax.random.normal(kx, (N, D)),
                jax.random.randint(ky, (N,), 0, C))

    def build_lik(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return dict(W0=0.1 * jax.random.normal(k1, (C, D)),
                    B=0.1 * jax.random.normal(k2, (K, C, r)),
                    A=0.1 * jax.random.normal(k3, (K, r, D)))
    return make_data, build_lik


def _compile_costs(p):
    """Capture + AOT-compile one config; return XLA's cost dict (never executes)."""
    if p["family"] == "ebm-gibbs":
        # SWEEPS / N_CHAINS are baked into the GibbsSpec at trace time, so set them before train.
        ebg = _ebm_gibbs()
        ebg.SWEEPS, ebg.N_CHAINS = p["S_q"], p["n_chains"]
    make_data, build_lik = _mk_calls(p["N"], p["D"], p["C"], p["K"], p["r"])
    hook = []
    tc = TrainConfig(K=p["K"], C=p["C"], D=p["D"], r=p["r"],
                     make_data=make_data, build_lik=build_lik,
                     family=p["family"], estimator=p["estimator"],
                     T=p["T"], S=p["S"],
                     field_kind=p["field_kind"], field_n_hidden=p["H_f"],
                     gamma0=p["gamma0"],
                     epochs=2,                        # shapes are epoch-independent
                     step_out=hook)
    train(tc)
    cap = hook[0]
    ca = cap["step"].lower(cap["carry"], cap["inp"]).compile().cost_analysis()
    if isinstance(ca, (list, tuple)):                 # older jaxlibs: list of dicts
        ca = ca[0]
    return ca


# XLA's cost analysis counts a scan body once regardless of trip count, and model.gibbs
# is a scan of length S.  Each cell therefore gets an S = 0 companion compile:
# corrected = flops(S=0) + S * (flops(S) - flops(S=0)).  main() probes the backend first.
_S0_CACHE = {}


def measure(sweep, correct_scan=True, **kw):
    """Compile one cell (plus its S=0 companion) and return the row."""
    p = {**BASE, **kw}
    t0 = time.time()
    ca = _compile_costs(p)
    row = dict(sweep=sweep, backend=jax.default_backend(),
               **{k: p[k] for k in ("N", "D", "C", "K", "r", "T", "S",
                                    "family", "estimator", "field_kind",
                                    "H_f", "gamma0")})
    row["flops_raw"] = float(ca.get("flops", float("nan")))
    row["transcendentals"] = float(ca.get("transcendentals", float("nan")))
    row["bytes"] = float(ca.get("bytes accessed", float("nan")))
    row["S_q"], row["n_chains"] = p["S_q"], p["n_chains"]
    if correct_scan and p["family"] == "ebm-gibbs":
        # Two scan levels (prior S, q-side S_q): three compiles linearise both; raw(0,0) is the fully offloaded residue.
        k0q = tuple(sorted((k, v) for k, v in p.items() if k != "S"))
        if k0q not in _S0_CACHE:
            _S0_CACHE[k0q] = float(_compile_costs({**p, "S": 0}).get("flops"))
        r0q = _S0_CACHE[k0q]
        k00 = tuple(sorted((k, v) for k, v in p.items() if k not in ("S", "S_q")))
        if k00 not in _S0_CACHE:
            _S0_CACHE[k00] = float(
                _compile_costs({**p, "S": 0, "S_q": 0}).get("flops"))
        r00 = _S0_CACHE[k00]
        b1 = row["flops_raw"] - r0q if p["S"] > 0 else 0.0
        b2 = r0q - r00 if p["S_q"] > 0 else 0.0
        row["flops_s0"], row["gibbs_body"], row["qchain_body"] = r00, b1, b2
        row["flops"] = r00 + p["S"] * b1 + p["S_q"] * b2
    elif correct_scan and p["S"] > 0:
        key0 = tuple(sorted((k, v) for k, v in p.items() if k != "S"))
        if key0 not in _S0_CACHE:
            _S0_CACHE[key0] = float(_compile_costs({**p, "S": 0}).get("flops"))
        f0 = _S0_CACHE[key0]
        body = row["flops_raw"] - f0                  # one sweep's arithmetic
        row["flops_s0"], row["gibbs_body"] = f0, body
        row["flops"] = f0 + p["S"] * body
    else:
        row["flops"] = row["flops_raw"]
    pred = fm.predict(**fm.pick(p))
    row["model_flops"] = pred["total_flops"]
    row["model_sampling_share"] = pred["sampling_share"]
    row["seconds"] = round(time.time() - t0, 1)
    print(f"  {sweep:12s} {row['family']:5s}/{row['estimator']:7s} "
          f"N={row['N']:5d} K={row['K']:3d} T={row['T']:3d} S={row['S']:3d} "
          f"D={row['D']:4d}  flops {row['flops']:.3e}  model {row['model_flops']:.3e} "
          f"kappa {row['flops'] / row['model_flops']:.2f}  ({row['seconds']}s)")
    return row


# One knob per sweep; toggles are two-point sweeps.
SWEEPS = dict(
    S=[dict(S=v) for v in (0, 6, 12, 24)],
    T=[dict(T=v) for v in (2, 8, 16, 32)],
    K=[dict(K=v) for v in (4, 8, 12, 16)],
    r=[dict(r=v) for v in (2, 4, 8, 16)],   # separates the r-proportional writers/readers

    N=[dict(N=v) for v in (256, 512, 1024, 2048)],
    D=[dict(D=v) for v in (16, 32, 64, 128)],
    family=[dict(family=v) for v in ("mf", "tree")],
    estimator=[dict(estimator=v) for v in ("sfe-loo", "relaxed")],
    gamma=[dict(gamma0=v) for v in (0.0, 1.0)],
    field=[dict(field_kind=v) for v in ("linear", "mlp")],
)
SMOKE = dict(S=[dict(S=v) for v in (6, 12)], base=[dict()])

# Growth curves over K for the scaling plot; budget sweeps at two K values.
SCALING = dict(
    K=[dict(K=v) for v in (4, 8, 12, 16, 24, 32, 48, 64)],
    Keg=[dict(family="ebm-gibbs", K=v) for v in (4, 8, 12, 16, 24, 32, 48)],
    Sx=[dict(S=s, K=k) for k in (8, 16) for s in (6, 24, 48)],
    Sq=[dict(family="ebm-gibbs", S_q=q, K=k) for k in (8, 16) for q in (2, 6, 24)],
)
SMOKE2 = dict(K=[dict(K=24)], Keg=[dict(family="ebm-gibbs", K=8)])

# Fashion-MNIST configuration of thesis E2/E4, K swept for both families (thesis E5).
FASHION_BASE = dict(N=10000, D=85, C=10, r=8, T=8, S=12, gamma0=1.0)
FASHION = dict(
    K=[dict(**FASHION_BASE, K=v) for v in (4, 8, 10, 16, 32, 64)],
    Keg=[dict(**FASHION_BASE, family="ebm-gibbs", K=v) for v in (4, 8, 10, 16, 32, 48)],
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=("smoke", "full", "scaling", "smoke2", "fashion"),
                    default="full")
    ap.add_argument("--out", default=None, help="output JSON basename override")
    args = ap.parse_args()
    grid = {"smoke": SMOKE, "full": dict(base=[dict()], **SWEEPS),
            "scaling": dict(base=[dict()], **SCALING),
            "smoke2": SMOKE2, "fashion": FASHION}[args.grid]
    os.makedirs(ART, exist_ok=True)
    print(f"backend: {jax.default_backend()}   x64: on")
    # Probe whether the backend is trip-count blind before measuring.
    f6 = float(_compile_costs({**BASE, "S": 6}).get("flops"))
    f12 = float(_compile_costs({**BASE, "S": 12}).get("flops"))
    trip_blind = abs(f6 - f12) < 1e-6 * max(f6, f12)
    print(f"scan-body probe: flops(S=6)={f6:.4e}  flops(S=12)={f12:.4e}  "
          f"-> trip-count {'IGNORED, applying S-linearisation' if trip_blind else 'counted, raw kept'}")
    rows = []
    for name, cells in grid.items():
        for cell in cells:
            rows.append(measure(name, correct_scan=trip_blind, **cell))
    fn = args.out or ("flops_measurements.json" if args.grid in ("smoke", "full")
                      else f"flops_{args.grid}.json")
    out = os.path.join(ART, fn)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(base=BASE, scan_trip_count_blind=trip_blind, rows=rows), f, indent=1)
    print(f"\n{len(rows)} cells -> {out}")


if __name__ == "__main__":
    main()
