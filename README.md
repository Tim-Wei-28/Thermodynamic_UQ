# Scalable Bayesian Uncertainty Quantification using Thermodynamic Co-Processors

Code accompanying the MSc thesis of the same title (Imperial College London, Imperial-X,
September 2026). The thesis develops the **EBM-gated adapter mixture model**: a frozen
backbone whose last layer is replaced by a family of low-rank adapters, switched on and
off by a binary gate vector that is drawn from an input-conditioned Boltzmann-machine
prior. The Gibbs update of that prior is the operation a discrete thermodynamic sampling
unit performs natively, so all of the model's randomness could be offloaded to such
hardware. Every experiment here runs in simulation on ordinary CPUs/GPUs.

## Repository layout

```
pipeline/                the Excel-driven experiment pipeline (thesis Chapter 5.1)
  runner.py              reads run_config.xlsx, executes the enabled rows, writes results
  engine.py              composes one row into a training run (one function per task)
  config.py              parses / validates / resolves a row (schema read from the workbook)
  defaults.py            what a blank cell means, per task
  metrics.py, plots.py   result registry, per-run CSV / results.xlsx, per-run figures
  run_config.xlsx        the run configuration workbook (sheets Runs, Optionen, Parameters)
  runs/results.xlsx      the result library of every run behind the thesis (6 059 rows)
  model/                 the model itself, shared by the pipeline and the studies
    trainer.py           the one training loop (Algorithm 1)
    recognition.py       the recognition-family registry: mf | tree | tree_max_span | ebm | ebm-gibbs
    tree_recognition.py, ebm_recognition.py, ebm_recognition_gibbs.py   the families
    model.py, field.py, likelihood.py, estimators.py, regularisers.py, anchoring.py
    readouts.py, calibration.py, ood_metrics.py, baselines.py
    blob_task.py, routing_task.py, fashion_task.py, cifar10_resnet_task.py   the data
    lenet.py, resnet.py  the convolutional trunks (LeNet-5, ResNet-13)
studies/                 the standalone experiment scripts that produced the thesis numbers
  e1_artificial/         E1: blob and routing tasks, recognition-family ladder
  fashion_lenet5/        E2/E3 on Fashion-MNIST behind a LeNet-5 trunk, plus the baselines
  cifar10_resnet13/      E2/E3 on CIFAR-10 behind a ResNet-13 trunk, plus the baselines
  ebm_recognition/       E4: the energy-based recognition model, K x r grids, Gibbs sweeps
  flops_model/           E5: the FLOPs-per-epoch model and its XLA measurement
data/                    datasets (fetched by download_data.py) and the shipped LeNet-5 trunks
```

## Setup

Python 3.13 with JAX (CPU or CUDA), NumPy, SciPy, Matplotlib and openpyxl:

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows
.venv/bin/python -m pip install -r requirements.txt             # Linux / macOS
```

Datasets are not part of the repository. Fetch them once (about 800 MB of downloads, the
EMNIST archive being the largest):

```
python data/download_data.py                 # fashion, emnist, cifar, svhn
python data/download_data.py fashion emnist  # only what the Fashion-MNIST experiments need
```

Set `THRML_DATA_DIR` to keep the data somewhere else; every loader honours it. The
trained LeNet-5 trunks of the thesis are shipped (`data/fashion/trunks/`,
`studies/fashion_lenet5/artifacts/`); the ResNet-13 trunks are trained on first use
(about one GPU hour per seed).

## Two ways to run an experiment

**1. The pipeline (Excel-driven).** Every row of the `Runs` sheet in
`pipeline/run_config.xlsx` is one configuration; a blank cell means the documented default
of that task. Dropdowns in the `Optionen` sheet list the allowed values, the `Parameters`
sheet documents every column. As shipped, four cheap rows are enabled (`fam_04`, `fam_05`
on blob and `routing_op_mf`, `routing_op_tree` on routing); those tasks are synthetic, so
they need no dataset and finish in about half a minute each on a laptop CPU. Every other
row is disabled. Set `enabled` to `yes` and run

```
python pipeline/runner.py --list                 # which rows are enabled
python pipeline/runner.py --dry-run              # validate + resolve, no training
python pipeline/runner.py                        # execute the enabled rows
python pipeline/runner.py --only fam_04 fam_05   # specific run_ids, enabled or not
```

Each run writes `pipeline/runs/<run_id>_<timestamp>/` (config, per-seed metrics, figures)
and upserts its rows into `pipeline/runs/results.xlsx`. See `pipeline/README.md`.

**2. The studies.** The numbers reported in the thesis were produced by the standalone
scripts under `studies/`, which call the same trainer and task modules as the pipeline
but add the study-specific measurements (exact posterior probes, refit ladders, OOD
ladders, the baseline networks). Each study folder has a README with the commands.

## Where each thesis experiment lives

| Thesis | Script(s) | Workbook rows |
|---|---|---|
| E1.1 blob, E1.2 routing, E1.3 posterior fidelity (Tables 6.1-6.4, Figs 6.1-6.4) | `studies/e1_artificial/run_baselines.py`, `run_temp_scaling.py` | `fam_04`-`fam_09`, `routing_op_*` |
| E2 base comparison on Fashion-MNIST (Table 6.6) | `studies/fashion_lenet5/experiments/run_seeds.py`, `run_vibnn.py` | `fashion_repro_a4a1` |
| E2 base comparison on CIFAR-10 (Table 6.7) | `studies/cifar10_resnet13/experiments/run_trunk.py`, `run_seeds.py`, `run_vibnn.py` | `c10r_repro_a4a1` |
| E3.1 competence of W0, alpha sweep (Figs 6.6, 6.7, A.4, A.5) | `run_seeds.py --alphas ...`, `run_controls.py` (both studies) | `ex_alpha_*` |
| E3.2 OOD fraction ladder (Figs 6.8-6.10, A.6) | the OOD ladder every `run_seeds.py` run writes | `*_ood_ladder=yes` |
| E3.3 capacity scaling K x r (Figs 6.11-6.14, A.7) | `studies/ebm_recognition/experiments/run_fashion_kr_gibbs.py --family tree` (`--trunk-width 0.25`, `--approach a1`) | `ap4_grid_tree_*`, `c10r_grid_*` |
| E4.1 EBM recognition model (Fig. 6.15, Table 6.8, A.8-A.10) | `run_fashion_kr_gibbs.py --family ebm-gibbs`, `run_fashion_topologies.py`, `run_fashion_families.py` | `family = ebm` / `ebm-gibbs` |
| E4.2 Gibbs sweep sensitivity (Fig. 6.16) | `run_fashion_sweepfine.py`, `run_mech_check.py` | `gibbs_sweeps_S` |
| E5 FLOPs model (Fig. 6.17, App. A.19) | `studies/flops_model/experiments/measure_flops.py`, `reporting/make_table.py`, `plot_flops_composition.py` | - |

## Compute

The blob and routing experiments, the E1 posterior ladders and the FLOPs measurement run
on a laptop CPU in minutes to a few hours. One Fashion-MNIST head (10 000 training points,
2 400 full-batch epochs) takes about 15 minutes on a CPU and 30 seconds on a data-centre
GPU; the K x r grids (360 heads per family and level) and everything on CIFAR-10 were run
on GPUs. Every script is restart-safe and caches trained heads and trunks, so a campaign
can be spread over several sessions.

## Acknowledgements

I thank my supervisor, Dr. Andrew Duncan, for the guidance and the feedback that shaped
this project.

**Use of AI tools.** Generative AI assistants were used while building this repository: to
help write and refactor code, and to draft the in-code comments and the README files. The research question, the experimental
design, every number reported in the thesis and all conclusions drawn from them are my
own, and I am responsible for the contents of this repository.

## Licence

MIT, see `LICENSE`.
