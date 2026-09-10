# E2 / E3 on Fashion-MNIST behind a LeNet-5 trunk

The mixture model as the last layer of a trained LeNet-5, compared against the same
network's deterministic, mean-field-VI and MC-dropout versions (thesis E2, Table 6.6) and
studied along the base-map competence alpha (E3.1) and the EMNIST-letter blend fraction f
(E3.2). The reference architecture and the evaluation protocol follow

> Liu, Xiao, Kwon, Debusschere, Agarwal, Incorvia, Bennett (2022). *Bayesian neural
> networks using magnetic tunnel junction-based probabilistic in-memory computing.*
> Front. Nanotechnol. 4:1021943.

## The network and the three arms

```
32x32x1 (28x28 padded by 2) -> C1 conv 6@5x5 -> maxpool -> C3 conv 16@5x5 -> maxpool
        -> C5 dense 400->120 -> F6 dense 120->84 -> Out dense 84->10        61 706 parameters
```

- **A1, raw pixels**: the mixture on the 784 centred pixels, weak random frozen W0.
  Linear in x; its accuracy is not that of a convnet, its calibration is comparable.
- **A2, deterministic**: the LeNet-5 itself (Adam, 20 epochs, batch 32).
- **A3 / A4, trunk**: the trunk up to F6 is frozen, the Out layer (850 parameters) is
  replaced by the mixture with `W0(alpha) = (1 - alpha) W0_trained + alpha W0_random`.
  alpha = 0 keeps the trained last layer (the collapsed regime), alpha = 1 is the weak
  random map the thesis reports as "random W0". The F6 activations get a constant column
  that carries the layer's bias, so W0 is exactly (10, 85).

The baselines: `lib/vi_bnn.py` is the mean-field VI LeNet-5 (a Gaussian over every
weight, deterministic biases, 123 176 trained parameters); MC dropout evaluates the
deterministic network under 100 dropout masks.

## Protocol

Train 55 000 / validation 5 000 (temperature fitting only) / test the full official
10 000. 100 prior draws per test image. Metrics: accuracy, ECE (15 bins), NLL, Brier, the
entropy decomposition H_total = H(E[p]), H_alea = E[H(p)], BALD = their difference.
OOD probes: `letters` (x = (1 - f) clothing + f EMNIST letter, label kept), `noise` and
`rotate`, each over f = 0, 0.1, ..., 0.9 with 1 000 pairs per fraction.

## Running

```
cd studies/fashion_lenet5/experiments
python run_seeds.py --seeds 0,1,2,3,4,5,6,7,8,9        # LeNet, MC dropout, the mixture at alpha 1.0 / 0.9 / 0.0, A1
python run_seeds.py --alphas 0,0.25,0.5,0.7,0.75,0.8,0.9,0.95,1 --force   # the alpha sweep of E3.1
python run_vibnn.py --kl-weights 0.2                    # the mean-field VI baseline
python run_controls.py                                  # dropout-gates / all-on controls per alpha
python run_ansatz4.py --alphas 0,0.5,1                  # a single-seed alpha sweep
cd ../reporting
python collect_seeds.py                                 # mean +- SE over the seed files
python make_table.py                                    # raw vs temperature-scaled table
```

Every run writes `artifacts/seed_<n>.json` (all rows and the full OOD ladders) and caches
trained heads (`ebm_*.pkl`) and trunks (`lenet_*.npz`); the trunks used in the thesis are
shipped in `artifacts/`. One head is about 15 minutes on a CPU and 30 seconds on a
data-centre GPU; a whole seed (two LeNets, four heads, the evaluation) about five minutes
on a GPU.

## Layout

```
lib/            paths, fashion_data (splits, probes), ebm_head (the mixture as a head on
                arbitrary features), metrics_liu (ECE / NLL / entropy decomposition),
                vi_bnn (the mean-field VI LeNet-5), family_probe (exact posterior probes),
                mechanism (cached training + collapse readouts)
experiments/    run_seeds, run_vibnn, run_controls, run_ansatz4
reporting/      collect_seeds, make_table
artifacts/      trained parameters, result JSONs (created on first run)
```

The LeNet-5, the IDX readers, the EMNIST loader and the corruption probes live in
`pipeline/model/` (`lenet.py`, `fashion_task.py`) and are imported, so the pipeline's
`fashion` task and this study run the same data and the same trunk. Data is read from
`data/fashion` and `data/emnist` (or `$THRML_DATA_DIR`), see `data/download_data.py`.
