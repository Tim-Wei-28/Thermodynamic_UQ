# E4: the energy-based recognition model

The recognition model q(z | x, y) as a full-pairwise Boltzmann machine with per-input
fields and per-input couplings (thesis Chapter 4.4.2), compared with the mean-field and
tree families on Fashion-MNIST and CIFAR-10. The family itself lives in
`pipeline/model/ebm_recognition.py` (exact enumeration of the 2^K gate patterns, K <= 14)
and `pipeline/model/ebm_recognition_gibbs.py` (the Gibbs-chain backend, family
`ebm-gibbs`, any K); this folder holds the experiments around it.

The family hierarchy is mf (no couplings) < tree (K - 1 couplings) < ebm (all
K(K-1)/2), and the head width follows: K, 2K - 1, K + K(K-1)/2. Neither EBM backend has a
relaxed (Gumbel) sampler, so every EBM arm and every arm it is compared with runs the
score-function estimator (`sfe-loo`).

## Scripts

| Script | Thesis | What it does |
|---|---|---|
| `experiments/verify_family.py` | - | correctness checks of the enumerated family (mean-field limit, moments, score = grad log q, sampler frequencies, KL identity, end-to-end training) |
| `experiments/verify_gibbs.py` | - | certifies the Gibbs backend against the enumeration (moments, KL and entropy gradients, the leave-one-out absorption of log Z) |
| `experiments/run_fashion_families.py` | E4.1 | mf / tree / learned tree / ebm at K = 10, r = 8 on paired seeds: downstream metrics, exact in-run KL(q‖p) on held-out images, the refit ladder, the learned couplings Jq |
| `experiments/run_fashion_topologies.py` | E4.1, Table 6.8, A.18 | adds the binary and star trees; the in-run / amortised / free ladder of every family against every arm's posterior |
| `experiments/run_fashion_kr_gibbs.py` | E3.3, E4.1, Figs 6.11-6.15, A.7-A.10 | the K x r capacity grid (K in {1..50}, r in {1..36}, 3 seeds) for `--family tree` or `--family ebm-gibbs`; `--trunk-width 0.25` and `--approach a1` give the weakened-trunk and raw-pixel levels; `--task cifar10` the CIFAR-10 twin |
| `experiments/run_fashion_sweepfine.py` | E4.2, Fig. 6.16 | the Gibbs-sweep ladder S in {1..24} on four K x r cells against the enumerated ceiling |
| `experiments/run_mech_check.py` | E4.1/E4.2 on CIFAR-10 | the family comparison plus the coarse sweep ladder on CIFAR-10 |
| `reporting/collect_*.py` | | validate and aggregate the JSONs a runner wrote into `artifacts/` |

```
python studies/ebm_recognition/experiments/verify_family.py
python studies/ebm_recognition/experiments/verify_gibbs.py
cd studies/ebm_recognition/experiments
python run_fashion_families.py --seeds 0,1,2
python run_fashion_topologies.py --seeds 0,1,2
python run_fashion_kr_gibbs.py --family tree --seeds 0,1,2
python run_fashion_kr_gibbs.py --family ebm-gibbs --seeds 0,1,2
python run_fashion_kr_gibbs.py --family ebm-gibbs --trunk-width 0.25 --seeds 0,1,2
python run_fashion_kr_gibbs.py --family tree --approach a1 --seeds 0,1,2
python run_fashion_sweepfine.py --seeds 0,1,2
python run_mech_check.py --task cifar10 --seeds 0,1,2
python run_fashion_kr_gibbs.py --task cifar10 --family ebm-gibbs --seeds 0,1,2
```

Every runner supports `--ebm-epochs`, `--n-test`, `--n-probe` and `--n-ladder` for smoke
runs, restarts from its per-head `.pkl` cache and writes one JSON per seed into
`artifacts/`. The full grids are GPU campaigns (360 heads per family and level); a single
Fashion head takes about 15 minutes on a CPU.

## Library

`lib/gap_tools.py` enumerates the true gate posterior p(z | x, y) and fits any family to it
by exact reverse KL (amortised through a linear head, or free per input);
`lib/task_prep.py` prepares the Fashion-MNIST (LeNet-5, EMNIST blend) and CIFAR-10
(ResNet-13, SVHN blend) inputs with the same per-seed trunk, subset and W0 draw the
`fashion_lenet5` and `cifar10_resnet13` studies use, so rows pair across studies. The
measurements reuse those studies' libraries (`ebm_head`, `metrics_liu`, `family_probe`).
