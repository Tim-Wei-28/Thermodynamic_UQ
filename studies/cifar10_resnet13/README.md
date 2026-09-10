# E2 / E3 on CIFAR-10 behind a ResNet-13 trunk

The CIFAR-10 counterpart of `studies/fashion_lenet5`: the mixture model replaces the last
layer of a ResNet-13 (thesis Chapter 5.2, Figure 6.5) and is compared against the
mean-field VI and MC dropout (E2, Table 6.7), along the base-map
competence alpha (E3.1, Figures A.4 and A.5) and along the SVHN blend fraction f (E3.2,
Figures 6.9 and A.6). Same protocol as the Fashion-MNIST study: 100 prior draws per
image, ECE with 15 bins, OOD probe x = (1 - f) cifar + f svhn with the CIFAR label kept.

The trunk: 45 000 training images (4 500 per class, the remaining 500 per class form the
validation pool), 100 epochs with the step schedule, 1 228 522 parameters of which the
2 570 of the output layer are replaced. The mixture reads the 256 pooled features plus a
constant column (D = 257); K = 10, rank 8.

## Running

```
cd studies/cifar10_resnet13/experiments
python run_trunk.py --seeds 0                     # train + cache the trunk (about one GPU hour per seed)
python run_seeds.py --seeds 0,1,2,3,4,5,6,7,8,9   # ResNet, MC dropout, the mixture at alpha 1.0 / 0.0, A1 pixels
python run_seeds.py --alphas 0,0.25,0.5,0.7,0.75,0.8,0.9,0.95,1 --force   # the alpha sweep
python run_vibnn.py --seeds 0,1,2,3,4,5,6,7,8,9   # the mean-field VI ResNet (2 454 602 parameters)
python run_controls.py                            # dropout-gates / all-on controls per alpha
cd ../reporting
python collect_seeds.py
```

Trunks are cached in `data/cifar/resnet_trunks/`, heads and results in `artifacts/`.
The task module (split, trunk cache, feature scale, probes) is `pipeline/model/
cifar10_resnet_task.py`, shared with the pipeline's `cifar10_resnet` task. Data: `data/cifar`
and `data/svhn` (or `$THRML_DATA_DIR`), fetched automatically on first use or by
`data/download_data.py`.

These experiments need a GPU: the trunk, the VI ResNet and the 45 000-point heads are not
practical on a CPU.
