"""Per-path recipe defaults: what a blank spreadsheet cell means.
"""
from __future__ import annotations

# recipe skeleton shared by the synthetic tasks (thesis Ch. 4.5.1)
_BASE = dict(lr=0.05, T=8, S=12, anneal_frac=0.4, free_bits=0.02, tau=1.0,
             clip=1.0, wd=0.005, beta_max=0.5, wd_anchor=0.1, warmup=0,
             head_init=0.3, node_order=None)

DEFAULTS = {
    # free adapters plus a gates-on bootstrap (c_bias / field_b2) and a negative J init.
    "routing": dict(
        epochs=1200, lr=0.05, T=8, S=12, anneal_frac=0.4, free_bits=0.02, tau=1.0,
        clip=1.0, wd=0.01, beta_max=1.0, wd_anchor=0.1, warmup=0,
        head_init=0.1, node_order=None,
        rank=4, w0_scale=0.1, adapter_init=0.1,
        j_init=-1.5, field_kind="mlp", field_n_hidden=24,
        c_bias=0.5, field_b2=0.5,
        n_tr=2400, n_te=600, keep=0.6,          # train / test size, dropout keep-rate
        gamma0=0.0, warm_frac=0.15),            # no q-target: the task grounds the gates

    # Fashion-MNIST behind a LeNet trunk.  n_per / n_te / n_val are per class.
    # w0_alpha=1.0 is the weak random base map, alpha=0 the trained last layer.
    "fashion": dict(
        epochs=2400, lr=0.1, T=8, S=12, anneal_frac=0.4, free_bits=0.02, tau=1.0,
        clip=1.0, wd=0.01, beta_max=1.0, wd_anchor=0.1, warmup=0,
        head_init=0.1, node_order=None,
        rank=8, w0_scale=0.1, adapter_init=0.1,
        j_init=-1.5, field_kind="mlp", field_n_hidden=24,
        c_bias=0.5, field_b2=0.5,
        # n_val is the whole validation pool and is used only to fit T*.
        n_per=1000, n_te=1000, n_val=500, keep=0.6,
        gamma0=1.0, warm_frac=0.15,
        div=0.0, sparse=0.0, loadbal=0.0, label_dropout_p=0.0,
        classes="all", trunk="lenet55k", trunk_epochs=20, w0_alpha=1.0,
        probe="letters", ood_fraction=0.5, ood_n=1000, n_samples=100, ood_ladder="no",
        edge_weight="abs_cov", refit="once", struct_warmup=0.3),

    # CIFAR-10 behind the ResNet trunk.  Same recipe as fashion; n_* are per class.
    # trunk_schedule keys the trunk cache, so it belongs to the recipe.
    "cifar10_resnet": dict(
        epochs=2400, lr=0.1, T=8, S=12, anneal_frac=0.4, free_bits=0.02, tau=1.0,
        clip=1.0, wd=0.01, beta_max=1.0, wd_anchor=0.1, warmup=0,
        head_init=0.1, node_order=None,
        rank=8, w0_scale=0.1, adapter_init=0.1,
        j_init=-1.5, field_kind="mlp", field_n_hidden=24,
        c_bias=0.5, field_b2=0.5,
        n_per=4500, n_te=1000, n_val=500, keep=0.6,
        gamma0=1.0, warm_frac=0.15,
        div=0.0, sparse=0.0, loadbal=0.0, label_dropout_p=0.0,
        classes="all", trunk="resnet45k", trunk_epochs=100, trunk_schedule="step",
        w0_alpha=1.0,
        probe="svhn", ood_fraction=0.5, ood_n=1000, n_samples=100, ood_ladder="no",
        edge_weight="abs_cov", refit="once", struct_warmup=0.3),

    # gain is the class-handle strength; gamma0=1.0 switches the q-target warm-up on.
    "blob": dict(
        **_BASE, epochs=1500,
        K=6, rank=2, gain=5.0, w0_scale=0.2, n_per=200,
        j_init=-0.5, field_kind="mlp", field_n_hidden=24,
        gamma0=1.0, warm_frac=0.15,
        div=0.0, sparse=0.0, loadbal=0.0),
}
