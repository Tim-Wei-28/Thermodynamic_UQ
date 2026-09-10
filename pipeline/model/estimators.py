"""
Building blocks of the score-function (sfe-loo) reconstruction estimator.
The relaxed (Gumbel-concrete) path lives inline in the trainer.
"""
from __future__ import annotations
import jax


def loo_advantage(logp):
    """Leave-one-out advantage.  logp:(T,N) -> adv:(N,T); baseline of sample t
       is the mean of the other T-1 samples."""
    T = logp.shape[0]
    base = (logp.sum(0, keepdims=True) - logp) / (T - 1)
    return (logp - base).T


def sfe_recon(adv, logq):
    """SFE surrogate (1/T) sum_t stopgrad(adv_t) log q(z_t), batch mean.
       adv:(N,T)  logq:(T,N) -> scalar.  Only its gradient is meaningful."""
    # The 1/T makes this the sample mean the score-function identity calls for.
    T = logq.shape[0]
    return (jax.lax.stop_gradient(adv.T) * logq).sum(0).mean() / T
