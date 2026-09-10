"""
Recognition-family registry.  `make(family, K, ...)` returns (spec, R) where R is a
dict of callables: make_spec, forward -> (mu, M, H), sample, sample_relaxed, logq,
score, enumerate_logq.  Families: tree, mf, tree_max_span, ebm, ebm-gibbs.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # allow sibling imports
import tree_recognition as tr
import ebm_recognition as ebmr
import ebm_recognition_gibbs as ebmg


def _tree_spec(K, node_order=None, topology="chain", seed=0, task_parents=None):
    """Fixed topology over K gates.  node_order only applies to a chain and is
       ignored for branching shapes; task_parents is only read by topology='hier'."""
    if (topology or "chain") == "chain":
        return tr.make_chain_from_order(node_order) if node_order is not None \
            else tr.make_chain_spec(K)
    return tr.make_tree_from_parents(
        tr.make_topology_parents(K, topology, seed, task_parents))


def _mf_spec(K, node_order=None, topology="chain", seed=0, task_parents=None):
    # extra arguments accepted for signature parity, mean-field has no topology
    return tr.make_mf_spec(K)


def _tree_max_span_spec(K, node_order=None, topology="chain", seed=0, task_parents=None):
    """Chow-Liu family: the topology is learned during training, so this spec is only
       the warm-up structure that the first refit replaces."""
    if (topology or "chain") == "chain":
        return tr.make_chain_spec(K)
    return tr.make_tree_from_parents(
        tr.make_topology_parents(K, topology, seed, task_parents))


REGISTRY = {
    "tree": dict(
        make_spec=_tree_spec, forward=tr.tree_forward,
        sample=tr.tree_sample, sample_relaxed=tr.tree_sample_relaxed,
        logq=tr.tree_logq, score=tr.tree_score, enumerate_logq=tr.tree_enumerate_logq),
    "mf": dict(
        make_spec=_mf_spec, forward=tr.mf_forward,
        sample=tr.mf_sample, sample_relaxed=tr.mf_sample_relaxed,
        # logq is wired so mean-field can also run the sfe-loo estimator
        logq=tr.mf_logq, score=tr.mf_score, enumerate_logq=tr.mf_enumerate_logq),
    "tree_max_span": dict(
        make_spec=_tree_max_span_spec, forward=tr.tree_forward,
        sample=tr.tree_sample, sample_relaxed=tr.tree_sample_relaxed,
        logq=tr.tree_logq, score=tr.tree_score, enumerate_logq=tr.tree_enumerate_logq),
    # full-pairwise Boltzmann q (thesis Chapter 4.4.2); both need estimator='sfe-loo'
    "ebm": ebmr.BUNDLE,
    "ebm-gibbs": ebmg.BUNDLE,
}
EBM_FAMILIES = ("ebm", "ebm-gibbs")


def make(family, K, node_order=None, topology="chain", seed=0, task_parents=None):
    """(spec, R) for the named family.  topology: chain | binary | star | random | hier;
       seed only feeds 'random', task_parents only 'hier'."""
    if family not in REGISTRY:
        raise ValueError(f"unknown recognition family {family!r}; have {list(REGISTRY)}")
    R = REGISTRY[family]
    return R["make_spec"](K, node_order, topology, seed, task_parents), R
