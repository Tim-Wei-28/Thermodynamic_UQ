"""
Tree-structured Bernoulli recognition model q(z|x,y) with closed-form marginals,
second moments and entropy, plus a mean-field block with the same interface.
Head layout (2K-1 columns): pre[:, 0] = root logit; per non-root node k,
pre[:, pos_alpha[k]] -> q(z_k=1|z_pa=0), pre[:, pos_beta[k]] -> q(z_k=1|z_pa=1).
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp


class TreeSpec(NamedTuple):
    # static tree shape; plain tuples so it can be a static jit argument
    K: int
    root: int
    parents: tuple              # parents[k] = parent of k, or -1 for the root
    topo: tuple                 # node order with parents strictly before children
    n_pre: int                  # head width = 2K - 1
    pos_root: int
    pos_alpha: tuple            
    pos_beta: tuple             
    cov_lca: tuple
    cov_idx: tuple


def _root_path(parents, k):
    """[k, pa(k), ..., root]."""
    path = [k]
    while parents[k] != -1:
        k = parents[k]
        path.append(k)
    return path


def _build_cov_plan(parents, K):
    """For each pair (k, j): LCA and the slope indices with
       Cov(z_k, z_j) = sigma2[LCA] * prod_{c in idx} s_c."""
    cov_lca = [[0] * K for _ in range(K)]
    cov_idx = [[() for _ in range(K)] for _ in range(K)]
    for k in range(K):
        pk = _root_path(parents, k)
        setk = {n: i for i, n in enumerate(pk)}
        for j in range(K):
            if j == k:
                continue
            pj = _root_path(parents, j)
            lca = next(n for n in pj if n in setk)
            idx_k = pk[:setk[lca]]
            idx_j = pj[:pj.index(lca)]
            cov_lca[k][j] = lca
            cov_idx[k][j] = tuple(idx_k + idx_j)
    return tuple(tuple(r) for r in cov_lca), tuple(tuple(r) for r in cov_idx)


def make_chain_spec(K: int, root: int = 0) -> TreeSpec:
    """Chain root - (root+1) - ... over the natural node order."""
    parents = [root] * K
    parents[root] = -1
    for k in range(1, K):
        parents[k] = k - 1
    parents[root] = -1
    topo = list(range(root, K)) if root == 0 else None
    if topo is None:                       # generic: BFS from root
        topo, frontier = [], [root]
        while frontier:
            n = frontier.pop(0)
            topo.append(n)
            frontier += [c for c in range(K) if parents[c] == n]
    # head layout: column 0 = root; then (alpha, beta) per non-root node
    pos_alpha = [-1] * K
    pos_beta = [-1] * K
    col = 1
    for k in range(K):
        if k == root:
            continue
        pos_alpha[k] = col
        pos_beta[k] = col + 1
        col += 2
    cov_lca, cov_idx = _build_cov_plan(parents, K)
    return TreeSpec(K=K, root=root, parents=tuple(parents), topo=tuple(topo),
                    n_pre=2 * K - 1, pos_root=0,
                    pos_alpha=tuple(pos_alpha), pos_beta=tuple(pos_beta),
                    cov_lca=cov_lca, cov_idx=cov_idx)


def make_chain_from_order(order) -> TreeSpec:
    """Chain visiting the gates in the given order: root = order[0], parent of
       order[i] is order[i-1].  Gate indices stay tied to their adapters."""
    order = list(order)
    K = len(order)
    parents = [-1] * K
    root = order[0]
    for i in range(1, K):
        parents[order[i]] = order[i - 1]
    topo = order[:]
    pos_alpha = [-1] * K
    pos_beta = [-1] * K
    col = 1
    for k in range(K):                    # head layout stays in natural index order
        if k == root:
            continue
        pos_alpha[k] = col
        pos_beta[k] = col + 1
        col += 2
    cov_lca, cov_idx = _build_cov_plan(parents, K)
    return TreeSpec(K=K, root=root, parents=tuple(parents), topo=tuple(topo),
                    n_pre=2 * K - 1, pos_root=0,
                    pos_alpha=tuple(pos_alpha), pos_beta=tuple(pos_beta),
                    cov_lca=cov_lca, cov_idx=cov_idx)


# Named fixed shapes.  All have K-1 edges and the same head width; root is always gate 0.
TOPOLOGIES = ("chain", "binary", "star", "random", "hier")


def make_topology_parents(K, kind="chain", seed=0, task_parents=None):
    """Parent array for a named shape (root = 0).
       chain 0-1-2-...; binary parents[k] = (k-1)//2; star: all hang off 0;
       random: each gate attaches to a random earlier gate (seeded);
       hier: caller-supplied task_parents."""
    if kind not in TOPOLOGIES:
        raise ValueError(f"unknown tree topology {kind!r}; have {TOPOLOGIES}")
    if kind == "hier":
        if task_parents is None:
            raise ValueError(
                "tree_topology='hier' needs the task's generative parent array "
                "(task_parents=), which only a task with a declared hierarchy can supply.  "
                "It is not derivable from K alone.")
        par = [int(p) for p in task_parents]
        if len(par) != K:
            raise ValueError(f"tree_topology='hier': task_parents has {len(par)} entries "
                             f"but K={K}")
        return par
    if K <= 1:
        return [-1]
    if kind == "chain":
        return [-1] + list(range(K - 1))
    if kind == "binary":
        return [-1] + [(k - 1) // 2 for k in range(1, K)]
    if kind == "star":
        return [-1] + [0] * (K - 1)
    rng = np.random.default_rng(int(seed))            # numpy: never touches the JAX stream
    return [-1] + [int(rng.integers(0, k)) for k in range(1, K)]


def make_tree_from_parents(parents) -> TreeSpec:
    """TreeSpec from an arbitrary parent array (parents[root] == -1).
       Same head layout as the chain builders, so a chain's parents reproduce them."""
    parents = [int(p) for p in parents]
    K = len(parents)
    roots = [k for k in range(K) if parents[k] == -1]
    if len(roots) != 1:
        raise ValueError(f"parents must have exactly one root (parent == -1); got {roots}")
    root = roots[0]
    # BFS from the root so every parent precedes its children (the forward relies on it)
    topo, frontier = [], [root]
    while frontier:
        n = frontier.pop(0)
        topo.append(n)
        frontier += [c for c in range(K) if parents[c] == n]
    if len(topo) != K:
        raise ValueError("parents is not a single connected tree (topo did not reach every node)")
    pos_alpha = [-1] * K
    pos_beta = [-1] * K
    col = 1
    for k in range(K):
        if k == root:
            continue
        pos_alpha[k] = col
        pos_beta[k] = col + 1
        col += 2
    cov_lca, cov_idx = _build_cov_plan(parents, K)
    return TreeSpec(K=K, root=root, parents=tuple(parents), topo=tuple(topo),
                    n_pre=2 * K - 1, pos_root=0,
                    pos_alpha=tuple(pos_alpha), pos_beta=tuple(pos_beta),
                    cov_lca=cov_lca, cov_idx=cov_idx)


def pairwise_weights(mu, M, weight="abs_cov", eps=1e-9):
    """Symmetric (K,K) edge weights from the recognition moments, batch-averaged:
       "abs_cov" = mean |Cov(z_k, z_j)|, "mi" = mean MI from the 2x2 Bernoulli joint."""
    # NumPy: runs between training segments, outside jit
    mu = np.asarray(mu, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    cov = M - mu[:, :, None] * mu[:, None, :]                   # (N,K,K)
    if weight == "abs_cov":
        W = np.abs(cov).mean(0)
    elif weight == "mi":
        mk, mj = mu[:, :, None], mu[:, None, :]
        p11 = np.clip(M, eps, 1 - eps)
        p10 = np.clip(mk - M, eps, 1 - eps)
        p01 = np.clip(mj - M, eps, 1 - eps)
        p00 = np.clip(1 - mk - mj + M, eps, 1 - eps)
        mi = (p11 * np.log(p11 / (mk * mj))
              + p10 * np.log(p10 / (mk * (1 - mj)))
              + p01 * np.log(p01 / ((1 - mk) * mj))
              + p00 * np.log(p00 / ((1 - mk) * (1 - mj))))      # (N,K,K)
        W = mi.mean(0)
    else:
        raise ValueError(f"unknown edge weight {weight!r} (abs_cov | mi)")
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def max_spanning_tree_parents(W, root=0):
    """Maximum-weight spanning tree (Prim, dense) as a parent array rooted at `root`.
       Ties break by lowest index."""
    W = np.asarray(W, dtype=np.float64)
    K = W.shape[0]
    parents = [-1] * K
    if K <= 1:
        return parents
    in_tree = np.zeros(K, dtype=bool)
    in_tree[root] = True
    best_w = W[root].astype(np.float64).copy()
    best_from = np.full(K, root, dtype=int)
    best_w[root] = -np.inf
    for _ in range(K - 1):
        cand = np.where(in_tree, -np.inf, best_w)
        j = int(np.argmax(cand))
        parents[j] = int(best_from[j])
        in_tree[j] = True
        upd = (~in_tree) & (W[j] > best_w)
        best_w = np.where(upd, W[j], best_w)
        best_from = np.where(upd, j, best_from)
    return parents


def chow_liu_parents(mu, M, weight="abs_cov", root=0):
    """Chow-Liu tree from the recognition moments -> parent array for make_tree_from_parents."""
    W = pairwise_weights(mu, M, weight)
    return max_spanning_tree_parents(W, root=root)


def _unpack(pre, spec: TreeSpec):
    """pre: (N, 2K-1) -> mu_root (N,), alpha (N,K), beta (N,K), s (N,K).
       Root columns of alpha/beta/s are 0 (unused)."""
    N = pre.shape[0]
    mu_root = jax.nn.sigmoid(pre[:, spec.pos_root])
    alpha = jnp.zeros((N, spec.K))
    beta = jnp.zeros((N, spec.K))
    for k in range(spec.K):
        if k == spec.root:
            continue
        alpha = alpha.at[:, k].set(jax.nn.sigmoid(pre[:, spec.pos_alpha[k]]))
        beta = beta.at[:, k].set(jax.nn.sigmoid(pre[:, spec.pos_beta[k]]))
    s = beta - alpha                                           # edge slope
    return mu_root, alpha, beta, s


def tree_forward(pre, spec: TreeSpec):
    """Differentiable forward pass -> (mu (N,K), M (N,K,K), H (N,)).
       alpha_k = q(z_k=1|z_pa=0), beta_k = q(z_k=1|z_pa=1), s_k = beta_k - alpha_k."""
    mu_root, alpha, beta, s = _unpack(pre, spec)
    N = pre.shape[0]

    # marginals: mu_k = alpha_k + s_k mu_pa(k), walking topo so the parent exists
    mu_cols = [None] * spec.K
    mu_cols[spec.root] = mu_root
    for k in spec.topo:
        if k == spec.root:
            continue
        pa = spec.parents[k]
        mu_cols[k] = alpha[:, k] + s[:, k] * mu_cols[pa]
    mu = jnp.stack(mu_cols, axis=1)                 # (N, K)
    sigma2 = mu * (1.0 - mu)                         # (N, K)

    # second moments: Cov(z_k,z_j) = sigma2[LCA] * prod of slopes on both paths down.
    # D[:, m, k] = product of slopes from k up to (excluding) m, 0 if m is not an
    # ancestor, 1 if m == k.  Built by D[m,k] = D[m,pa(k)] * s_k, without division,
    # since s is driven to ~0 on cut edges and a quotient would be 0/0.
    D_cols = [None] * spec.K
    eye = jnp.eye(spec.K)
    D_cols[spec.root] = jnp.broadcast_to(eye[:, spec.root], (N, spec.K))
    for k in spec.topo:
        if k == spec.root:
            continue
        D_cols[k] = D_cols[spec.parents[k]] * s[:, k][:, None] + eye[:, k][None, :]
    D = jnp.stack(D_cols, axis=2)                    # (N, K, K), indexed [n, ancestor, node]

    _lca = jnp.asarray(np.asarray(spec.cov_lca), dtype=jnp.int32)      # (K, K), static
    _kk, _jj = np.meshgrid(np.arange(spec.K), np.arange(spec.K), indexing="ij")
    cov = (sigma2[:, _lca]
           * D[:, _lca, jnp.asarray(_kk)]
           * D[:, _lca, jnp.asarray(_jj)])
    M = jnp.einsum("nk,nj->nkj", mu, mu) + cov
    diag = jnp.arange(spec.K)
    M = M.at[:, diag, diag].set(mu)             # E[z^2] = E[z] for binary z

    # entropy: H_b(mu_r) + sum_{k!=r}[(1-mu_pa)H_b(alpha_k)+mu_pa H_b(beta_k)]
    H = _Hb(mu_root)
    for k in spec.topo:
        if k == spec.root:
            continue
        mpa = mu_cols[spec.parents[k]]
        H = H + (1.0 - mpa) * _Hb(alpha[:, k]) + mpa * _Hb(beta[:, k])
    return mu, M, H


def _Hb(p):
    # binary entropy in nats; clip keeps log(0) away
    p = jnp.clip(p, 1e-12, 1 - 1e-12)
    return -(p * jnp.log(p) + (1 - p) * jnp.log1p(-p))


def tree_sample(key, pre, spec: TreeSpec, n_samples: int = 1):
    """Ancestral sampling: z_r ~ Bern(mu_r); z_k ~ Bern((1-z_pa)alpha_k + z_pa beta_k).
       Returns (n_samples, N, K)."""
    mu_root, alpha, beta, _ = _unpack(pre, spec)
    N = pre.shape[0]
    keys = jax.random.split(key, spec.K)             # one sub-key per gate
    z = [None] * spec.K
    kr = jax.random.uniform(keys[spec.root], (n_samples, N))
    z[spec.root] = (kr < mu_root[None, :]).astype(jnp.float32)
    for k in spec.topo:
        if k == spec.root:
            continue
        pa = spec.parents[k]
        pi = (1.0 - z[pa]) * alpha[None, :, k] + z[pa] * beta[None, :, k]
        u = jax.random.uniform(jax.random.fold_in(keys[k], 0), (n_samples, N))
        z[k] = (u < pi).astype(jnp.float32)
    return jnp.stack(z, axis=2)                      # (n_samples, N, K)


def _logit(p):
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def tree_sample_relaxed(key, pre, spec: TreeSpec, tau: float = 1.0, n_samples: int = 1):
    """Binary-concrete relaxed ancestral sampling; the relaxed parent value feeds the
       child's conditional.  For the reconstruction gradient only.  Returns (T,N,K) in (0,1)."""
    mu_root, alpha, beta, _ = _unpack(pre, spec)
    N = pre.shape[0]
    keys = jax.random.split(key, spec.K)

    def noise(k):
        u = jax.random.uniform(k, (n_samples, N), minval=1e-6, maxval=1 - 1e-6)
        return jnp.log(u) - jnp.log1p(-u)            # logistic noise

    z = [None] * spec.K
    z[spec.root] = jax.nn.sigmoid((_logit(mu_root)[None, :] + noise(keys[spec.root])) / tau)
    for k in spec.topo:
        if k == spec.root:
            continue
        pa = spec.parents[k]
        pi = (1.0 - z[pa]) * alpha[None, :, k] + z[pa] * beta[None, :, k]   # soft z[pa]
        z[k] = jax.nn.sigmoid((_logit(pi) + noise(keys[k])) / tau)
    return jnp.stack(z, axis=2)                       # (n_samples, N, K)


def tree_score(z, pre, spec: TreeSpec):
    """d log q / d pre at a sampled z:  z:(...,N,K) -> (...,N,2K-1).
       In pre-activation space the sigmoid Jacobian cancels the Bernoulli denominator."""
    mu_root, alpha, beta, _ = _unpack(pre, spec)
    lead = z.shape[:-1]                              # (...,N)
    score = jnp.zeros(lead + (spec.n_pre,))
    score = score.at[..., spec.pos_root].set(z[..., spec.root] - mu_root)
    for k in spec.topo:
        if k == spec.root:
            continue
        pa = spec.parents[k]
        zpa = z[..., pa]
        # alpha gets gradient only when the parent was off, beta only when on
        score = score.at[..., spec.pos_alpha[k]].set((1.0 - zpa) * (z[..., k] - alpha[:, k]))
        score = score.at[..., spec.pos_beta[k]].set(zpa * (z[..., k] - beta[:, k]))
    return score


def tree_logq(z, pre, spec: TreeSpec):
    """log q(z | x, y) for given hard configs, differentiable in pre.  z:(...,N,K) -> (...,N).
       Its pre-gradient equals tree_score (the sfe-loo surrogate)."""
    mu_root, alpha, beta, _ = _unpack(pre, spec)
    eps = 1e-12
    zr = z[..., spec.root]
    lp = zr * jnp.log(mu_root + eps) + (1.0 - zr) * jnp.log1p(-mu_root + eps)
    for k in spec.topo:
        if k == spec.root:
            continue
        pa = spec.parents[k]
        zk = z[..., k]
        zpa = z[..., pa]
        pik = (1.0 - zpa) * alpha[:, k] + zpa * beta[:, k]
        lp = lp + zk * jnp.log(pik + eps) + (1.0 - zk) * jnp.log1p(-pik + eps)
    return lp


def all_configs(K: int):
    """(2^K, K) array of all binary gate configurations; row i is the binary expansion of i."""
    grid = jnp.array([[(i >> b) & 1 for b in range(K)] for i in range(2 ** K)],
                     dtype=jnp.float32)
    return grid


def tree_enumerate_logq(pre, spec: TreeSpec, configs=None):
    """Exact log q(z) for every config, per input.  pre:(N,2K-1) -> (N,2^K)."""
    if configs is None:
        configs = all_configs(spec.K)
    mu_root, alpha, beta, _ = _unpack(pre, spec)     # (N,), (N,K), (N,K)
    N = pre.shape[0]
    eps = 1e-12
    zr = configs[:, spec.root]                       # (C,)
    lp = (zr[None, :] * jnp.log(mu_root[:, None] + eps)
          + (1 - zr)[None, :] * jnp.log1p(-mu_root[:, None] + eps))   # (N, C)
    for k in spec.topo:
        if k == spec.root:
            continue
        pa = spec.parents[k]
        zk = configs[:, k][None, :]                  # (1,C)
        zpa = configs[:, pa][None, :]                # (1,C)
        pi = (1 - zpa) * alpha[:, k][:, None] + zpa * beta[:, k][:, None]   # (N,C)
        lp = lp + zk * jnp.log(pi + eps) + (1 - zk) * jnp.log1p(-pi + eps)
    return lp                                        # (N, 2^K)


# Mean-field block, same interface (independent gates, head width K).
class MeanFieldSpec(NamedTuple):
    K: int
    n_pre: int                  # = K


def make_mf_spec(K: int) -> MeanFieldSpec:
    return MeanFieldSpec(K=K, n_pre=K)


def mf_forward(pre, spec: MeanFieldSpec):
    mu = jax.nn.sigmoid(pre)                         # (N,K)
    M = jnp.einsum("nk,nj->nkj", mu, mu)             # off-diagonal covariance is forced to 0
    diag = jnp.arange(spec.K)
    M = M.at[:, diag, diag].set(mu)                   # E[z^2] = E[z] for binary z
    H = _Hb(mu).sum(-1)
    return mu, M, H


def mf_sample(key, pre, spec: MeanFieldSpec, n_samples: int = 1):
    mu = jax.nn.sigmoid(pre)
    u = jax.random.uniform(key, (n_samples,) + pre.shape)
    return (u < mu[None]).astype(jnp.float32)        # (n_samples, N, K)


def mf_score(z, pre, spec: MeanFieldSpec):
    mu = jax.nn.sigmoid(pre)
    return z - mu                                    # (..., N, K)


def mf_logq(z, pre, spec: MeanFieldSpec):
    """log q(z) of the factorised posterior, sum of Bernoulli log-probs -> (..., N).
       Lets mean-field run under the sfe-loo estimator too."""
    mu = jax.nn.sigmoid(pre)
    eps = 1e-12
    return (z * jnp.log(mu + eps) + (1.0 - z) * jnp.log1p(-mu + eps)).sum(-1)


def mf_sample_relaxed(key, pre, spec: MeanFieldSpec, tau: float = 1.0, n_samples: int = 1):
    """Binary-concrete relaxed sampling for the mean-field family."""
    mu = jax.nn.sigmoid(pre)
    u = jax.random.uniform(key, (n_samples,) + pre.shape, minval=1e-6, maxval=1 - 1e-6)
    g = jnp.log(u) - jnp.log1p(-u)                        # logistic noise
    return jax.nn.sigmoid((_logit(mu)[None] + g) / tau)   # (n_samples, N, K)


def mf_enumerate_logq(pre, spec: MeanFieldSpec, configs=None):
    if configs is None:
        configs = all_configs(spec.K)
    mu = jax.nn.sigmoid(pre)                          # (N,K)
    eps = 1e-12
    lp = (configs[None] * jnp.log(mu[:, None] + eps)
          + (1 - configs[None]) * jnp.log1p(-mu[:, None] + eps)).sum(-1)
    return lp                                         # (N, 2^K)


def moments_from_logq(logq, configs):
    """Brute-force (mu, M, H) from log q over configs (N,2^K) and configs (2^K,K);
       ground truth for verification."""
    q = jnp.exp(logq - jax.scipy.special.logsumexp(logq, axis=1, keepdims=True))
    mu = q @ configs                                 # (N,K)
    M = jnp.einsum("nc,ck,cj->nkj", q, configs, configs)   # E[z z^T]
    H = -(q * jnp.log(q + 1e-300)).sum(1)
    return mu, M, H
