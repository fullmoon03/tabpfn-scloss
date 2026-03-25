"""
Synthetic-task rollout utilities.

- rollout x sampling always uses a provided held-out feature pool
- training uses data-only trajectories; beliefs are recomputed through torch path
"""

from dataclasses import dataclass, field

import jax
import numpy as np
import torch
from jaxtyping import PRNGKeyArray

from predictive_rule import PredictiveRule


@dataclass
class TrajectoryData:
    """Data-only trajectory used during training."""

    x_buf: np.ndarray
    y_buf: np.ndarray
    n_base: int
    depth: int
    sampled_pool_indices: np.ndarray


def rollout_one_trajectory_data_only(
    key: PRNGKeyArray,
    pred_rule: PredictiveRule,
    x0: np.ndarray,
    y0: np.ndarray,
    depth: int,
    x_sampling_pool: np.ndarray,
    x_sample_without_replacement: bool = False,
    initial_used_pool_indices: np.ndarray | None = None,
) -> TrajectoryData:
    """
    Build one data-only rollout trajectory.

    x_new is sampled from x_sampling_pool at each rollout step.
    """
    _validate_y_dtype(y0)

    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    pool = np.asarray(x_sampling_pool)
    if pool.ndim != 2:
        raise ValueError(f"x_sampling_pool must be 2D, got shape={pool.shape}")
    if len(pool) == 0:
        raise ValueError("x_sampling_pool must be non-empty.")
    if x0.ndim != 2:
        raise ValueError(f"x0 must be 2D, got shape={x0.shape}")
    if x0.shape[1] != pool.shape[1]:
        raise ValueError(
            f"Feature dim mismatch between x0 and x_sampling_pool: {x0.shape[1]} vs {pool.shape[1]}"
        )

    n_base, dim_x = x0.shape
    x_buf = np.empty((n_base + depth, dim_x), dtype=x0.dtype)
    y_buf = np.empty(n_base + depth, dtype=y0.dtype)
    x_buf[:n_base] = x0
    y_buf[:n_base] = y0
    sampled_pool_indices = np.full(depth, -1, dtype=int)

    available_mask = np.ones(len(pool), dtype=bool)
    if initial_used_pool_indices is not None and len(initial_used_pool_indices) > 0:
        used = np.asarray(initial_used_pool_indices, dtype=int)
        used = used[(used >= 0) & (used < len(pool))]
        if len(used) > 0:
            available_mask[used] = False

    if x_sample_without_replacement and int(available_mask.sum()) < depth:
        raise ValueError(
            "Insufficient x_sampling_pool size for no-replacement rollout: "
            f"available={int(available_mask.sum())}, depth={depth}"
        )

    pred_rule.fit(x0, y0)

    for t in range(depth):
        cur_end = n_base + t

        key, subkey = jax.random.split(key)
        if x_sample_without_replacement:
            avail_idx = np.flatnonzero(available_mask)
            if len(avail_idx) == 0:
                raise RuntimeError("x_sampling_pool exhausted in no-replacement mode.")
            pos = int(jax.random.randint(subkey, shape=(), minval=0, maxval=len(avail_idx)))
            pool_idx = int(avail_idx[pos])
            available_mask[pool_idx] = False
        else:
            pool_idx = int(jax.random.randint(subkey, shape=(), minval=0, maxval=len(pool)))

        sampled_pool_indices[t] = pool_idx
        x_new = np.atleast_2d(pool[pool_idx])
        belief_new = pred_rule.get_belief(x_new)

        key, subkey = jax.random.split(key)
        y_new = pred_rule.sample_y(subkey, belief_new)

        x_buf[cur_end] = x_new.squeeze(0)
        y_buf[cur_end] = np.asarray(y_new).item()

        pred_rule.fit(x_buf[: cur_end + 1], y_buf[: cur_end + 1])

    return TrajectoryData(
        x_buf=x_buf,
        y_buf=y_buf,
        n_base=n_base,
        depth=depth,
        sampled_pool_indices=sampled_pool_indices,
    )


@dataclass
class PrefixBatchData:
    """Prefix batch holding B continuation trajectories."""

    prefix_depth: int
    x_prefix: np.ndarray
    y_prefix: np.ndarray
    continuations: list[TrajectoryData] = field(default_factory=list)


def build_prefix_batch_data(
    key: PRNGKeyArray,
    pred_rule_sampling: PredictiveRule,
    x0: np.ndarray,
    y0: np.ndarray,
    prefix_depth: int,
    continuation_depth: int,
    n_continuations: int = 8,
    x_sampling_pool: np.ndarray | None = None,
    x_sample_without_replacement: bool = False,
) -> PrefixBatchData:
    """
    Build one prefix batch for SC/EMD computation.

    In synthetic-only mode, x_sampling_pool is required.
    """
    if x_sampling_pool is None:
        raise ValueError("x_sampling_pool is required in synthetic-only rollout mode.")

    key, subkey = jax.random.split(key)
    prefix_traj = rollout_one_trajectory_data_only(
        key=subkey,
        pred_rule=pred_rule_sampling,
        x0=x0,
        y0=y0,
        depth=prefix_depth,
        x_sampling_pool=x_sampling_pool,
        x_sample_without_replacement=x_sample_without_replacement,
        initial_used_pool_indices=None,
    )

    prefix_end = prefix_traj.n_base + prefix_depth
    x_prefix = prefix_traj.x_buf[:prefix_end].copy()
    y_prefix = prefix_traj.y_buf[:prefix_end].copy()
    prefix_used_pool_indices = np.asarray(prefix_traj.sampled_pool_indices, dtype=int)

    cont_keys = []
    for _ in range(n_continuations):
        key, subkey = jax.random.split(key)
        cont_keys.append(subkey)

    continuations: list[TrajectoryData] = []
    for b in range(n_continuations):
        cont = rollout_one_trajectory_data_only(
            key=cont_keys[b],
            pred_rule=pred_rule_sampling,
            x0=x_prefix,
            y0=y_prefix,
            depth=continuation_depth,
            x_sampling_pool=x_sampling_pool,
            x_sample_without_replacement=x_sample_without_replacement,
            initial_used_pool_indices=prefix_used_pool_indices,
        )
        continuations.append(cont)

    return PrefixBatchData(
        prefix_depth=prefix_depth,
        x_prefix=x_prefix,
        y_prefix=y_prefix,
        continuations=continuations,
    )


def get_context_at_depth(
    traj: TrajectoryData,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract context (x, y) at rollout depth from TrajectoryData."""
    if not (0 <= depth <= traj.depth):
        raise ValueError(f"depth {depth} out of range [0, {traj.depth}]")
    end = traj.n_base + depth
    return traj.x_buf[:end], traj.y_buf[:end]


def belief_at_depth_torch(
    pred_rule,
    traj: TrajectoryData,
    depth: int,
    x_q: np.ndarray,
) -> torch.Tensor:
    """Torch belief at a specific depth for one trajectory."""
    x_ctx, y_ctx = get_context_at_depth(traj, depth)
    return pred_rule.get_belief_torch(
        x_query=np.atleast_2d(x_q),
        x_context=x_ctx,
        y_context=y_ctx,
    )


def belief_at_depth_torch_batched(
    pred_rule,
    trajs: list[TrajectoryData],
    depth: int,
    x_q: np.ndarray,
) -> torch.Tensor:
    """Torch batched belief at a specific depth for B trajectories."""
    x_q = np.atleast_2d(x_q)
    x_contexts = []
    y_contexts = []
    for traj in trajs:
        x_ctx, y_ctx = get_context_at_depth(traj, depth)
        x_contexts.append(x_ctx)
        y_contexts.append(y_ctx)
    return pred_rule.get_belief_torch_batched(
        x_query=x_q,
        x_contexts=x_contexts,
        y_contexts=y_contexts,
    )


def horizon_k_to_depth(k: int) -> int:
    """Map 1-based horizon k to continuation depth index (k-1)."""
    if k < 1:
        raise ValueError(f"horizon k must be >= 1, got {k}")
    return k - 1


def _validate_y_dtype(y: np.ndarray) -> None:
    """Validate that y is numeric."""
    if y.dtype.kind not in ("i", "u", "f"):
        raise TypeError(
            f"y.dtype={y.dtype} is not numeric. "
            "Classification labels must be integer-encoded before rollout. "
            "Use sklearn.preprocessing.LabelEncoder."
        )
