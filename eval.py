"""
Evaluation helpers (basic metrics + synthetic fixed-anchor evaluation).
"""

from dataclasses import dataclass
from typing import Any, Optional

import jax
import numpy as np
import torch
from jaxtyping import PRNGKeyArray

from lora import LoRAConfig, LoRALinear, get_tabpfn_model, inject_lora
from predictive_rule import ClassifierPredRule, PredictiveRule
from rollout import (
    belief_at_depth_torch_batched,
    build_prefix_batch_data,
    horizon_k_to_depth,
)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numpy softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class BasicMetrics:
    """기본 평가 메트릭."""
    accuracy: float
    nll: float         # mean negative log-likelihood
    ece: float         # expected calibration error
    n_samples: int
    n_bins: int = 15   # ECE bin count


def compute_basic_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> BasicMetrics:
    """
    ECE / NLL / Accuracy 계산.

    Args:
        probs:   (N, C) predicted probabilities
        y_true:  (N,) ground truth labels (0..C-1)
        n_bins:  ECE bin 수

    Returns:
        BasicMetrics
    """
    N = len(y_true)
    assert probs.shape[0] == N

    # Accuracy
    preds = probs.argmax(axis=1)
    accuracy = (preds == y_true).mean()

    # NLL: -log p(y_true)
    eps = 1e-8
    log_probs = np.log(probs[np.arange(N), y_true] + eps)
    nll = -log_probs.mean()

    # ECE (equal-width binning)
    confidences = probs.max(axis=1)
    correctness = (preds == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include confidence==0 in the first bin.
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correctness[mask].mean()
        ece += (n_in_bin / N) * abs(avg_acc - avg_conf)

    return BasicMetrics(
        accuracy=float(accuracy),
        nll=float(nll),
        ece=float(ece),
        n_samples=N,
        n_bins=n_bins,
    )


def compute_basic_metrics_regression(
    probs: np.ndarray,
    y_true: np.ndarray,
    pred_rule,
) -> BasicMetrics:
    """
    Regression용 기본 메트릭.

    - NLL: y_true를 bin index로 변환 후 -log p(bin)
    - Accuracy: N/A (0.0 반환)
    - ECE: bar-dist calibration (bin 확률 vs 실제 bin 적중률)

    Args:
        probs:      (N, n_bins) bar-distribution probabilities
        y_true:     (N,) continuous ground truth
        pred_rule:  RegressorPredRule (y_to_bin_index 사용)

    Returns:
        BasicMetrics
    """
    N = len(y_true)
    assert probs.shape[0] == N

    # y → bin index
    bin_idx = pred_rule.y_to_bin_index(y_true)  # (N,) int

    # NLL: -log p(true bin)
    eps = 1e-8
    log_probs = np.log(probs[np.arange(N), bin_idx] + eps)
    nll = -log_probs.mean()

    # "Accuracy": predicted mode bin == true bin (coarse metric)
    pred_bins = probs.argmax(axis=1)
    accuracy = (pred_bins == bin_idx).mean()

    # ECE: confidence vs bin-hit calibration
    confidences = probs.max(axis=1)
    correctness = (pred_bins == bin_idx).astype(float)
    n_ece_bins = 15
    bin_edges = np.linspace(0.0, 1.0, n_ece_bins + 1)
    ece = 0.0
    for i in range(n_ece_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include confidence==0 in the first bin.
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correctness[mask].mean()
        ece += (n_in_bin / N) * abs(avg_acc - avg_conf)

    return BasicMetrics(
        accuracy=float(accuracy),
        nll=float(nll),
        ece=float(ece),
        n_samples=N,
        n_bins=n_ece_bins,
    )


def evaluate_basic(
    pred_rule,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    use_torch: bool = True,
    task_type: str = "classification",
) -> BasicMetrics:
    """
    pred_rule로 기본 메트릭 계산.

    Args:
        pred_rule:  PredictiveRule instance
        x_context:  context dataset features
        y_context:  context dataset labels
        x_test:     test features
        y_test:     test labels (ground truth)
        use_torch:  True면 get_belief_torch (학습/eval 일관된 경로)
        task_type:  "classification" or "regression"

    Returns:
        BasicMetrics
    """
    # fit()은 항상 호출: _n_classes 초기화 + lock_weights 복원
    pred_rule.fit(x_context, y_context)

    if use_torch:
        with torch.no_grad():
            probs_t = pred_rule.get_belief_torch(x_test, x_context, y_context)
            probs = probs_t.cpu().numpy()
    else:
        probs = pred_rule.get_belief(x_test)
        if task_type == "regression":
            # get_belief returns logits for regression; convert to probs
            probs = _softmax_np(probs)

    if task_type == "regression":
        return compute_basic_metrics_regression(probs, y_test, pred_rule)
    else:
        return compute_basic_metrics(probs, y_test)


def _sample_indices_without_replacement(
    key: PRNGKeyArray,
    n_total: int,
    sample_size: int,
) -> np.ndarray:
    """Sample indices uniformly without replacement."""
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    size = int(np.clip(sample_size, 1, n_total))
    perm = np.array(jax.random.permutation(key, n_total))
    return perm[:size].astype(int)


def _split_synthetic_task_indices(
    *,
    key: PRNGKeyArray,
    y_task: np.ndarray,
    context_size: int,
    query_pool_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one synthetic task indices into context/query-pool/rollout-pool."""
    n_points = int(len(np.asarray(y_task)))
    if n_points < 3:
        raise ValueError(f"Synthetic task must have >=3 points, got {n_points}")
    n_ctx = int(np.clip(context_size, 1, n_points - 2))
    n_qpool = int(np.clip(query_pool_size, 1, n_points - n_ctx - 1))

    perm = np.asarray(jax.random.permutation(key, n_points)).astype(int)
    idx_ctx = perm[:n_ctx]
    idx_rem = perm[n_ctx:]
    idx_qpool = idx_rem[:n_qpool]
    idx_roll = idx_rem[n_qpool:]
    return idx_ctx, idx_qpool, idx_roll


def build_fixed_synthetic_anchor_suite(
    *,
    key: PRNGKeyArray,
    x_tasks: np.ndarray,
    y_tasks: np.ndarray,
    anchor_count: int,
    context_size: int,
    query_pool_size: int,
    queries_per_anchor: int,
    prefix_depths: tuple[int, ...],
    fixed_rollout_paths: bool,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    Optional[dict[tuple[int, int], PRNGKeyArray]],
    PRNGKeyArray,
]:
    """
    Precompute fixed synthetic anchor suite.

    Returns:
      - anchor_contexts: [(x_ctx, y_ctx_local)]
      - anchor_contexts_global: [(x_ctx, y_ctx_global)]
      - anchor_context_class_ids: [local->global class id mapping]
      - anchor_query_banks_x
      - anchor_query_banks_y (global labels)
      - anchor_rollout_pools
      - fixed_rollout_keys[(anchor_idx, prefix_n)] if requested
      - updated key
    """
    n_tasks = len(x_tasks)
    if n_tasks == 0:
        raise ValueError("Held-out synthetic task pool is empty.")

    contexts: list[tuple[np.ndarray, np.ndarray]] = []
    contexts_global: list[tuple[np.ndarray, np.ndarray]] = []
    context_class_ids: list[np.ndarray] = []
    query_banks_x: list[np.ndarray] = []
    query_banks_y: list[np.ndarray] = []
    rollout_pools: list[np.ndarray] = []
    fixed_keys: Optional[dict[tuple[int, int], PRNGKeyArray]] = (
        {} if fixed_rollout_paths else None
    )

    if n_tasks >= anchor_count:
        key, k_idx = jax.random.split(key)
        anchor_task_idx = _sample_indices_without_replacement(
            k_idx, n_total=n_tasks, sample_size=anchor_count
        )
    else:
        key, k_idx = jax.random.split(key)
        anchor_task_idx = np.asarray(
            jax.random.choice(k_idx, n_tasks, shape=(anchor_count,), replace=True)
        ).astype(int)

    for a, t_idx in enumerate(anchor_task_idx.tolist()):
        x_task = np.asarray(x_tasks[int(t_idx)])
        y_task = np.asarray(y_tasks[int(t_idx)]).astype(int)
        key, k_split = jax.random.split(key)
        idx_ctx, idx_qpool, idx_roll = _split_synthetic_task_indices(
            key=k_split,
            y_task=y_task,
            context_size=context_size,
            query_pool_size=query_pool_size,
        )
        x_ctx_a = x_task[idx_ctx]
        y_ctx_global = y_task[idx_ctx]
        class_ids_a, y_ctx_local = np.unique(y_ctx_global, return_inverse=True)
        y_ctx_a = y_ctx_local.astype(np.int64)
        x_roll_a = x_task[idx_roll]
        contexts.append((x_ctx_a, y_ctx_a))
        contexts_global.append((x_ctx_a, np.asarray(y_ctx_global).astype(np.int64)))
        context_class_ids.append(np.asarray(class_ids_a, dtype=np.int64))
        rollout_pools.append(x_roll_a)

        q_src_idx = idx_qpool if len(idx_qpool) > 0 else idx_roll
        key, k_q = jax.random.split(key)
        if len(q_src_idx) >= queries_per_anchor:
            q_local = _sample_indices_without_replacement(
                k_q, n_total=len(q_src_idx), sample_size=queries_per_anchor
            )
            q_idx = q_src_idx[q_local]
        else:
            pos = np.asarray(
                jax.random.choice(
                    k_q,
                    len(q_src_idx),
                    shape=(queries_per_anchor,),
                    replace=True,
                )
            ).astype(int)
            q_idx = q_src_idx[pos]
        query_banks_x.append(np.asarray(x_task[q_idx]))
        query_banks_y.append(np.asarray(y_task[q_idx]).astype(np.int64))

        if fixed_keys is not None:
            for n in prefix_depths:
                key, k_roll = jax.random.split(key)
                fixed_keys[(int(a), int(n))] = k_roll

    return (
        contexts,
        contexts_global,
        context_class_ids,
        query_banks_x,
        query_banks_y,
        rollout_pools,
        fixed_keys,
        key,
    )


def _emd_1d_probs_np(p: np.ndarray, q: np.ndarray) -> float:
    """L1 distance over class-probability vectors (used for EMD logging)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p = p / np.clip(p.sum(), 1e-12, None)
    q = q / np.clip(q.sum(), 1e-12, None)
    return float(np.sum(np.abs(p - q)))


def _build_prefix_continuation_map(
    *,
    key: PRNGKeyArray,
    pred_rule_sampling: PredictiveRule,
    x_context: np.ndarray,
    y_context: np.ndarray,
    prefix_depths: tuple[int, ...],
    continuation_depth: int,
    n_continuations: int,
    fixed_rollout_keys: Optional[dict[int, PRNGKeyArray]] = None,
    x_sampling_pool: Optional[np.ndarray] = None,
    x_sample_without_replacement: bool = False,
) -> tuple[dict[int, list], PRNGKeyArray]:
    continuations_by_prefix: dict[int, list] = {}
    for n in prefix_depths:
        n_int = int(n)
        if fixed_rollout_keys is not None:
            subkey = fixed_rollout_keys[n_int]
        else:
            key, subkey = jax.random.split(key)
        prefix_batch = build_prefix_batch_data(
            key=subkey,
            pred_rule_sampling=pred_rule_sampling,
            x0=x_context,
            y0=y_context,
            prefix_depth=n_int,
            continuation_depth=continuation_depth,
            n_continuations=n_continuations,
            x_sampling_pool=x_sampling_pool,
            x_sample_without_replacement=x_sample_without_replacement,
        )
        continuations_by_prefix[n_int] = prefix_batch.continuations
    return continuations_by_prefix, key


def _compute_query_marginals_for_ks(
    *,
    pred_rule_train: PredictiveRule,
    continuations: list,
    ks: tuple[int, ...],
    x_query: np.ndarray,
    no_grad_keys: set[int],
) -> dict[int, torch.Tensor]:
    p_hat: dict[int, torch.Tensor] = {}
    for k in ks:
        depth = horizon_k_to_depth(int(k))
        if int(k) in no_grad_keys:
            with torch.no_grad():
                b_batch = belief_at_depth_torch_batched(
                    pred_rule_train, continuations, depth, x_query
                )
        else:
            b_batch = belief_at_depth_torch_batched(
                pred_rule_train, continuations, depth, x_query
            )
        p_hat[int(k)] = b_batch.mean(dim=0).squeeze(0)
    return p_hat


def _compute_emd_fixed_anchor_suite(
    *,
    key: PRNGKeyArray,
    pred_rule_train: PredictiveRule,
    pred_rule_sampling: PredictiveRule,
    anchor_contexts: list[tuple[np.ndarray, np.ndarray]],
    anchor_query_banks: list[np.ndarray],
    prefix_depths: tuple[int, ...],
    k_values: tuple[int, ...],
    continuation_depth: int,
    n_continuations: int,
    fixed_rollout_keys: Optional[dict[tuple[int, int], PRNGKeyArray]] = None,
    anchor_rollout_pools: Optional[list[np.ndarray]] = None,
) -> tuple[float, float, int, PRNGKeyArray]:
    if anchor_rollout_pools is None:
        raise ValueError(
            "Synthetic-only EMD requires anchor_rollout_pools for rollout x sampling."
        )
    eval_ks = tuple(sorted({int(k) for k in k_values if int(k) > 1}))
    if len(eval_ks) == 0:
        return float("nan"), float("nan"), 0, key

    all_k = tuple(sorted({1, *eval_ks}))
    emd_needed_depth = max(horizon_k_to_depth(int(k)) for k in all_k)
    if emd_needed_depth > continuation_depth:
        raise ValueError(
            f"EMD needs continuation depth {emd_needed_depth} from k values {all_k}, "
            f"but continuation_depth={continuation_depth}"
        )

    all_k_set = set(all_k)
    anchor_scores: list[float] = []
    coverage = 0

    for a, ((x_ctx_a, y_ctx_a), q_bank_a) in enumerate(
        zip(anchor_contexts, anchor_query_banks)
    ):
        fixed_keys_for_prefix: Optional[dict[int, PRNGKeyArray]] = None
        if fixed_rollout_keys is not None:
            fixed_keys_for_prefix = {
                int(n): fixed_rollout_keys[(int(a), int(n))]
                for n in prefix_depths
            }
        rollout_pool_a = np.asarray(anchor_rollout_pools[a])
        conts_by_prefix, key = _build_prefix_continuation_map(
            key=key,
            pred_rule_sampling=pred_rule_sampling,
            x_context=x_ctx_a,
            y_context=y_ctx_a,
            prefix_depths=prefix_depths,
            continuation_depth=emd_needed_depth,
            n_continuations=n_continuations,
            fixed_rollout_keys=fixed_keys_for_prefix,
            x_sampling_pool=rollout_pool_a,
            x_sample_without_replacement=True,
        )

        per_anchor_scores: list[float] = []
        for q_pos in range(len(q_bank_a)):
            q2 = np.atleast_2d(q_bank_a[q_pos])
            for n in prefix_depths:
                conts = conts_by_prefix[int(n)]
                p_hat = _compute_query_marginals_for_ks(
                    pred_rule_train=pred_rule_train,
                    continuations=conts,
                    ks=all_k,
                    x_query=q2,
                    no_grad_keys=all_k_set,
                )
                p1 = p_hat[1].detach().cpu().numpy().astype(np.float64)
                dists = [
                    _emd_1d_probs_np(
                        p_hat[int(k)].detach().cpu().numpy().astype(np.float64),
                        p1,
                    )
                    for k in eval_ks
                ]
                if len(dists) > 0:
                    per_anchor_scores.append(float(np.mean(dists)))
                    coverage += 1

        if len(per_anchor_scores) > 0:
            anchor_scores.append(float(np.mean(per_anchor_scores)))

    if len(anchor_scores) == 0:
        return float("nan"), float("nan"), 0, key

    vals = np.asarray(anchor_scores, dtype=np.float64)
    return float(np.mean(vals)), float(np.std(vals)), int(coverage), key


def _snapshot_lora_adapters(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            snapshot[name] = {
                "lora_A": module.lora_A.detach().cpu().clone(),
                "lora_B": module.lora_B.detach().cpu().clone(),
            }
    return snapshot


def _load_lora_adapters(
    model: torch.nn.Module,
    snapshot: dict[str, dict[str, torch.Tensor]],
) -> None:
    name_to_module = dict(model.named_modules())
    with torch.no_grad():
        for name, tensors in snapshot.items():
            module = name_to_module.get(name)
            if not isinstance(module, LoRALinear):
                continue
            module.lora_A.copy_(tensors["lora_A"].to(module.lora_A.device, dtype=module.lora_A.dtype))
            module.lora_B.copy_(tensors["lora_B"].to(module.lora_B.device, dtype=module.lora_B.dtype))


def _compute_fixed_anchor_suite_eval_with_current_lora(
    *,
    key: PRNGKeyArray,
    train_model: torch.nn.Module,
    categorical_x: list[bool],
    n_estimators: int,
    lora_config: LoRAConfig,
    base_model_state_dict: Optional[dict[str, torch.Tensor]] = None,
    anchor_contexts: list[tuple[np.ndarray, np.ndarray]],
    anchor_context_class_ids: list[np.ndarray],
    anchor_query_banks_x: list[np.ndarray],
    anchor_query_banks_y: list[np.ndarray],
    global_num_classes: int,
    prefix_depths: tuple[int, ...],
    k_values: tuple[int, ...],
    continuation_depth: int,
    n_continuations: int,
    fixed_rollout_keys: Optional[dict[tuple[int, int], PRNGKeyArray]] = None,
    anchor_rollout_pools: Optional[list[np.ndarray]] = None,
) -> tuple[float, float, int, float, float, float, PRNGKeyArray]:
    if anchor_rollout_pools is None:
        raise ValueError(
            "Synthetic-only EMD requires anchor_rollout_pools for rollout x sampling."
        )

    lora_snapshot = _snapshot_lora_adapters(train_model)
    c_global = int(global_num_classes)
    if c_global < 2:
        raise ValueError(f"global_num_classes must be >=2, got {c_global}")

    anchor_emd_means: list[float] = []
    total_coverage = 0
    acc_vals: list[float] = []
    nll_vals: list[float] = []
    ece_vals: list[float] = []

    n_anchor = len(anchor_contexts)
    if not (
        len(anchor_context_class_ids) == n_anchor
        and len(anchor_query_banks_x) == n_anchor
        and len(anchor_query_banks_y) == n_anchor
        and len(anchor_rollout_pools) == n_anchor
    ):
        raise ValueError("Fixed-anchor eval suite length mismatch.")

    for a in range(n_anchor):
        x_ctx_a, y_ctx_a = anchor_contexts[a]
        local_classes = np.asarray(anchor_context_class_ids[a]).astype(int)
        x_q = np.asarray(anchor_query_banks_x[a], dtype=np.float32)
        y_q = np.asarray(anchor_query_banks_y[a]).astype(int)
        if len(x_q) == 0:
            continue

        sampling_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
        train_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
        if base_model_state_dict is not None:
            locked_state = {
                k: v.clone().cpu() for k, v in base_model_state_dict.items()
            }
            sampling_rule._locked_state_dict = {k: v.clone() for k, v in locked_state.items()}
            train_rule._locked_state_dict = locked_state

        sampling_rule.fit(x_ctx_a, y_ctx_a)
        train_rule.fit(x_ctx_a, y_ctx_a)
        train_rule_model = get_tabpfn_model(train_rule)
        inject_lora(train_rule_model, lora_config)
        _load_lora_adapters(train_rule_model, lora_snapshot)

        with torch.no_grad():
            probs_local = train_rule.get_belief_torch(x_q, x_ctx_a, y_ctx_a).cpu().numpy()
        probs_local = np.asarray(probs_local, dtype=np.float64)
        if probs_local.shape[1] < len(local_classes):
            raise ValueError(
                f"Local prob dim {probs_local.shape[1]} is smaller than local class count "
                f"{len(local_classes)} at anchor {a}."
            )
        probs_global = np.zeros((len(x_q), c_global), dtype=np.float64)
        probs_global[:, local_classes] = probs_local[:, : len(local_classes)]
        probs_global = probs_global / np.clip(
            probs_global.sum(axis=1, keepdims=True), 1e-12, None
        )
        m = compute_basic_metrics(probs_global, y_q)
        acc_vals.append(float(m.accuracy))
        nll_vals.append(float(m.nll))
        ece_vals.append(float(m.ece))

        anchor_fixed_keys: Optional[dict[tuple[int, int], PRNGKeyArray]] = None
        if fixed_rollout_keys is not None:
            anchor_fixed_keys = {
                (0, int(n)): fixed_rollout_keys[(int(a), int(n))]
                for n in prefix_depths
            }

        emd_mean_a, _emd_std_a, cov_a, key = _compute_emd_fixed_anchor_suite(
            key=key,
            pred_rule_train=train_rule,
            pred_rule_sampling=sampling_rule,
            anchor_contexts=[(x_ctx_a, y_ctx_a)],
            anchor_query_banks=[x_q],
            prefix_depths=prefix_depths,
            k_values=k_values,
            continuation_depth=continuation_depth,
            n_continuations=n_continuations,
            fixed_rollout_keys=anchor_fixed_keys,
            anchor_rollout_pools=[np.asarray(anchor_rollout_pools[a])],
        )
        if np.isfinite(emd_mean_a):
            anchor_emd_means.append(float(emd_mean_a))
        total_coverage += int(cov_a)

    if len(anchor_emd_means) == 0:
        emd_mean = float("nan")
        emd_std = float("nan")
    else:
        emd_vals = np.asarray(anchor_emd_means, dtype=np.float64)
        emd_mean = float(np.mean(emd_vals))
        emd_std = float(np.std(emd_vals))

    if len(acc_vals) == 0:
        acc = float("nan")
        nll = float("nan")
        ece = float("nan")
    else:
        acc = float(np.mean(np.asarray(acc_vals, dtype=np.float64)))
        nll = float(np.mean(np.asarray(nll_vals, dtype=np.float64)))
        ece = float(np.mean(np.asarray(ece_vals, dtype=np.float64)))

    return emd_mean, emd_std, int(total_coverage), acc, nll, ece, key


def _compute_fixed_anchor_suite_eval_for_model_pair(
    *,
    key: PRNGKeyArray,
    categorical_x: list[bool],
    n_estimators: int,
    sampling_model_state_dict: Optional[dict[str, torch.Tensor]] = None,
    belief_model_state_dict: Optional[dict[str, torch.Tensor]] = None,
    anchor_contexts: list[tuple[np.ndarray, np.ndarray]],
    anchor_context_class_ids: list[np.ndarray],
    anchor_query_banks_x: list[np.ndarray],
    anchor_query_banks_y: list[np.ndarray],
    global_num_classes: int,
    prefix_depths: tuple[int, ...],
    k_values: tuple[int, ...],
    continuation_depth: int,
    n_continuations: int,
    fixed_rollout_keys: Optional[dict[tuple[int, int], PRNGKeyArray]] = None,
    anchor_rollout_pools: Optional[list[np.ndarray]] = None,
) -> tuple[float, float, int, float, float, float, PRNGKeyArray]:
    """
    Exact fixed-anchor eval for a model pair.

    Typical use:
      - base sampling + tuned belief:
          sampling_model_state_dict=None
          belief_model_state_dict=<merged tuned state>
    """
    if anchor_rollout_pools is None:
        raise ValueError(
            "Synthetic-only EMD requires anchor_rollout_pools for rollout x sampling."
        )

    c_global = int(global_num_classes)
    if c_global < 2:
        raise ValueError(f"global_num_classes must be >=2, got {c_global}")

    anchor_emd_means: list[float] = []
    total_coverage = 0
    acc_vals: list[float] = []
    nll_vals: list[float] = []
    ece_vals: list[float] = []

    n_anchor = len(anchor_contexts)
    if not (
        len(anchor_context_class_ids) == n_anchor
        and len(anchor_query_banks_x) == n_anchor
        and len(anchor_query_banks_y) == n_anchor
        and len(anchor_rollout_pools) == n_anchor
    ):
        raise ValueError("Fixed-anchor eval suite length mismatch.")

    for a in range(n_anchor):
        x_ctx_a, y_ctx_a = anchor_contexts[a]
        local_classes = np.asarray(anchor_context_class_ids[a]).astype(int)
        x_q = np.asarray(anchor_query_banks_x[a], dtype=np.float32)
        y_q = np.asarray(anchor_query_banks_y[a]).astype(int)
        if len(x_q) == 0:
            continue

        sampling_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
        train_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
        if sampling_model_state_dict is not None:
            sampling_rule._locked_state_dict = {
                k: v.clone().cpu() for k, v in sampling_model_state_dict.items()
            }
        if belief_model_state_dict is not None:
            train_rule._locked_state_dict = {
                k: v.clone().cpu() for k, v in belief_model_state_dict.items()
            }

        sampling_rule.fit(x_ctx_a, y_ctx_a)
        train_rule.fit(x_ctx_a, y_ctx_a)

        with torch.no_grad():
            probs_local = train_rule.get_belief_torch(x_q, x_ctx_a, y_ctx_a).cpu().numpy()
        probs_local = np.asarray(probs_local, dtype=np.float64)
        if probs_local.shape[1] < len(local_classes):
            raise ValueError(
                f"Local prob dim {probs_local.shape[1]} is smaller than local class count "
                f"{len(local_classes)} at anchor {a}."
            )
        probs_global = np.zeros((len(x_q), c_global), dtype=np.float64)
        probs_global[:, local_classes] = probs_local[:, : len(local_classes)]
        probs_global = probs_global / np.clip(
            probs_global.sum(axis=1, keepdims=True), 1e-12, None
        )
        m = compute_basic_metrics(probs_global, y_q)
        acc_vals.append(float(m.accuracy))
        nll_vals.append(float(m.nll))
        ece_vals.append(float(m.ece))

        anchor_fixed_keys: Optional[dict[tuple[int, int], PRNGKeyArray]] = None
        if fixed_rollout_keys is not None:
            anchor_fixed_keys = {
                (0, int(n)): fixed_rollout_keys[(int(a), int(n))]
                for n in prefix_depths
            }

        emd_mean_a, _emd_std_a, cov_a, key = _compute_emd_fixed_anchor_suite(
            key=key,
            pred_rule_train=train_rule,
            pred_rule_sampling=sampling_rule,
            anchor_contexts=[(x_ctx_a, y_ctx_a)],
            anchor_query_banks=[x_q],
            prefix_depths=prefix_depths,
            k_values=k_values,
            continuation_depth=continuation_depth,
            n_continuations=n_continuations,
            fixed_rollout_keys=anchor_fixed_keys,
            anchor_rollout_pools=[np.asarray(anchor_rollout_pools[a])],
        )
        if np.isfinite(emd_mean_a):
            anchor_emd_means.append(float(emd_mean_a))
        total_coverage += int(cov_a)

    if len(anchor_emd_means) == 0:
        emd_mean = float("nan")
        emd_std = float("nan")
    else:
        emd_vals = np.asarray(anchor_emd_means, dtype=np.float64)
        emd_mean = float(np.mean(emd_vals))
        emd_std = float(np.std(emd_vals))

    if len(acc_vals) == 0:
        acc = float("nan")
        nll = float("nan")
        ece = float("nan")
    else:
        acc = float(np.mean(np.asarray(acc_vals, dtype=np.float64)))
        nll = float(np.mean(np.asarray(nll_vals, dtype=np.float64)))
        ece = float(np.mean(np.asarray(ece_vals, dtype=np.float64)))

    return emd_mean, emd_std, int(total_coverage), acc, nll, ece, key


@dataclass
class SCMetric:
    """Self-consistency 평가 메트릭."""
    l2_mean: float        # ||p_early - p_late_mean||_2 평균 (over B)
    kl_mean: float        # KL(p_late_mean || p_early) 평균 (over B)
    l2_std: float         # l2 표준편차
    kl_std: float         # kl 표준편차
    B: int                # continuation 수
    k1: int               # early horizon
    k2: int               # late horizon


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL(p || q) = Σ p_c * log(p_c / q_c)"""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float((p * np.log(p / q)).sum())


def compute_sc_metric(
    pred_rule,
    prefix_batch,
    horizon,
    x_q: np.ndarray,
    use_torch: bool = False,
) -> SCMetric:
    """
    SC metric: early-late belief 불일치 측정.

    같은 D_n에서 continuation B개를 뽑고,
    ||p_{k1}^{(b)} - p̂^{(k2)}||_2 와 KL(p̂^{(k2)} || p_{k1}^{(b)}) 평균.

    Args:
        pred_rule:    PredictiveRule (inference용 또는 LoRA-injected)
        prefix_batch: PrefixBatchData 또는 PrefixBatch
        horizon:      HorizonPair
        x_q:          (1, d) query point
        use_torch:    True면 torch path 사용

    Returns:
        SCMetric
    """
    k1, k2 = horizon.k1, horizon.k2
    x_q = np.atleast_2d(x_q)

    if use_torch:
        from rollout import belief_at_depth_torch_batched, horizon_k_to_depth
        conts = prefix_batch.continuations
        d1 = horizon_k_to_depth(k1)
        d2 = horizon_k_to_depth(k2)
        with torch.no_grad():
            early_batch = belief_at_depth_torch_batched(pred_rule, conts, d1, x_q)  # (B, M, C)
            late_batch = belief_at_depth_torch_batched(pred_rule, conts, d2, x_q)   # (B, M, C)
        early_arr = early_batch.squeeze(1).cpu().numpy()  # (B, C)
        late_arr = late_batch.squeeze(1).cpu().numpy()     # (B, C)
    else:
        d1 = max(k1 - 1, 0)
        d2 = max(k2 - 1, 0)
        early_beliefs = []
        late_beliefs = []
        for cont in prefix_batch.continuations:
            early_beliefs.append(cont.beliefs[d1])
            late_beliefs.append(cont.beliefs[d2])
        early_arr = np.stack(early_beliefs)   # (B, C)
        late_arr = np.stack(late_beliefs)     # (B, C)

    late_mean = late_arr.mean(axis=0)     # (C,)

    # L2 distances
    l2_per_b = np.linalg.norm(early_arr - late_mean[None, :], axis=1)
    # KL divergences
    kl_per_b = np.array([_kl_divergence(late_mean, early_arr[b]) for b in range(len(early_arr))])

    return SCMetric(
        l2_mean=float(l2_per_b.mean()),
        kl_mean=float(kl_per_b.mean()),
        l2_std=float(l2_per_b.std()),
        kl_std=float(kl_per_b.std()),
        B=len(early_arr),
        k1=k1,
        k2=k2,
    )


@dataclass
class EvalComparison:
    """학습 전/후 비교 결과."""
    before: BasicMetrics
    after: BasicMetrics
    sc_before: Optional[SCMetric] = None
    sc_after: Optional[SCMetric] = None


def print_comparison(comp: EvalComparison) -> None:
    """학습 전/후 비교 출력."""
    print("\n── Evaluation: Before vs After ──")
    print(f"  {'Metric':<12} {'Before':>10} {'After':>10} {'Δ':>10}")
    print(f"  {'─'*44}")

    b, a = comp.before, comp.after
    for name, bv, av in [
        ("Accuracy", b.accuracy, a.accuracy),
        ("NLL", b.nll, a.nll),
        ("ECE", b.ece, a.ece),
    ]:
        delta = av - bv
        direction = "↑" if (name == "Accuracy" and delta > 0) or \
                         (name != "Accuracy" and delta < 0) else "↓" if delta != 0 else "─"
        print(f"  {name:<12} {bv:>10.4f} {av:>10.4f} {delta:>+9.4f} {direction}")

    if comp.sc_before and comp.sc_after:
        print(f"\n  SC Metric (k1={comp.sc_before.k1}, k2={comp.sc_before.k2}):")
        print(f"    L2: {comp.sc_before.l2_mean:.4f} → {comp.sc_after.l2_mean:.4f} "
              f"(Δ={comp.sc_after.l2_mean - comp.sc_before.l2_mean:+.4f})")
        print(f"    KL: {comp.sc_before.kl_mean:.4f} → {comp.sc_after.kl_mean:.4f} "
              f"(Δ={comp.sc_after.kl_mean - comp.sc_before.kl_mean:+.4f})")
