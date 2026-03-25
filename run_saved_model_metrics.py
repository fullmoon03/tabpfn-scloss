"""
run_saved_model_metrics.py

Evaluate baseline vs a saved merged model state on the same split used by
run_classification.py, with reproducible randomness.
"""

"""
python run_saved_model_metrics.py \
    --model-path "/home/boreum/project/tabpfn-scloss/vehicle_model/merged_model_state_openml_54_20260303_123657.pt"
"""


import argparse
import os
from typing import Any, Optional

import jax
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eval import compute_basic_metrics, _compute_emd_fixed_anchor_suite
from predictive_rule import ClassifierPredRule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved merged model .pt (e.g., merged_model_state_*.pt).",
    )
    parser.add_argument("--openml-data-id", type=int, default=54)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--metric-context-size",
        type=int,
        default=100,
        help="Context size n sampled from X_train for metric estimation.",
    )
    parser.add_argument(
        "--metric-context-repeats",
        type=int,
        default=4,
        help="Number of sampled contexts b for metric averaging.",
    )
    parser.add_argument(
        "--enable-emd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict fixed-anchor EMD comparison (default: enabled).",
    )
    parser.add_argument("--emd-anchor-count", type=int, default=3)
    parser.add_argument("--emd-context-size", type=int, default=100)
    parser.add_argument("--emd-queries-per-anchor", type=int, default=4)
    parser.add_argument("--emd-prefix-depths", type=str, default="0,2,4")
    parser.add_argument("--emd-k-values", type=str, default="3,7,11,15")
    parser.add_argument("--emd-continuation-depth", type=int, default=30)
    parser.add_argument("--emd-n-continuations", type=int, default=8)
    parser.add_argument(
        "--emd-fixed-rollout-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--emd-rng-fold-in", type=int, default=202)
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_openml_classification_dataset(
    *,
    data_id: int,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool], LabelEncoder, str]:
    X, y_raw = fetch_openml(data_id=data_id, as_frame=False, return_X_y=True)
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed), stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    categorical_x = [False] * int(X_train.shape[1])
    dataset_name = f"openml_{int(data_id)}"
    return X_train, X_test, y_train, y_test, categorical_x, le, dataset_name


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """Read merged model state_dict from common checkpoint formats."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        raw = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        raw = ckpt
    else:
        raise ValueError(
            "Unsupported checkpoint format. "
            "Use merged model checkpoint (merged_model_state_*.pt)."
        )

    state: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            state[str(k)] = v.detach().cpu().clone()
    if len(state) == 0:
        raise ValueError("Checkpoint contains no tensor state_dict entries.")
    return state


def _print_metrics(title: str, m: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Accuracy: {m['accuracy_mean']:.4f} ± {m['accuracy_std']:.4f}")
    print(f"  NLL:      {m['nll_mean']:.4f} ± {m['nll_std']:.4f}")
    print(f"  ECE:      {m['ece_mean']:.4f} ± {m['ece_std']:.4f}")
    print(f"  Repeats:  {int(m['n_repeats'])}")


def _parse_int_tuple(csv_text: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in str(csv_text).split(",") if v.strip() != ""]
    if len(vals) == 0:
        raise ValueError("Expected non-empty comma-separated integer list.")
    return tuple(vals)


def _sample_indices_without_replacement(
    key: Any,
    n_total: int,
    sample_size: int,
) -> np.ndarray:
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    size = int(np.clip(sample_size, 1, n_total))
    perm = np.asarray(jax.random.permutation(key, n_total)).astype(int)
    return perm[:size]


def _relabel_to_contiguous(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    _, y_local = np.unique(y, return_inverse=True)
    return y_local.astype(np.int64)


def _build_fixed_openml_emd_anchor_suite(
    *,
    key: Any,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    anchor_count: int,
    context_size: int,
    queries_per_anchor: int,
    prefix_depths: tuple[int, ...],
    fixed_rollout_paths: bool,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[np.ndarray],
    Optional[dict[tuple[int, int], Any]],
    list[np.ndarray],
    Any,
]:
    """
    Build fixed EMD anchor suite from OpenML test pool for synthetic-only EMD core.
    """
    x_pool = np.asarray(x_pool, dtype=np.float32)
    y_pool = np.asarray(y_pool).astype(int)
    n_total = len(x_pool)
    if n_total < 3:
        raise ValueError(f"Need at least 3 pool samples for EMD anchors, got {n_total}")

    if n_total >= int(anchor_count):
        key, k_idx = jax.random.split(key)
        anchor_indices = _sample_indices_without_replacement(
            k_idx, n_total=n_total, sample_size=int(anchor_count)
        )
    else:
        key, k_idx = jax.random.split(key)
        anchor_indices = np.asarray(
            jax.random.choice(k_idx, n_total, shape=(int(anchor_count),), replace=True)
        ).astype(int)

    contexts: list[tuple[np.ndarray, np.ndarray]] = []
    query_banks: list[np.ndarray] = []
    rollout_pools: list[np.ndarray] = []
    fixed_keys: Optional[dict[tuple[int, int], Any]] = (
        {} if bool(fixed_rollout_paths) else None
    )

    for a, anchor_idx in enumerate(anchor_indices.tolist()):
        key, k_split = jax.random.split(key)
        k_split = jax.random.fold_in(k_split, int(anchor_idx))
        perm = np.asarray(jax.random.permutation(k_split, n_total)).astype(int)
        n_ctx = int(np.clip(context_size, 1, n_total - 2))
        n_q = int(np.clip(queries_per_anchor, 1, n_total - n_ctx - 1))
        idx_ctx = perm[:n_ctx]
        idx_q = perm[n_ctx:n_ctx + n_q]
        idx_roll = perm[n_ctx + n_q:]
        if len(idx_roll) == 0:
            idx_roll = idx_q

        x_ctx_a = np.asarray(x_pool[idx_ctx], dtype=np.float32)
        y_ctx_a = _relabel_to_contiguous(y_pool[idx_ctx])
        x_q_a = np.asarray(x_pool[idx_q], dtype=np.float32)
        x_roll_a = np.asarray(x_pool[idx_roll], dtype=np.float32)

        contexts.append((x_ctx_a, y_ctx_a))
        query_banks.append(x_q_a)
        rollout_pools.append(x_roll_a)

        if fixed_keys is not None:
            for n in prefix_depths:
                key, k_roll = jax.random.split(key)
                fixed_keys[(int(a), int(n))] = k_roll

    return contexts, query_banks, fixed_keys, rollout_pools, key


def _print_emd(title: str, emd_mean: float, emd_std: float, emd_cov: int) -> None:
    print("\n" + "=" * 60)
    print(f"  {title} EMD")
    print("=" * 60)
    print(f"  EMD mean: {emd_mean:.6f}")
    print(f"  EMD std:  {emd_std:.6f}")
    print(f"  Coverage: {emd_cov}")


def _sample_context_subsets(
    *,
    n_train: int,
    context_size: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Sample b context subsets of size n from train indices."""
    if repeats < 1:
        raise ValueError(f"metric-context-repeats must be >=1, got {repeats}")
    n_ctx = int(np.clip(context_size, 1, n_train))
    rng = np.random.default_rng(seed + 4242)
    subsets = []
    for _ in range(repeats):
        if n_ctx == n_train:
            idx = np.arange(n_train, dtype=int)
            rng.shuffle(idx)
        else:
            idx = np.array(rng.choice(n_train, size=n_ctx, replace=False), dtype=int)
        subsets.append(idx)
    return np.stack(subsets, axis=0)


def _evaluate_metrics_over_context_subsets(
    *,
    locked_state: dict[str, torch.Tensor] | None,
    categorical_x: list[bool],
    n_estimators: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    context_subsets: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate Acc/NLL/ECE over multiple sampled train-context subsets and average.

    For each subset, local labels are remapped to contiguous 0..C_local-1 for fit(),
    then predicted probabilities are projected back to the global class axis.
    """
    n_classes_global = int(np.max(y_train)) + 1
    acc_vals: list[float] = []
    nll_vals: list[float] = []
    ece_vals: list[float] = []

    for idx in context_subsets:
        pred_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
        if locked_state is not None:
            pred_rule._locked_state_dict = {
                k: v.clone().cpu() for k, v in locked_state.items()
            }

        x_ctx = x_train[idx]
        y_ctx = y_train[idx]
        local_classes, y_local = np.unique(y_ctx.astype(int), return_inverse=True)
        y_local = y_local.astype(np.int64)

        pred_rule.fit(x_ctx, y_local)
        with torch.no_grad():
            probs_local = pred_rule.get_belief_torch(x_test, x_ctx, y_local).cpu().numpy()

        probs_global = np.zeros((probs_local.shape[0], n_classes_global), dtype=np.float64)
        probs_global[:, local_classes] = probs_local
        m = compute_basic_metrics(probs_global, y_test)
        acc_vals.append(float(m.accuracy))
        nll_vals.append(float(m.nll))
        ece_vals.append(float(m.ece))

    acc_arr = np.asarray(acc_vals, dtype=np.float64)
    nll_arr = np.asarray(nll_vals, dtype=np.float64)
    ece_arr = np.asarray(ece_vals, dtype=np.float64)
    return {
        "accuracy_mean": float(acc_arr.mean()),
        "accuracy_std": float(acc_arr.std()),
        "nll_mean": float(nll_arr.mean()),
        "nll_std": float(nll_arr.std()),
        "ece_mean": float(ece_arr.mean()),
        "ece_std": float(ece_arr.std()),
        "n_repeats": float(len(context_subsets)),
    }


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    locked_state = _extract_state_dict(ckpt)

    X_train, X_test, y_train, y_test, categorical_x, _, dataset_name = (
        _load_openml_classification_dataset(
            data_id=args.openml_data_id,
            seed=args.seed,
            test_size=args.test_size,
        )
    )
    emd_prefix_depths = _parse_int_tuple(args.emd_prefix_depths)
    emd_k_values = _parse_int_tuple(args.emd_k_values)
    context_subsets = _sample_context_subsets(
        n_train=len(X_train),
        context_size=args.metric_context_size,
        repeats=args.metric_context_repeats,
        seed=args.seed,
    )
    print(f"\nDataset: {dataset_name}")
    print(f"Model checkpoint: {model_path}")
    print(f"State tensors: {len(locked_state)}")
    print(
        "Metric contexts: "
        f"n={context_subsets.shape[1]}, b={context_subsets.shape[0]}"
    )

    baseline_metrics = _evaluate_metrics_over_context_subsets(
        locked_state=None,
        categorical_x=categorical_x,
        n_estimators=args.n_estimators,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        context_subsets=context_subsets,
    )
    _print_metrics("BASELINE", baseline_metrics)

    loaded_metrics = _evaluate_metrics_over_context_subsets(
        locked_state=locked_state,
        categorical_x=categorical_x,
        n_estimators=args.n_estimators,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        context_subsets=context_subsets,
    )
    _print_metrics("LOADED MODEL", loaded_metrics)

    print("\n" + "-" * 60)
    print("  Delta (loaded - baseline)")
    print("-" * 60)
    print(
        f"  Accuracy: "
        f"{loaded_metrics['accuracy_mean'] - baseline_metrics['accuracy_mean']:+.4f}"
    )
    print(
        f"  NLL:      "
        f"{loaded_metrics['nll_mean'] - baseline_metrics['nll_mean']:+.4f}"
    )
    print(
        f"  ECE:      "
        f"{loaded_metrics['ece_mean'] - baseline_metrics['ece_mean']:+.4f}"
    )

    if args.enable_emd:
        # Build train-fitted rules for EMD (keep EMD semantics unchanged).
        pred_rule_baseline_emd = ClassifierPredRule(
            categorical_x,
            n_estimators=args.n_estimators,
        )
        pred_rule_baseline_emd.fit(X_train, y_train)
        pred_rule_loaded_emd = ClassifierPredRule(
            categorical_x,
            n_estimators=args.n_estimators,
        )
        pred_rule_loaded_emd._locked_state_dict = {
            k: v.clone().cpu() for k, v in locked_state.items()
        }
        pred_rule_loaded_emd.fit(X_train, y_train)

        # Build fixed EMD suite once from test pool (shared for baseline/loaded).
        key_suite = jax.random.fold_in(
            jax.random.PRNGKey(args.seed),
            int(args.emd_rng_fold_in),
        )
        (
            anchor_contexts,
            anchor_query_banks,
            fixed_rollout_keys,
            anchor_rollout_pools,
            _,
        ) = _build_fixed_openml_emd_anchor_suite(
            key=key_suite,
            x_pool=np.asarray(X_test),
            y_pool=np.asarray(y_test),
            anchor_count=args.emd_anchor_count,
            context_size=args.emd_context_size,
            queries_per_anchor=args.emd_queries_per_anchor,
            prefix_depths=emd_prefix_depths,
            fixed_rollout_paths=bool(args.emd_fixed_rollout_paths),
        )

        # Sampling rule is shared across both comparisons.
        pred_rule_sampling = ClassifierPredRule(
            categorical_x,
            n_estimators=args.n_estimators,
        )

        key_eval = jax.random.PRNGKey(args.seed + 999)
        emd_base_mean, emd_base_std, emd_base_cov, _ = _compute_emd_fixed_anchor_suite(
            key=key_eval,
            pred_rule_train=pred_rule_baseline_emd,
            pred_rule_sampling=pred_rule_sampling,
            anchor_contexts=anchor_contexts,
            anchor_query_banks=anchor_query_banks,
            prefix_depths=emd_prefix_depths,
            k_values=emd_k_values,
            continuation_depth=args.emd_continuation_depth,
            n_continuations=args.emd_n_continuations,
            fixed_rollout_keys=fixed_rollout_keys,
            anchor_rollout_pools=anchor_rollout_pools,
        )
        _print_emd("BASELINE", emd_base_mean, emd_base_std, emd_base_cov)

        key_eval_loaded = jax.random.PRNGKey(args.seed + 999)
        emd_loaded_mean, emd_loaded_std, emd_loaded_cov, _ = _compute_emd_fixed_anchor_suite(
            key=key_eval_loaded,
            pred_rule_train=pred_rule_loaded_emd,
            pred_rule_sampling=pred_rule_sampling,
            anchor_contexts=anchor_contexts,
            anchor_query_banks=anchor_query_banks,
            prefix_depths=emd_prefix_depths,
            k_values=emd_k_values,
            continuation_depth=args.emd_continuation_depth,
            n_continuations=args.emd_n_continuations,
            fixed_rollout_keys=fixed_rollout_keys,
            anchor_rollout_pools=anchor_rollout_pools,
        )
        _print_emd("LOADED MODEL", emd_loaded_mean, emd_loaded_std, emd_loaded_cov)

        print("\n" + "-" * 60)
        print("  Delta EMD (loaded - baseline)")
        print("-" * 60)
        print(f"  EMD mean: {emd_loaded_mean - emd_base_mean:+.6f}")
        print(f"  EMD std:  {emd_loaded_std - emd_base_std:+.6f}")
        print(f"  Coverage: {emd_loaded_cov - emd_base_cov:+d}")


if __name__ == "__main__":
    main()
