"""
run_cross_dataset_lora_transfer.py

Cross-dataset transfer evaluation for a single LoRA adapter:
  - Base TabPFN
  - Base + alpha-scaled LoRA, alpha in {0.5, 1.0}

Evaluates on:
  - OpenML 188 (Eucalyptus)
  - OpenML 44  (Splice, subsample 1000 by default)
  - and more

Metrics:
  - Accuracy / NLL / ECE
  - EMD (strict fixed-anchor path)
  - Belief drift visualization grids
"""

"""
python run_cross_dataset_lora_transfer.py \
  --adapter-path "/home/boreum/project/tabpfn-scloss/vehicle_model/lora_adapter_openml_54_20260303_123657.pt"


python run_cross_dataset_lora_transfer.py \
  --adapter-path "/home/boreum/project/tabpfn-scloss/synthetic_model/lora_adapter_generated_train_800_seed42_20260316_053443.pt"
  
  

  """



import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import jax
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from eval import evaluate_basic, _compute_emd_fixed_anchor_suite
from lora import get_tabpfn_model
from predictive_rule import ClassifierPredRule
from run_classification import set_global_seeds
from run_saved_model_rollout_compare import _run_fixed_queries_rollout_grid_analysis


@dataclass(frozen=True)
class DatasetSpec:
    data_id: int
    name_hint: str
    subsample_n: Optional[int] = None
    emd_context_size: Optional[int] = None  # None => use global default (--emd-context-size)
    rollout_base_n: Optional[int] = None  # None => use global default (--rollout-base-n)


DEFAULT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(54, "vehicle", None),
    DatasetSpec(188, "eucalyptus", None),
    # DatasetSpec(44, "splice", 1000),
    # DatasetSpec(11, "balance_scale", None),           # 이미 EMD 낮고 acc 너무 높음
    DatasetSpec(36, "segment", None),
    DatasetSpec(181, "yeast", None),
    DatasetSpec(23, "contraceptive", 1000),
)


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return s if s else "dataset"


def _encode_features_for_tabpfn(X: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    """Same encoding rule used in run_classification.py."""
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={X_arr.shape}")

    n_samples, n_features = X_arr.shape
    encoded_cols = []
    categorical_x: list[bool] = []
    for j in range(n_features):
        col = X_arr[:, j]
        if np.issubdtype(col.dtype, np.number):
            encoded_cols.append(col.astype(np.float32))
            categorical_x.append(False)
        else:
            _, inv = np.unique(col.astype(str), return_inverse=True)
            encoded_cols.append(inv.astype(np.float32))
            categorical_x.append(True)

    X_enc = np.column_stack(encoded_cols).astype(np.float32)
    if X_enc.shape[0] != n_samples:
        raise RuntimeError("Feature encoding produced invalid sample count.")
    return X_enc, categorical_x


def _subsample_stratified(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n >= len(y):
        return X, y
    idx_all = np.arange(len(y), dtype=int)
    try:
        idx_keep, _ = train_test_split(
            idx_all,
            train_size=n,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        idx_keep, _ = train_test_split(
            idx_all,
            train_size=n,
            random_state=seed,
            stratify=None,
        )
    idx_keep = np.asarray(idx_keep, dtype=int)
    return X[idx_keep], y[idx_keep]


def _filter_top_k_classes_if_many(
    X: np.ndarray,
    y_raw: np.ndarray,
    *,
    min_classes: int = 10,
    top_k: int = 4,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """
    If class count >= min_classes, keep only samples from top-k frequent classes.

    Returns:
      (X_filtered, y_raw_filtered, applied, original_n_classes)
    """
    y_raw = np.asarray(y_raw)
    uniq, counts = np.unique(y_raw, return_counts=True)
    n_classes = int(len(uniq))
    if n_classes < int(min_classes):
        return X, y_raw, False, n_classes

    order = np.argsort(-counts)  # frequency-desc
    keep_labels = uniq[order[: int(top_k)]]
    keep_mask = np.isin(y_raw, keep_labels)
    return X[keep_mask], y_raw[keep_mask], True, n_classes


def load_openml_split(
    *,
    data_id: int,
    test_size: float,
    seed: int,
    subsample_n: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool], LabelEncoder, str]:
    data = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    X_raw, y_raw = data.data, data.target
    dataset_name = getattr(data, "name", f"openml_{data_id}")

    X_enc, categorical_x = _encode_features_for_tabpfn(X_raw)
    X_enc, y_raw, topk_applied, orig_n_classes = _filter_top_k_classes_if_many(
        X_enc,
        y_raw,
        min_classes=7,
        top_k=4,
    )
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if topk_applied:
        print(
            f"  [class-filter] {dataset_name}: original_classes={orig_n_classes} "
            f"-> kept top-4 frequent classes (samples={len(y)})"
        )

    if subsample_n is not None and int(subsample_n) > 0:
        X_enc, y = _subsample_stratified(X_enc, y, int(subsample_n), seed)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=test_size, random_state=seed, stratify=None
        )

    return X_train, X_test, y_train, y_test, categorical_x, le, dataset_name


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
    Build fixed EMD anchors from OpenML test pool for current synthetic-only EMD core.
    """
    x_pool = np.asarray(x_pool, dtype=np.float32)
    y_pool = np.asarray(y_pool).astype(int)
    n_total = len(x_pool)
    if n_total < 3:
        raise ValueError(f"Need at least 3 pool samples for EMD anchors, got {n_total}")

    if n_total >= int(anchor_count):
        key, k_idx = jax.random.split(key)
        anchor_task_idx = _sample_indices_without_replacement(
            k_idx, n_total=n_total, sample_size=int(anchor_count)
        )
    else:
        key, k_idx = jax.random.split(key)
        anchor_task_idx = np.asarray(
            jax.random.choice(k_idx, n_total, shape=(int(anchor_count),), replace=True)
        ).astype(int)

    contexts: list[tuple[np.ndarray, np.ndarray]] = []
    query_banks: list[np.ndarray] = []
    rollout_pools: list[np.ndarray] = []
    fixed_keys: Optional[dict[tuple[int, int], Any]] = (
        {} if bool(fixed_rollout_paths) else None
    )

    for a, anchor_idx in enumerate(anchor_task_idx.tolist()):
        # Build anchor-specific split from a deterministic permutation
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


def _load_adapter_modules(path: str) -> dict[str, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported adapter checkpoint format.")
    modules = ckpt.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Adapter checkpoint must contain dict field 'modules'.")
    return modules


def _extract_base_state_from_fitted_rule(pred_rule: ClassifierPredRule) -> dict[str, torch.Tensor]:
    model = get_tabpfn_model(pred_rule)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _build_alpha_scaled_state(
    *,
    base_state: dict[str, torch.Tensor],
    adapter_modules: dict[str, dict[str, Any]],
    alpha_scale: float,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """
    Build state_dict for W = W_base + alpha_scale * DeltaW_lora
    using selected adapter modules.

    Returns:
      (scaled_state, n_applied_modules, n_skipped_modules)
    """
    out = {k: v.clone() for k, v in base_state.items()}
    n_applied = 0
    n_skipped = 0

    for name, info in adapter_modules.items():
        weight_key = f"{name}.weight"
        if weight_key not in out:
            n_skipped += 1
            continue

        if "lora_A" not in info or "lora_B" not in info:
            n_skipped += 1
            continue

        lora_A = info["lora_A"]
        lora_B = info["lora_B"]
        if not torch.is_tensor(lora_A) or not torch.is_tensor(lora_B):
            n_skipped += 1
            continue

        scaling = info.get("scaling")
        if scaling is None:
            r = float(info.get("r", max(1, lora_A.shape[1])))
            alpha = float(info.get("alpha", r))
            scaling = alpha / max(r, 1.0)
        scaling = float(scaling)

        delta = (lora_A.detach().cpu().float() @ lora_B.detach().cpu().float()).T
        delta = delta * scaling * float(alpha_scale)
        base_w = out[weight_key]
        out[weight_key] = base_w + delta.to(dtype=base_w.dtype)
        n_applied += 1

    return out, n_applied, n_skipped


def _make_shared_pred_rule_factory(
    *,
    categorical_x: list[bool],
    n_estimators: int,
    locked_state_dict: Optional[dict[str, torch.Tensor]] = None,
):
    shared = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    if locked_state_dict is not None:
        shared._locked_state_dict = {k: v.clone().cpu() for k, v in locked_state_dict.items()}

    def factory():
        return shared

    return factory, shared


def parse_int_tuple(csv_text: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in str(csv_text).split(",") if v.strip() != ""]
    if len(vals) == 0:
        raise ValueError("Expected non-empty comma-separated integer list.")
    return tuple(vals)


def parse_float_tuple(csv_text: str) -> tuple[float, ...]:
    vals = [float(v.strip()) for v in str(csv_text).split(",") if v.strip() != ""]
    if len(vals) == 0:
        raise ValueError("Expected non-empty comma-separated float list.")
    return tuple(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter .pt (encoder-only or full).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--alphas", type=str, default="0.5,1.0")
    parser.add_argument("--out-dir", type=str, default="cross_dataset_transfer")
    parser.add_argument("--save-csv", action="store_true")

    # EMD settings (strict fixed-anchor)
    parser.add_argument(
        "--enable-emd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict fixed-anchor EMD evaluation (default: enabled).",
    )
    parser.add_argument("--emd-anchor-count", type=int, default=4)
    parser.add_argument("--emd-context-size", type=int, default=100)
    parser.add_argument("--emd-queries-per-anchor", type=int, default=5)
    parser.add_argument("--emd-prefix-depths", type=str, default="0,2,4")
    parser.add_argument("--emd-k-values", type=str, default="3,7,11,15")
    parser.add_argument("--emd-continuation-depth", type=int, default=30)
    parser.add_argument("--emd-n-continuations", type=int, default=8)
    parser.add_argument("--emd-rng-fold-in", type=int, default=202)
    parser.add_argument(
        "--emd-fixed-rollout-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed rollout keys in EMD suite (default: enabled).",
    )

    # Belief drift visualization settings
    parser.add_argument(
        "--enable-drift-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fixed-query belief drift grid plots (default: enabled).",
    )
    parser.add_argument("--rollout-n-queries", type=int, default=6)
    parser.add_argument("--rollout-grid-rows", type=int, default=2)
    parser.add_argument("--rollout-grid-cols", type=int, default=3)
    parser.add_argument("--rollout-base-n", type=int, default=100)
    parser.add_argument("--rollout-depth", type=int, default=20)
    parser.add_argument("--rollout-n-paths", type=int, default=8)
    args = parser.parse_args()

    set_global_seeds(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    alphas = parse_float_tuple(args.alphas)
    emd_prefix_depths = parse_int_tuple(args.emd_prefix_depths)
    emd_k_values = parse_int_tuple(args.emd_k_values)
    adapter_path = os.path.abspath(args.adapter_path)
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    adapter_modules = _load_adapter_modules(adapter_path)
    n_encoder_modules = sum(
        1 for name in adapter_modules.keys()
        if str(name).startswith("transformer_encoder.")
    )
    n_decoder_modules = sum(
        1 for name in adapter_modules.keys()
        if str(name).startswith("decoder_dict.")
    )
    print(f"Loaded adapter: {adapter_path}")
    print(f"  total modules: {len(adapter_modules)}")
    print(f"  module split: encoder={n_encoder_modules}, decoder={n_decoder_modules}")
    print(
        "Run options: "
        f"EMD={'on' if args.enable_emd else 'off'}, "
        f"drift_plot={'on' if args.enable_drift_plot else 'off'}, "
        f"fixed_emd_rollout_keys={'on' if args.emd_fixed_rollout_paths else 'off'}"
    )

    all_rows: list[dict[str, Any]] = []

    for spec in DEFAULT_DATASETS:
        print("\n" + "=" * 80)
        print(f"DATASET OpenML ID={spec.data_id} ({spec.name_hint})")
        print("=" * 80)
        set_global_seeds(args.seed)

        X_train, X_test, y_train, y_test, categorical_x, le, dataset_name = load_openml_split(
            data_id=spec.data_id,
            test_size=args.test_size,
            seed=args.seed,
            subsample_n=spec.subsample_n,
        )
        dataset_tag = _slugify(dataset_name)
        ds_dir = os.path.join(args.out_dir, dataset_tag)
        os.makedirs(ds_dir, exist_ok=True)
        ds_emd_context_size = (
            int(spec.emd_context_size)
            if spec.emd_context_size is not None
            else int(args.emd_context_size)
        )
        ds_rollout_base_n = (
            int(spec.rollout_base_n)
            if spec.rollout_base_n is not None
            else int(args.rollout_base_n)
        )
        print(
            f"  name={dataset_name}, train={len(y_train)}, test={len(y_test)}, "
            f"classes={len(le.classes_)}, subsample={spec.subsample_n}, "
            f"emd_context_size={ds_emd_context_size}, "
            f"rollout_base_n={ds_rollout_base_n}"
        )

        # Build sampling rule and fixed EMD suite once per dataset (shared across model variants).
        pred_rule_sampling = ClassifierPredRule(categorical_x, n_estimators=args.n_estimators)
        pred_rule_sampling.fit(X_train, y_train)
        base_state = _extract_base_state_from_fitted_rule(pred_rule_sampling)

        emd_suite = None
        if args.enable_emd:
            key_emd = jax.random.fold_in(
                jax.random.PRNGKey(args.seed),
                int(args.emd_rng_fold_in),
            )
            emd_suite = _build_fixed_openml_emd_anchor_suite(
                key=key_emd,
                x_pool=np.asarray(X_test),
                y_pool=np.asarray(y_test),
                anchor_count=args.emd_anchor_count,
                context_size=ds_emd_context_size,
                queries_per_anchor=args.emd_queries_per_anchor,
                prefix_depths=emd_prefix_depths,
                fixed_rollout_paths=bool(args.emd_fixed_rollout_paths),
            )

        # Fixed query indices for belief drift plots (shared across model variants)
        rollout_query_indices = None
        if args.enable_drift_plot:
            rng = np.random.default_rng(args.seed)
            replace = len(X_test) < args.rollout_n_queries
            rollout_query_indices = np.array(
                rng.choice(len(X_test), size=args.rollout_n_queries, replace=replace),
                dtype=int,
            )
            print(f"  fixed rollout queries: {rollout_query_indices.tolist()}")

        # Evaluate base + alpha variants.
        variants: list[tuple[str, Optional[float], Optional[dict[str, torch.Tensor]]]] = [
            ("base", None, None)
        ]
        for a in alphas:
            scaled_state, n_applied, n_skipped = _build_alpha_scaled_state(
                base_state=base_state,
                adapter_modules=adapter_modules,
                alpha_scale=float(a),
            )
            print(
                f"  alpha={a:.2f} state built: "
                f"applied_modules={n_applied}, skipped={n_skipped}"
            )
            variants.append((f"alpha_{a:.2f}", a, scaled_state))

        for variant_name, alpha_val, locked_state in variants:
            pred_rule_eval = ClassifierPredRule(categorical_x, n_estimators=args.n_estimators)
            if locked_state is not None:
                pred_rule_eval._locked_state_dict = {
                    k: v.clone().cpu() for k, v in locked_state.items()
                }

            metrics = evaluate_basic(
                pred_rule_eval,
                X_train,
                y_train,
                X_test,
                y_test,
                use_torch=True,
                task_type="classification",
            )

            emd_mean = float("nan")
            emd_std = float("nan")
            emd_cov = -1
            if args.enable_emd and emd_suite is not None:
                anchor_contexts, anchor_query_banks, fixed_rollout_keys, anchor_rollout_pools, _ = emd_suite
                key_eval = jax.random.PRNGKey(args.seed + 999)
                emd_mean, emd_std, emd_cov, _ = _compute_emd_fixed_anchor_suite(
                    key=key_eval,
                    pred_rule_train=pred_rule_eval,
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

            print(
                f"  [{variant_name}] "
                f"acc={metrics.accuracy:.4f}, nll={metrics.nll:.4f}, ece={metrics.ece:.4f}, "
                f"emd={emd_mean:.6f}, emd_std={emd_std:.6f}, cov={emd_cov}"
            )

            if args.enable_drift_plot and rollout_query_indices is not None:
                tag = f"{dataset_tag}_{variant_name}"
                res = _run_fixed_queries_rollout_grid_analysis(
                    x_context_pool=X_train,
                    y_context_pool=y_train,
                    x_query_pool=X_test,
                    x_sampling_pool=X_train,
                    query_indices=rollout_query_indices.tolist(),
                    class_names=le.classes_,
                    pred_rule_factory=lambda pr=pred_rule_eval: pr,
                    base_n=ds_rollout_base_n,
                    depth=args.rollout_depth,
                    n_paths=args.rollout_n_paths,
                    seed=args.seed,
                    n_rows=args.rollout_grid_rows,
                    n_cols=args.rollout_grid_cols,
                    save_dir=ds_dir,
                    tag=tag,
                    show=False,
                )
                src = res["paths"]["grid_plot_png"]
                dst = os.path.join(ds_dir, f"belief_{tag}.png")
                if os.path.abspath(src) != os.path.abspath(dst):
                    os.replace(src, dst)

            row = {
                "dataset_id": spec.data_id,
                "dataset_name": dataset_name,
                "variant": variant_name,
                "alpha": "" if alpha_val is None else float(alpha_val),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "accuracy": float(metrics.accuracy),
                "nll": float(metrics.nll),
                "ece": float(metrics.ece),
                "emd_mean": float(emd_mean),
                "emd_std": float(emd_std),
                "emd_coverage": int(emd_cov),
            }
            all_rows.append(row)

        # Save per-dataset CSV
        if args.save_csv:
            csv_path = os.path.join(ds_dir, f"transfer_metrics_{dataset_tag}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[-len(variants):][0].keys()))
                w.writeheader()
                for r in all_rows[-len(variants):]:
                    w.writerow(r)
            print(f"  saved: {csv_path}")

    # Save aggregate CSV
    if args.save_csv and len(all_rows) > 0:
        out_csv = os.path.join(args.out_dir, "transfer_metrics_all.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nSaved aggregate CSV: {out_csv}")


if __name__ == "__main__":
    main()
