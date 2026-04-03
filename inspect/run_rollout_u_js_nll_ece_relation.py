"""
run_rollout_u_js_nll_ece_relation.py

Task-level relation table between rollout-based synthetic U_JS(k) and predictive
quality on a fixed initial context/query split.

For each synthetic task:
  1) sample one fixed context/query/rollout split
  2) build B rollout continuations from the fixed initial context
  3) for each rollout step k:
       - compute query-wise U_JS across continuations
       - store median over queries as U_JS(t, k)
  4) evaluate baseline TabPFN on the same initial context and fixed query bank
       - store NLL(t), ECE(t), Accuracy(t)

Outputs:
  - one CSV row per task

# Example:
# python inspect/run_rollout_u_js_nll_ece_relation.py --setup-group single_mode --synthetic-mode scm_mix
# python inspect/run_rollout_u_js_nll_ece_relation.py --setup-group nonlinear_link_setups
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import torch

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from eval import compute_basic_metrics
from generate_synthetic import MixtureConfig, make_mixture_config
from predictive_rule import ClassifierPredRule
from rollout import belief_at_depth_torch_batched, build_prefix_batch_data, horizon_k_to_depth
from run_classification import set_global_seeds
from run_synthetic_emd_nll_ece_relation import (
    _generate_setup_task,
    _get_setup_specs,
    _sample_valid_split,
)
from run_synthetic_uncertainty_scaling import js_disagreement_from_replicates, save_csv


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    vals = [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if len(vals) == 0:
        raise ValueError(f"Expected comma-separated integers, got {text!r}")
    return tuple(vals)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_setup_specs_local(
    *,
    setup_group: str,
    synthetic_mode: str,
    priors_raw: str,
) -> list[tuple[str, MixtureConfig]]:
    if str(setup_group).strip().lower() == "single_mode":
        return [(str(synthetic_mode), make_mixture_config(str(synthetic_mode)))]
    return _get_setup_specs(
        setup_group=str(setup_group),
        synthetic_mode=str(synthetic_mode),
        priors_raw=str(priors_raw),
    )


def _normalize_k_grid(k_grid_raw: str, rollout_depth: int) -> tuple[int, ...]:
    if str(k_grid_raw).strip() == "":
        return tuple(range(1, int(rollout_depth) + 1))
    vals = _parse_int_tuple(k_grid_raw)
    vals = tuple(sorted({int(v) for v in vals if 1 <= int(v) <= int(rollout_depth)}))
    if len(vals) == 0:
        raise ValueError(f"ujs-k-grid has no valid values within 1..{int(rollout_depth)}")
    return vals


def _validate_sizes(
    *,
    task_size: int,
    context_size: int,
    query_count: int,
) -> None:
    if int(task_size) <= int(context_size) + int(query_count):
        raise ValueError(
            f"Task size too small: task_size={task_size}, context_size={context_size}, query_count={query_count}"
        )


def _evaluate_initial_context_metrics(
    *,
    categorical_x: list[bool],
    n_estimators: int,
    x_context: np.ndarray,
    y_context_global: np.ndarray,
    x_query: np.ndarray,
    y_query_global: np.ndarray,
    global_num_classes: int,
) -> dict[str, float]:
    local_classes, y_ctx_local = np.unique(np.asarray(y_context_global).astype(int), return_inverse=True)
    pred_rule = ClassifierPredRule(categorical_x, n_estimators=int(n_estimators))
    pred_rule.fit(np.asarray(x_context, dtype=np.float32), y_ctx_local.astype(np.int64))
    with torch.no_grad():
        probs_local = pred_rule.get_belief_torch(
            np.asarray(x_query, dtype=np.float32),
            np.asarray(x_context, dtype=np.float32),
            y_ctx_local.astype(np.int64),
        ).cpu().numpy()
    probs_global = np.zeros((len(x_query), int(global_num_classes)), dtype=np.float64)
    probs_global[:, local_classes] = probs_local[:, : len(local_classes)]
    probs_global = probs_global / np.clip(probs_global.sum(axis=1, keepdims=True), 1e-12, None)
    m = compute_basic_metrics(probs=probs_global, y_true=np.asarray(y_query_global).astype(np.int64))
    return {
        "accuracy": float(m.accuracy),
        "nll": float(m.nll),
        "ece": float(m.ece),
    }


def _compute_rollout_ujs_curve(
    *,
    categorical_x: list[bool],
    n_estimators: int,
    x_context: np.ndarray,
    y_context_global: np.ndarray,
    x_query: np.ndarray,
    y_query_global: np.ndarray,
    x_rollout_pool: np.ndarray,
    k_grid: tuple[int, ...],
    rollout_depth: int,
    n_continuations: int,
    seed: int,
) -> dict[str, float]:
    local_classes, y_ctx_local = np.unique(np.asarray(y_context_global).astype(int), return_inverse=True)
    class_to_local = {int(c): i for i, c in enumerate(local_classes.tolist())}
    y_query_local = np.asarray([class_to_local[int(c)] for c in np.asarray(y_query_global).astype(int)], dtype=np.int64)

    pred_rule_sampling = ClassifierPredRule(categorical_x, n_estimators=int(n_estimators))
    pred_rule_belief = ClassifierPredRule(categorical_x, n_estimators=int(n_estimators))
    pred_rule_sampling.fit(np.asarray(x_context, dtype=np.float32), y_ctx_local.astype(np.int64))
    pred_rule_belief.fit(np.asarray(x_context, dtype=np.float32), y_ctx_local.astype(np.int64))

    key = jax.random.PRNGKey(int(seed))
    prefix_batch = build_prefix_batch_data(
        key=key,
        pred_rule_sampling=pred_rule_sampling,
        x0=np.asarray(x_context, dtype=np.float32),
        y0=y_ctx_local.astype(np.int64),
        prefix_depth=0,
        continuation_depth=int(rollout_depth),
        n_continuations=int(n_continuations),
        x_sampling_pool=np.asarray(x_rollout_pool, dtype=np.float32),
        x_sample_without_replacement=True,
    )
    continuations = prefix_batch.continuations

    row: dict[str, float] = {}
    for k in k_grid:
        depth = horizon_k_to_depth(int(k))
        probs_bqc = (
            belief_at_depth_torch_batched(
                pred_rule_belief,
                continuations,
                depth,
                np.asarray(x_query, dtype=np.float32),
            )
            .detach()
            .cpu()
            .numpy()
        )
        u_q = js_disagreement_from_replicates(probs_bqc)
        row[f"u_js_k{int(k)}"] = float(np.median(np.asarray(u_q, dtype=np.float64)))
        true_probs_bq = np.stack(
            [probs_bqc[:, q_idx, int(y_query_local[q_idx])] for q_idx in range(len(y_query_local))],
            axis=1,
        )
        iqr_q = np.percentile(true_probs_bq, 75.0, axis=0) - np.percentile(true_probs_bq, 25.0, axis=0)
        row[f"iqr_k{int(k)}"] = float(np.median(np.asarray(iqr_q, dtype=np.float64)))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute task-level rollout U_JS(k) and NLL/ECE table on synthetic tasks."
    )
    parser.add_argument(
        "--setup-group",
        type=str,
        default="single_mode",
        choices=(
            "single_mode",
            "standard_priors",
            "simple_linear",
            "simple_linear_ablations",
            "scm_variants",
            "nonlinear_link_setups",
        ),
    )
    parser.add_argument("--synthetic-mode", type=str, default="scm_mix")
    parser.add_argument("--priors", type=str, default="gbdt,scm,smooth_mlp,sparse_linear")
    parser.add_argument("--task-seed-base", type=int, default=42)
    parser.add_argument("--n-tasks-per-setup", type=int, default=200)
    parser.add_argument("--task-size", type=int, default=2000)
    parser.add_argument("--context-size", type=int, default=100)
    parser.add_argument("--query-count", type=int, default=24)
    parser.add_argument("--rollout-depth", type=int, default=30)
    parser.add_argument("--ujs-k-grid", type=str, default="")
    parser.add_argument("--n-continuations", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="synthetic_rollout_ujs_relation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_grid = _normalize_k_grid(args.ujs_k_grid, int(args.rollout_depth))
    _validate_sizes(
        task_size=int(args.task_size),
        context_size=int(args.context_size),
        query_count=int(args.query_count),
    )

    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seeds(int(args.task_seed_base))

    setup_specs = _get_setup_specs_local(
        setup_group=str(args.setup_group),
        synthetic_mode=str(args.synthetic_mode),
        priors_raw=str(args.priors),
    )

    print("\n── Synthetic Rollout U_JS(k) vs NLL/ECE Task Table ──")
    print(f"  setup_group={args.setup_group}")
    print(f"  setups={[name for name, _ in setup_specs]}")
    print(f"  tasks/setup={int(args.n_tasks_per_setup)}")
    print(
        f"  task_size={int(args.task_size)}, context_size={int(args.context_size)}, "
        f"query_count={int(args.query_count)}"
    )
    print(
        f"  rollout_depth={int(args.rollout_depth)}, k_grid={k_grid}, "
        f"n_continuations={int(args.n_continuations)}"
    )

    rows: list[dict[str, Any]] = []

    for setup_idx, (setup_name, cfg_in) in enumerate(setup_specs):
        cfg = copy.deepcopy(cfg_in)
        cfg.n_samples_min = int(args.task_size)
        cfg.n_samples_max = int(args.task_size)

        for task_offset in range(int(args.n_tasks_per_setup)):
            task_seed = int(args.task_seed_base) + setup_idx * 1_000_003 + task_offset
            x_task, y_task, meta = _generate_setup_task(
                setup_name=setup_name,
                cfg=cfg,
                seed=task_seed,
            )
            n = int(len(y_task))
            d = int(x_task.shape[1])
            n_classes = int(np.max(y_task)) + 1
            categorical_x = [False] * d

            split = _sample_valid_split(
                x_task=np.asarray(x_task, dtype=np.float32),
                y_task=np.asarray(y_task, dtype=np.int64),
                rng=np.random.default_rng(int(task_seed) + 17),
                context_size=int(args.context_size),
                query_pool_size=int(args.query_count),
                require_all_classes_in_context=True,
            )
            x_context = np.asarray(x_task[split["idx_context"]], dtype=np.float32)
            y_context = np.asarray(y_task[split["idx_context"]], dtype=np.int64)
            x_query = np.asarray(x_task[split["idx_query"]], dtype=np.float32)
            y_query = np.asarray(y_task[split["idx_query"]], dtype=np.int64)
            x_rollout_pool = np.asarray(x_task[split["idx_rollout"]], dtype=np.float32)

            print(
                f"[setup={setup_name}][task={task_offset+1:03d}/{int(args.n_tasks_per_setup)}] "
                f"seed={task_seed}, n={n}, d={d}, classes={n_classes}"
            )

            row: dict[str, Any] = {
                "setup_name": setup_name,
                "prior_type": str(meta.get("prior_type", "unknown")),
                "task_seed": int(task_seed),
                "synthetic_mode": str(cfg.mode_name),
                "n_samples": n,
                "n_features": d,
                "n_classes": n_classes,
                "context_size": int(args.context_size),
                "query_count": int(len(split["idx_query"])),
                "rollout_pool_size": int(len(split["idx_rollout"])),
                "rollout_depth": int(args.rollout_depth),
                "n_continuations": int(args.n_continuations),
            }

            row.update(
                _compute_rollout_ujs_curve(
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_context,
                    y_context_global=y_context,
                    x_query=x_query,
                    y_query_global=y_query,
                    x_rollout_pool=x_rollout_pool,
                    k_grid=k_grid,
                    rollout_depth=int(args.rollout_depth),
                    n_continuations=int(args.n_continuations),
                    seed=task_seed + 101_000,
                )
            )
            row.update(
                _evaluate_initial_context_metrics(
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_context,
                    y_context_global=y_context,
                    x_query=x_query,
                    y_query_global=y_query,
                    global_num_classes=n_classes,
                )
            )
            rows.append(row)

    tag = _now_tag()
    setup_tag = str(args.setup_group).strip().lower()
    csv_path = out_dir / f"synthetic_rollout_ujs_nll_ece_relation_{setup_tag}_{tag}.csv"

    fieldnames = [
        "setup_name",
        "prior_type",
        "task_seed",
        "synthetic_mode",
        "n_samples",
        "n_features",
        "n_classes",
        "context_size",
        "query_count",
        "rollout_pool_size",
        "rollout_depth",
        "n_continuations",
        *[f"u_js_k{int(k)}" for k in k_grid],
        *[f"iqr_k{int(k)}" for k in k_grid],
        "accuracy",
        "nll",
        "ece",
    ]
    save_csv(str(csv_path), rows, fieldnames)
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
