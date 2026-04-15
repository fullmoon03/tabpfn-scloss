"""
run_synthetic_u_js_nll_ece_relation.py

Task-level relation table between synthetic U_JS scaling and predictive quality.

For each synthetic task:
  1) sample a fixed query bank Q_t
  2) define the remaining points as train reservoir R_t
  3) for each N_JS in a grid:
       - sample replicate subsets from R_t
       - compute query-wise U_JS on Q_t
       - store median over queries as U_JS(t, N_JS)
  4) sample context subsets of size 100 from R_t
       - compute NLL / ECE on the same Q_t
       - store mean + median over subset replicates

Outputs:
  - one wide CSV row per (task, model)

# Example:
# python inspect/run_synthetic_u_js_nll_ece_relation.py --setup-group simple_linear
# python inspect/run_synthetic_u_js_nll_ece_relation.py --setup-group simple_linear_ablations
# python inspect/run_synthetic_u_js_nll_ece_relation.py --setup-group nonlinear_link_setups
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from eval import compute_basic_metrics
from run_classification import set_global_seeds
from run_synthetic_emd_nll_ece_relation import _generate_setup_task, _get_setup_specs
from run_synthetic_uncertainty_scaling import (
    ModelSpec,
    collect_probs_for_model_synthetic,
    js_disagreement_from_replicates,
    load_locked_state,
    sample_replicate_subsets,
    save_csv,
)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    vals = [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if len(vals) == 0:
        raise ValueError(f"Expected comma-separated integers, got {text!r}")
    return tuple(vals)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute task-level synthetic U_JS scaling and NLL/ECE summary table."
    )
    parser.add_argument(
        "--setup-group",
        type=str,
        default="simple_linear",
        choices=("standard_priors", "simple_linear", "simple_linear_ablations", "nonlinear_link_setups"),
    )
    parser.add_argument("--synthetic-mode", type=str, default="mixed_full")
    parser.add_argument("--priors", type=str, default="gbdt,scm,smooth_mlp,sparse_linear")
    parser.add_argument("--task-seed-base", type=int, default=42)
    parser.add_argument("--n-tasks-per-setup", type=int, default=200)
    parser.add_argument("--task-size", type=int, default=2000)
    parser.add_argument("--query-count", type=int, default=24)
    parser.add_argument("--ujs-n-grid", type=str, default="16,32,64,128,256")
    parser.add_argument("--ujs-replicates", type=int, default=50)
    parser.add_argument("--metric-context-size", type=int, default=100)
    parser.add_argument("--metric-replicates", type=int, default=30)
    parser.add_argument("--ece-bins", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument(
        "--tuned-merged-state",
        type=str,
        default="",
        help="Optional merged model .pt path. If provided, rows are saved for baseline and tuned.",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--save-dir", type=str, default="synthetic_ujs_relation")
    return parser.parse_args()


def _build_model_specs(tuned_path: str) -> list[ModelSpec]:
    specs = [ModelSpec(name="baseline", locked_state=None)]
    if str(tuned_path).strip():
        specs.append(ModelSpec(name="tuned", locked_state=load_locked_state(str(tuned_path).strip())))
    return specs


def _validate_sizes(*, task_size: int, query_count: int, ujs_n_grid: tuple[int, ...], metric_context_size: int) -> None:
    max_needed = max(max(ujs_n_grid), int(metric_context_size))
    if int(task_size) - int(query_count) < max_needed:
        raise ValueError(
            f"Task size too small: task_size={task_size}, query_count={query_count}, "
            f"need at least max_context={max_needed} points in reservoir."
        )


def _sample_query_and_reservoir_indices(*, n: int, query_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    query_idx = np.asarray(rng.choice(n, size=int(query_count), replace=False), dtype=np.int64)
    mask = np.ones(int(n), dtype=bool)
    mask[query_idx] = False
    reservoir_idx = np.arange(int(n), dtype=np.int64)[mask]
    return query_idx, reservoir_idx


def _summarize_metric_replicates(
    *,
    probs_rqc: np.ndarray,
    y_query: np.ndarray,
    ece_bins: int,
) -> dict[str, float]:
    nlls: list[float] = []
    eces: list[float] = []
    accs: list[float] = []
    for probs_qc in np.asarray(probs_rqc, dtype=np.float64):
        metrics = compute_basic_metrics(probs=probs_qc, y_true=np.asarray(y_query, dtype=np.int64), n_bins=int(ece_bins))
        nlls.append(float(metrics.nll))
        eces.append(float(metrics.ece))
        accs.append(float(metrics.accuracy))
    nll_arr = np.asarray(nlls, dtype=np.float64)
    ece_arr = np.asarray(eces, dtype=np.float64)
    acc_arr = np.asarray(accs, dtype=np.float64)
    return {
        "accuracy_mean": float(np.mean(acc_arr)),
        "accuracy_median": float(np.median(acc_arr)),
        "nll_mean": float(np.mean(nll_arr)),
        "nll_median": float(np.median(nll_arr)),
        "ece_mean": float(np.mean(ece_arr)),
        "ece_median": float(np.median(ece_arr)),
    }


def main() -> None:
    args = parse_args()
    ujs_n_grid = _parse_int_tuple(args.ujs_n_grid)
    _validate_sizes(
        task_size=int(args.task_size),
        query_count=int(args.query_count),
        ujs_n_grid=ujs_n_grid,
        metric_context_size=int(args.metric_context_size),
    )

    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seeds(int(args.task_seed_base))

    setup_specs = _get_setup_specs(
        setup_group=str(args.setup_group),
        synthetic_mode=str(args.synthetic_mode),
        priors_raw=str(args.priors),
    )
    model_specs = _build_model_specs(args.tuned_merged_state)

    print("\n── Synthetic U_JS vs NLL/ECE Task Table ──")
    print(f"  setup_group={args.setup_group}")
    print(f"  setups={[name for name, _ in setup_specs]}")
    print(f"  tasks/setup={int(args.n_tasks_per_setup)}")
    print(f"  task_size={int(args.task_size)}, query_count={int(args.query_count)}")
    print(f"  ujs_n_grid={ujs_n_grid}, ujs_replicates={int(args.ujs_replicates)}")
    print(
        f"  metric_context_size={int(args.metric_context_size)}, "
        f"metric_replicates={int(args.metric_replicates)}, ece_bins={int(args.ece_bins)}"
    )
    print(f"  models={[m.name for m in model_specs]}")

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
            query_idx, reservoir_idx = _sample_query_and_reservoir_indices(
                n=n,
                query_count=int(args.query_count),
                seed=task_seed + 17,
            )
            x_query = np.asarray(x_task[query_idx], dtype=np.float32)
            y_query = np.asarray(y_task[query_idx], dtype=np.int64)
            x_res = np.asarray(x_task[reservoir_idx], dtype=np.float32)
            y_res = np.asarray(y_task[reservoir_idx], dtype=np.int64)

            print(
                f"[setup={setup_name}][task={task_offset+1:03d}/{int(args.n_tasks_per_setup)}] "
                f"seed={task_seed}, n={n}, d={d}, classes={n_classes}"
            )

            for model_spec in model_specs:
                row: dict[str, Any] = {
                    "setup_name": setup_name,
                    "prior_type": str(meta.get("prior_type", "unknown")),
                    "task_seed": int(task_seed),
                    "synthetic_mode": str(cfg.mode_name),
                    "model": model_spec.name,
                    "n_samples": n,
                    "n_features": d,
                    "n_classes": n_classes,
                    "reservoir_size": int(len(reservoir_idx)),
                    "query_count": int(len(query_idx)),
                    "ujs_replicates": int(args.ujs_replicates),
                    "metric_context_size": int(args.metric_context_size),
                    "metric_replicates": int(args.metric_replicates),
                }

                for n_js in ujs_n_grid:
                    subsets_rn = sample_replicate_subsets(
                        n_train=len(x_res),
                        n0=int(n_js),
                        replicates=int(args.ujs_replicates),
                        seed=task_seed + 101_000,
                    )
                    probs_rqc = collect_probs_for_model_synthetic(
                        model_spec=model_spec,
                        categorical_x=categorical_x,
                        n_estimators=int(args.n_estimators),
                        x_train=x_res,
                        y_train=y_res,
                        x_queries=x_query,
                        subsets_rn=subsets_rn,
                        n_classes_global=n_classes,
                    )
                    u_q = js_disagreement_from_replicates(probs_rqc, eps=float(args.eps))
                    row[f"u_js_n{int(n_js)}"] = float(np.median(np.asarray(u_q, dtype=np.float64)))

                metric_subsets = sample_replicate_subsets(
                    n_train=len(x_res),
                    n0=int(args.metric_context_size),
                    replicates=int(args.metric_replicates),
                    seed=task_seed + 202_000,
                )
                metric_probs_rqc = collect_probs_for_model_synthetic(
                    model_spec=model_spec,
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_train=x_res,
                    y_train=y_res,
                    x_queries=x_query,
                    subsets_rn=metric_subsets,
                    n_classes_global=n_classes,
                )
                row.update(
                    _summarize_metric_replicates(
                        probs_rqc=metric_probs_rqc,
                        y_query=y_query,
                        ece_bins=int(args.ece_bins),
                    )
                )
                rows.append(row)

    tag = _now_tag()
    setup_tag = str(args.setup_group).strip().lower()
    csv_path = out_dir / f"synthetic_ujs_nll_ece_relation_{setup_tag}_{tag}.csv"

    fieldnames = [
        "setup_name",
        "prior_type",
        "task_seed",
        "synthetic_mode",
        "model",
        "n_samples",
        "n_features",
        "n_classes",
        "reservoir_size",
        "query_count",
        "ujs_replicates",
        "metric_context_size",
        "metric_replicates",
        *[f"u_js_n{int(n_js)}" for n_js in ujs_n_grid],
        "accuracy_mean",
        "accuracy_median",
        "nll_mean",
        "nll_median",
        "ece_mean",
        "ece_median",
    ]
    save_csv(str(csv_path), rows, fieldnames)
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
