"""
run_rollout_uncertainty_scaling.py

Baseline-TabPFN rollout uncertainty scaling experiment on synthetic tasks.

For each task:
  1) sample one fixed context/query/rollout split
  2) generate B rollout continuations from the fixed context
  3) at each rollout step k, measure
     - U_JS(k): JS disagreement across rollout predictive distributions
     - IQR(k): IQR of the ground-truth-class probability across rollouts
  4) fit log-log slopes for U_JS and IQR vs k

Outputs:
  - per-task curve CSV
  - per-task slope CSV
  - summary CSV
  - setup-wise log-log curve PNGs
  - setup-wise slope histogram PNGs
"""

# Example:
# python inspect/run_rollout_uncertainty_scaling.py --setup-group single_mode --synthetic-mode scm_mix
# python inspect/run_rollout_uncertainty_scaling.py --setup-group nonlinear_link_setups --n-tasks-per-setup 20

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from predictive_rule import ClassifierPredRule
from rollout import belief_at_depth_torch_batched, build_prefix_batch_data, horizon_k_to_depth
from run_synthetic_emd_nll_ece_relation import (
    _generate_setup_task,
    _get_setup_specs,
    _sample_valid_split,
)
from run_synthetic_uncertainty_scaling import js_disagreement_from_replicates, save_csv
from generate_synthetic import MixtureConfig, make_mixture_config


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


def _fit_slope_loglog(k_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.log(np.asarray(k_values, dtype=np.float64))
    y = np.log(np.clip(np.asarray(y_values, dtype=np.float64), 1e-12, None))
    return float(np.polyfit(x, y, 1)[0])


def _evaluate_task_rollout_scaling(
    *,
    x_task: np.ndarray,
    y_task: np.ndarray,
    seed: int,
    context_size: int,
    query_count: int,
    rollout_depth: int,
    n_continuations: int,
    n_estimators: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    split_rng = np.random.default_rng(int(seed))
    split = _sample_valid_split(
        x_task=x_task,
        y_task=y_task,
        rng=split_rng,
        context_size=int(context_size),
        query_pool_size=int(query_count),
        require_all_classes_in_context=True,
    )

    idx_ctx = split["idx_context"]
    idx_query = split["idx_query"]
    idx_roll = split["idx_rollout"]

    x_ctx = np.asarray(x_task[idx_ctx], dtype=np.float32)
    y_ctx_global = np.asarray(y_task[idx_ctx]).astype(int)
    x_query = np.asarray(x_task[idx_query], dtype=np.float32)
    y_query_global = np.asarray(y_task[idx_query]).astype(int)
    x_roll_pool = np.asarray(x_task[idx_roll], dtype=np.float32)

    local_classes, y_ctx_local = np.unique(y_ctx_global, return_inverse=True)
    class_to_local = {int(c): i for i, c in enumerate(local_classes.tolist())}
    y_query_local = np.asarray([class_to_local[int(c)] for c in y_query_global], dtype=int)

    categorical_x = [False] * int(x_task.shape[1])
    pred_rule_sampling = ClassifierPredRule(categorical_x, n_estimators=int(n_estimators))
    pred_rule_belief = ClassifierPredRule(categorical_x, n_estimators=int(n_estimators))
    pred_rule_sampling.fit(x_ctx, y_ctx_local.astype(np.int64))
    pred_rule_belief.fit(x_ctx, y_ctx_local.astype(np.int64))

    key = jax.random.PRNGKey(int(seed))
    prefix_batch = build_prefix_batch_data(
        key=key,
        pred_rule_sampling=pred_rule_sampling,
        x0=x_ctx,
        y0=y_ctx_local.astype(np.int64),
        prefix_depth=0,
        continuation_depth=int(rollout_depth),
        n_continuations=int(n_continuations),
        x_sampling_pool=x_roll_pool,
        x_sample_without_replacement=True,
    )
    continuations = prefix_batch.continuations

    curve_rows: list[dict[str, float]] = []
    k_values = np.arange(1, int(rollout_depth) + 1, dtype=int)
    u_curve = []
    iqr_curve = []
    for k in k_values:
        depth = horizon_k_to_depth(int(k))
        probs_bqc = (
            belief_at_depth_torch_batched(
                pred_rule_belief,
                continuations,
                depth,
                x_query,
            )
            .detach()
            .cpu()
            .numpy()
        )
        u_q = js_disagreement_from_replicates(probs_bqc)
        true_probs_bq = np.stack(
            [probs_bqc[:, q_idx, int(y_query_local[q_idx])] for q_idx in range(len(y_query_local))],
            axis=1,
        )
        iqr_q = np.percentile(true_probs_bq, 75.0, axis=0) - np.percentile(true_probs_bq, 25.0, axis=0)

        u_stat = float(np.median(u_q))
        iqr_stat = float(np.median(iqr_q))
        u_curve.append(u_stat)
        iqr_curve.append(iqr_stat)
        curve_rows.append(
            {
                "k": int(k),
                "u_js": u_stat,
                "iqr": iqr_stat,
            }
        )

    slope_row = {
        "u_js_slope": _fit_slope_loglog(k_values, np.asarray(u_curve, dtype=np.float64)),
        "iqr_slope": _fit_slope_loglog(k_values, np.asarray(iqr_curve, dtype=np.float64)),
    }
    return curve_rows, slope_row


def _plot_setup_curves(
    *,
    curve_rows: list[dict[str, Any]],
    out_path: Path,
    setup_name: str,
) -> None:
    rows = [r for r in curve_rows if r["setup_name"] == setup_name]
    ks = sorted({int(r["k"]) for r in rows})

    u_by_k = defaultdict(list)
    iqr_by_k = defaultdict(list)
    for row in rows:
        u_by_k[int(row["k"])].append(float(row["u_js"]))
        iqr_by_k[int(row["k"])].append(float(row["iqr"]))

    k_arr = np.asarray(ks, dtype=np.float64)
    u_mean = np.asarray([np.mean(u_by_k[k]) for k in ks], dtype=np.float64)
    u_std = np.asarray([np.std(u_by_k[k]) for k in ks], dtype=np.float64)
    i_mean = np.asarray([np.mean(iqr_by_k[k]) for k in ks], dtype=np.float64)
    i_std = np.asarray([np.std(iqr_by_k[k]) for k in ks], dtype=np.float64)

    def _ref_line(y_mean: np.ndarray) -> np.ndarray:
        if len(y_mean) == 0:
            return y_mean
        return y_mean[0] * (k_arr / k_arr[0]) ** (-1.0)

    u_ref = _ref_line(u_mean)
    i_ref = _ref_line(i_mean)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    series = [
        ("U_JS", u_mean, u_std, u_ref, "#1f77b4"),
        ("IQR", i_mean, i_std, i_ref, "#d62728"),
    ]
    for ax, (title, mean_vals, std_vals, ref_vals, color) in zip(axes, series):
        ax.plot(k_arr, mean_vals, color=color, linewidth=2, label=f"{title} mean")
        ax.fill_between(
            k_arr,
            np.clip(mean_vals - std_vals, 1e-12, None),
            mean_vals + std_vals,
            color=color,
            alpha=0.2,
            label="±1 std",
        )
        ax.plot(k_arr, ref_vals, linestyle="--", color="black", linewidth=1.5, label=r"reference $1/k$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{setup_name}: log {title}(k) vs log k")
        ax.set_xlabel("k")
        ax.set_ylabel(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_setup_histograms(
    *,
    slope_rows: list[dict[str, Any]],
    out_path: Path,
    setup_name: str,
) -> None:
    rows = [r for r in slope_rows if r["setup_name"] == setup_name]
    u_slopes = np.asarray([float(r["u_js_slope"]) for r in rows], dtype=np.float64)
    i_slopes = np.asarray([float(r["iqr_slope"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    series = [
        ("U_JS slope", u_slopes, "#1f77b4"),
        ("IQR slope", i_slopes, "#d62728"),
    ]
    for ax, (title, vals, color) in zip(axes, series):
        ax.hist(vals, bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(-1.0, color="black", linestyle="--", linewidth=1.5, label="ideal -1")
        ax.set_title(f"{setup_name}: {title}")
        ax.set_xlabel("slope")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic rollout uncertainty scaling with baseline TabPFN."
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
    parser.add_argument("--context-size", type=int, default=100)
    parser.add_argument("--query-count", type=int, default=24)
    parser.add_argument("--n-continuations", type=int, default=24)
    parser.add_argument("--rollout-depth", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="rollout_uncertainty_scaling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_specs = _get_setup_specs_local(
        setup_group=str(args.setup_group),
        synthetic_mode=str(args.synthetic_mode),
        priors_raw=str(args.priors),
    )
    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    curve_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []

    for setup_idx, (setup_name, cfg) in enumerate(setup_specs):
        print("\n" + "=" * 80)
        print(f"SETUP {setup_idx}: {setup_name} (mode={cfg.mode_name})")
        print("=" * 80)
        for task_idx in range(int(args.n_tasks_per_setup)):
            task_seed = int(args.task_seed_base) + setup_idx * 100_000 + task_idx
            x_task, y_task, meta = _generate_setup_task(
                setup_name=setup_name,
                cfg=cfg,
                seed=task_seed,
            )
            prior_type = str(meta.get("prior_type", "unknown"))

            task_curve_rows, task_slope_row = _evaluate_task_rollout_scaling(
                x_task=x_task,
                y_task=y_task,
                seed=task_seed + 10_000,
                context_size=int(args.context_size),
                query_count=int(args.query_count),
                rollout_depth=int(args.rollout_depth),
                n_continuations=int(args.n_continuations),
                n_estimators=int(args.n_estimators),
            )
            for row in task_curve_rows:
                curve_rows.append(
                    {
                        "setup_name": setup_name,
                        "prior_type": prior_type,
                        "task_index": task_idx,
                        "task_seed": task_seed,
                        "synthetic_mode": cfg.mode_name,
                        **row,
                    }
                )
            slope_rows.append(
                {
                    "setup_name": setup_name,
                    "prior_type": prior_type,
                    "task_index": task_idx,
                    "task_seed": task_seed,
                    "synthetic_mode": cfg.mode_name,
                    **task_slope_row,
                }
            )

        setup_slopes = [r for r in slope_rows if r["setup_name"] == setup_name]
        print(
            f"  mean slopes: U_JS={np.mean([r['u_js_slope'] for r in setup_slopes]):.4f}, "
            f"IQR={np.mean([r['iqr_slope'] for r in setup_slopes]):.4f}"
        )

    summary_rows: list[dict[str, Any]] = []
    for setup_name, _cfg in setup_specs:
        setup_slopes = [r for r in slope_rows if r["setup_name"] == setup_name]
        summary_rows.extend(
            [
                {
                    "setup_name": setup_name,
                    "metric": "u_js",
                    "mean_slope": float(np.mean([r["u_js_slope"] for r in setup_slopes])),
                    "std_slope": float(np.std([r["u_js_slope"] for r in setup_slopes])),
                    "n_tasks": len(setup_slopes),
                },
                {
                    "setup_name": setup_name,
                    "metric": "iqr",
                    "mean_slope": float(np.mean([r["iqr_slope"] for r in setup_slopes])),
                    "std_slope": float(np.std([r["iqr_slope"] for r in setup_slopes])),
                    "n_tasks": len(setup_slopes),
                },
            ]
        )

    curves_csv = out_dir / f"rollout_uncertainty_curves_{tag}.csv"
    slopes_csv = out_dir / f"rollout_uncertainty_slopes_{tag}.csv"
    summary_csv = out_dir / f"rollout_uncertainty_summary_{tag}.csv"
    save_csv(str(curves_csv), curve_rows, list(curve_rows[0].keys()))
    save_csv(str(slopes_csv), slope_rows, list(slope_rows[0].keys()))
    save_csv(str(summary_csv), summary_rows, list(summary_rows[0].keys()))

    for setup_name, _cfg in setup_specs:
        curve_png = out_dir / f"rollout_uncertainty_curve_{setup_name}_{tag}.png"
        hist_png = out_dir / f"rollout_uncertainty_hist_{setup_name}_{tag}.png"
        _plot_setup_curves(curve_rows=curve_rows, out_path=curve_png, setup_name=setup_name)
        _plot_setup_histograms(slope_rows=slope_rows, out_path=hist_png, setup_name=setup_name)

    print("\nSaved:")
    print(f"  curves:  {curves_csv}")
    print(f"  slopes:  {slopes_csv}")
    print(f"  summary: {summary_csv}")
    print(f"  setup plots dir: {out_dir}")


if __name__ == "__main__":
    main()
