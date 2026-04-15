"""
run_synthetic_nll_ece.py

Synthetic task-level NLL/ECE scatter experiment.

For each generated synthetic task:
  - sample one context/query split
  - evaluate baseline model on the query pool
  - optionally evaluate an externally provided tuned merged model
  - save task-level metrics and plot NLL vs ECE scatter
"""

# Example:
# python inspect/run_synthetic_nll_ece.py --setup-group single_mode --synthetic-mode scm_mix --no-include-tuned
# python inspect/run_synthetic_nll_ece.py --setup-group single_mode --synthetic-mode scm_mix --tuned-merged-state /path/to/merged_model_state.pt
# python inspect/run_synthetic_nll_ece.py --setup-group nonlinear_link_setups --tuned-merged-state /path/to/merged_model_state.pt

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from generate_synthetic import MixtureConfig, make_mixture_config
from run_synthetic_emd_nll_ece_relation import (
    _evaluate_anchor_metrics,
    _generate_setup_task,
    _get_setup_specs,
    _sample_valid_split,
)
from run_synthetic_uncertainty_scaling import load_locked_state


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _plot_scatter(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    setup_names: tuple[str, ...],
) -> None:
    n_setups = len(setup_names)
    ncols = min(3, max(1, n_setups))
    nrows = int(math.ceil(n_setups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), squeeze=False)

    color_map = {"baseline": "#1f77b4", "tuned": "#d62728"}
    marker_map = {"baseline": "o", "tuned": "s"}

    for idx, setup_name in enumerate(setup_names):
        ax = axes[idx // ncols][idx % ncols]
        setup_rows = [r for r in rows if r["setup_name"] == setup_name]
        for model_name in ("baseline", "tuned"):
            model_rows = [r for r in setup_rows if r["model_name"] == model_name]
            if not model_rows:
                continue
            ax.scatter(
                [float(r["nll"]) for r in model_rows],
                [float(r["ece"]) for r in model_rows],
                s=32,
                alpha=0.82,
                c=color_map[model_name],
                marker=marker_map[model_name],
                edgecolors="black",
                linewidths=0.35,
                label=model_name,
            )
        ax.set_title(setup_name)
        ax.set_xlabel("NLL")
        ax.set_ylabel("ECE")
        ax.grid(True, alpha=0.3)

    for idx in range(n_setups, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        fig.legend(
            list(seen.values()),
            list(seen.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(4, len(seen)),
            frameon=False,
        )

    fig.suptitle("Synthetic NLL vs ECE relation\n(task-level baseline/tuned overlay)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic task-level NLL/ECE scatter experiment."
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
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--split-seed-offset", type=int, default=1_000)
    parser.add_argument("--context-size", type=int, default=100)
    parser.add_argument("--query-pool-size", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-tuned",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tuned-merged-state",
        type=str,
        default="",
        help="Merged model state_dict path to use as the tuned model.",
    )
    parser.add_argument("--save-dir", type=str, default="synthetic_nll_ece")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuned_path = str(args.tuned_merged_state).strip()
    if args.include_tuned and tuned_path == "":
        raise ValueError("--include-tuned requires --tuned-merged-state.")

    setup_specs = _get_setup_specs_local(
        setup_group=str(args.setup_group),
        synthetic_mode=str(args.synthetic_mode),
        priors_raw=str(args.priors),
    )
    setup_names = tuple(name for name, _ in setup_specs)

    tuned_state: dict[str, torch.Tensor] | None = None
    if args.include_tuned:
        tuned_state = load_locked_state(tuned_path)

    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    rows: list[dict[str, Any]] = []

    for setup_idx, (setup_name, cfg) in enumerate(setup_specs):
        print("\n" + "=" * 80)
        print(f"SETUP {setup_idx}: {setup_name} (mode={cfg.mode_name})")
        print("=" * 80)
        for task_i in range(int(args.n_tasks)):
            task_seed = int(args.task_seed_base) + setup_idx * 100_000 + task_i
            x_task, y_task, meta = _generate_setup_task(
                setup_name=setup_name,
                cfg=cfg,
                seed=task_seed,
            )
            prior_type = str(meta.get("prior_type", "unknown"))
            d = int(x_task.shape[1])
            categorical_x = [False] * d
            global_num_classes = int(np.max(y_task)) + 1

            split_rng = np.random.default_rng(task_seed + int(args.split_seed_offset))
            split = _sample_valid_split(
                x_task=x_task,
                y_task=y_task,
                rng=split_rng,
                context_size=int(args.context_size),
                query_pool_size=int(args.query_pool_size),
                require_all_classes_in_context=False,
            )

            idx_ctx = split["idx_context"]
            idx_query = split["idx_query"]
            x_ctx = np.asarray(x_task[idx_ctx], dtype=np.float32)
            y_ctx = np.asarray(y_task[idx_ctx]).astype(int)
            x_query = np.asarray(x_task[idx_query], dtype=np.float32)
            y_query = np.asarray(y_task[idx_query]).astype(int)

            if args.include_baseline:
                baseline_metrics = _evaluate_anchor_metrics(
                    model_state_dict=None,
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_ctx,
                    y_context_global=y_ctx,
                    x_query=x_query,
                    y_query_global=y_query,
                    global_num_classes=global_num_classes,
                )
                rows.append(
                    {
                        "setup_name": setup_name,
                        "prior_type": prior_type,
                        "task_index": task_i,
                        "task_seed": task_seed,
                        "synthetic_mode": cfg.mode_name,
                        "model_name": "baseline",
                        "accuracy": baseline_metrics["accuracy"],
                        "nll": baseline_metrics["nll"],
                        "ece": baseline_metrics["ece"],
                    }
                )

            if args.include_tuned and tuned_state is not None:
                tuned_metrics = _evaluate_anchor_metrics(
                    model_state_dict=tuned_state,
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_ctx,
                    y_context_global=y_ctx,
                    x_query=x_query,
                    y_query_global=y_query,
                    global_num_classes=global_num_classes,
                )
                rows.append(
                    {
                        "setup_name": setup_name,
                        "prior_type": prior_type,
                        "task_index": task_i,
                        "task_seed": task_seed,
                        "synthetic_mode": cfg.mode_name,
                        "model_name": "tuned",
                        "accuracy": tuned_metrics["accuracy"],
                        "nll": tuned_metrics["nll"],
                        "ece": tuned_metrics["ece"],
                    }
                )

        setup_rows = [r for r in rows if r["setup_name"] == setup_name]
        for model_name in ("baseline", "tuned"):
            model_rows = [r for r in setup_rows if r["model_name"] == model_name]
            if model_rows:
                print(
                    f"  [{model_name}] mean NLL={np.mean([r['nll'] for r in model_rows]):.4f}, "
                    f"mean ECE={np.mean([r['ece'] for r in model_rows]):.4f}"
                )

    details_csv = out_dir / f"synthetic_nll_ece_details_{tag}.csv"
    plot_png = out_dir / f"synthetic_nll_ece_scatter_{tag}.png"
    _write_csv(details_csv, rows)
    _plot_scatter(rows=rows, out_path=plot_png, setup_names=setup_names)

    print("\nSaved:")
    print(f"  details: {details_csv}")
    print(f"  plot:    {plot_png}")


if __name__ == "__main__":
    main()
