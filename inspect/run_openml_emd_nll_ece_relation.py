"""
run_openml_emd_nll_ece_relation.py

Baseline-only relation experiment on a single real dataset (OpenML 54 / Vehicle).

Goal:
  - sample many context/query/rollout splits from the same dataset
  - compute one scalar EMD and one scalar NLL/ECE per split
  - inspect whether lower EMD tends to imply lower NLL/ECE

Setup:
  - dataset: OpenML 54
  - model: baseline TabPFN only
  - per repeat: context=100, query=20, rollout=rest
  - EMD: all 20 query points, prefix depth fixed to 0, k in {3,5,7,9}
  - ECE: low-bin setting by default (5 bins)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval import _compute_emd_fixed_anchor_suite, compute_basic_metrics
from predictive_rule import ClassifierPredRule
from run_openml_classification import load_openml_dataset_split


@dataclass
class RepeatResult:
    repeat_id: int
    context_size: int
    query_size: int
    rollout_size: int
    context_classes: int
    unseen_query_labels: int
    accuracy: float
    nll: float
    ece: float
    emd: float
    emd_std: float
    emd_coverage: int


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _sample_repeat_split(
    *,
    y: np.ndarray,
    rng: np.random.Generator,
    context_size: int,
    query_size: int,
    require_all_context_classes: bool,
    max_tries: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(y))
    if n < context_size + query_size + 1:
        raise ValueError(
            f"Dataset too small for requested split: n={n}, context={context_size}, query={query_size}"
        )
    all_classes = np.unique(y)
    for _ in range(max_tries):
        perm = np.asarray(rng.permutation(n), dtype=int)
        idx_ctx = perm[:context_size]
        idx_query = perm[context_size:context_size + query_size]
        idx_roll = perm[context_size + query_size:]
        y_ctx = np.asarray(y[idx_ctx]).astype(int)
        if require_all_context_classes and len(np.unique(y_ctx)) < len(all_classes):
            continue
        if len(np.unique(y_ctx)) < 2:
            continue
        return idx_ctx, idx_query, idx_roll
    raise RuntimeError("Failed to sample a valid context/query/rollout split.")


def _evaluate_repeat(
    *,
    repeat_id: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
    categorical_x: list[bool],
    context_idx: np.ndarray,
    query_idx: np.ndarray,
    rollout_idx: np.ndarray,
    n_estimators: int,
    ece_bins: int,
    k_values: tuple[int, ...],
    continuation_depth: int,
    n_continuations: int,
    seed: int,
) -> RepeatResult:
    x_ctx = np.asarray(x_all[context_idx], dtype=np.float32)
    y_ctx_global = np.asarray(y_all[context_idx]).astype(np.int64)
    x_query = np.asarray(x_all[query_idx], dtype=np.float32)
    y_query = np.asarray(y_all[query_idx]).astype(np.int64)
    x_roll = np.asarray(x_all[rollout_idx], dtype=np.float32)

    local_classes, y_ctx_local = np.unique(y_ctx_global, return_inverse=True)
    y_ctx_local = y_ctx_local.astype(np.int64)
    global_num_classes = int(np.max(y_all)) + 1

    sampling_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    train_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    sampling_rule.fit(x_ctx, y_ctx_local)
    train_rule.fit(x_ctx, y_ctx_local)

    with torch.no_grad():
        probs_local = train_rule.get_belief_torch(x_query, x_ctx, y_ctx_local).cpu().numpy()
    probs_global = np.zeros((len(x_query), global_num_classes), dtype=np.float64)
    probs_global[:, np.asarray(local_classes).astype(int)] = np.asarray(probs_local, dtype=np.float64)
    probs_global = probs_global / np.clip(probs_global.sum(axis=1, keepdims=True), 1e-12, None)
    metrics = compute_basic_metrics(probs_global, y_query, n_bins=int(ece_bins))

    key = jax.random.PRNGKey(int(seed))
    key, subkey = jax.random.split(key)
    fixed_rollout_keys = {(0, 0): subkey}
    emd_mean, emd_std, emd_cov, _ = _compute_emd_fixed_anchor_suite(
        key=key,
        pred_rule_train=train_rule,
        pred_rule_sampling=sampling_rule,
        anchor_contexts=[(x_ctx, y_ctx_local)],
        anchor_query_banks=[x_query],
        prefix_depths=(0,),
        k_values=k_values,
        continuation_depth=int(continuation_depth),
        n_continuations=int(n_continuations),
        fixed_rollout_keys=fixed_rollout_keys,
        anchor_rollout_pools=[x_roll],
    )
    unseen_query_labels = int(np.sum(~np.isin(y_query, local_classes)))
    return RepeatResult(
        repeat_id=int(repeat_id),
        context_size=int(len(context_idx)),
        query_size=int(len(query_idx)),
        rollout_size=int(len(rollout_idx)),
        context_classes=int(len(local_classes)),
        unseen_query_labels=unseen_query_labels,
        accuracy=float(metrics.accuracy),
        nll=float(metrics.nll),
        ece=float(metrics.ece),
        emd=float(emd_mean),
        emd_std=float(emd_std),
        emd_coverage=int(emd_cov),
    )


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_scatter(
    *,
    rows: list[RepeatResult],
    dataset_name: str,
    out_path: Path,
) -> None:
    emd = np.asarray([r.emd for r in rows], dtype=np.float64)
    nll = np.asarray([r.nll for r in rows], dtype=np.float64)
    ece = np.asarray([r.ece for r in rows], dtype=np.float64)
    pearson_nll = _pearson_corr(emd, nll)
    spearman_nll = _spearman_corr(emd, nll)
    pearson_ece = _pearson_corr(emd, ece)
    spearman_ece = _spearman_corr(emd, ece)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    plots = [
        ("NLL", nll, pearson_nll, spearman_nll),
        ("ECE", ece, pearson_ece, spearman_ece),
    ]
    for ax, (name, values, pearson, spearman) in zip(axes, plots):
        ax.scatter(
            emd,
            values,
            s=62,
            c="#1f77b4",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_title(f"EMD vs {name}")
        ax.set_xlabel("EMD")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.text(
            0.03,
            0.97,
            f"Pearson={pearson:.3f}\nSpearman={spearman:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    fig.suptitle(f"OpenML 54 ({dataset_name}) baseline relation: EMD vs NLL/ECE", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--openml-data-id", type=int, default=54)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.25, help="Only used to access the loader; train/test are recombined.")
    p.add_argument("--n-repeats", type=int, default=200)
    p.add_argument("--context-size", type=int, default=100)
    p.add_argument("--query-size", type=int, default=20)
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--ece-bins", type=int, default=5)
    p.add_argument("--k-values", type=str, default="3,5,7,9")
    p.add_argument("--continuation-depth", type=int, default=20)
    p.add_argument("--n-continuations", type=int, default=8)
    p.add_argument("--require-all-context-classes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", type=str, default="openml_emd_relation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    X_train, X_test, y_train, y_test, categorical_x, le, dataset_name = load_openml_dataset_split(
        data_id=int(args.openml_data_id),
        seed=int(args.seed),
        test_size=float(args.test_size),
    )
    del le
    x_all = np.concatenate([np.asarray(X_train, dtype=np.float32), np.asarray(X_test, dtype=np.float32)], axis=0)
    y_all = np.concatenate([np.asarray(y_train, dtype=np.int64), np.asarray(y_test, dtype=np.int64)], axis=0)

    k_values = tuple(int(v.strip()) for v in str(args.k_values).split(",") if v.strip())
    if len(k_values) == 0:
        raise ValueError("k-values must not be empty.")

    print("── OpenML EMD/NLL/ECE Relation ──")
    print(
        f"dataset={dataset_name} (OpenML {args.openml_data_id}), n={len(y_all)}, d={x_all.shape[1]}, "
        f"classes={int(np.max(y_all)) + 1}"
    )
    print(
        f"repeats={args.n_repeats}, context={args.context_size}, query={args.query_size}, "
        f"rollout={len(y_all) - int(args.context_size) - int(args.query_size)}, "
        f"k={k_values}, prefix=(0,), ece_bins={args.ece_bins}"
    )

    rows: list[RepeatResult] = []
    split_rng = np.random.default_rng(int(args.seed))
    for repeat_id in range(int(args.n_repeats)):
        idx_ctx, idx_query, idx_roll = _sample_repeat_split(
            y=y_all,
            rng=split_rng,
            context_size=int(args.context_size),
            query_size=int(args.query_size),
            require_all_context_classes=bool(args.require_all_context_classes),
        )
        result = _evaluate_repeat(
            repeat_id=repeat_id,
            x_all=x_all,
            y_all=y_all,
            categorical_x=categorical_x,
            context_idx=idx_ctx,
            query_idx=idx_query,
            rollout_idx=idx_roll,
            n_estimators=int(args.n_estimators),
            ece_bins=int(args.ece_bins),
            k_values=k_values,
            continuation_depth=int(args.continuation_depth),
            n_continuations=int(args.n_continuations),
            seed=int(args.seed + repeat_id * 1009),
        )
        rows.append(result)
        print(
            f"[{repeat_id+1:2d}/{args.n_repeats}] "
            f"acc={result.accuracy:.4f} nll={result.nll:.4f} ece={result.ece:.4f} "
            f"emd={result.emd:.6f} (ctx_cls={result.context_classes}, unseen_q={result.unseen_query_labels})"
        )

    details_rows = [
        {
            "repeat_id": r.repeat_id,
            "context_size": r.context_size,
            "query_size": r.query_size,
            "rollout_size": r.rollout_size,
            "context_classes": r.context_classes,
            "unseen_query_labels": r.unseen_query_labels,
            "accuracy": r.accuracy,
            "nll": r.nll,
            "ece": r.ece,
            "emd": r.emd,
            "emd_std": r.emd_std,
            "emd_coverage": r.emd_coverage,
        }
        for r in rows
    ]
    emd = np.asarray([r.emd for r in rows], dtype=np.float64)
    nll = np.asarray([r.nll for r in rows], dtype=np.float64)
    ece = np.asarray([r.ece for r in rows], dtype=np.float64)
    corr_rows = [
        {
            "dataset_name": dataset_name,
            "openml_data_id": int(args.openml_data_id),
            "n_repeats": int(args.n_repeats),
            "emd_vs_nll_pearson": _pearson_corr(emd, nll),
            "emd_vs_nll_spearman": _spearman_corr(emd, nll),
            "emd_vs_ece_pearson": _pearson_corr(emd, ece),
            "emd_vs_ece_spearman": _spearman_corr(emd, ece),
        }
    ]

    details_csv = out_dir / f"openml_emd_relation_details_{tag}.csv"
    corr_csv = out_dir / f"openml_emd_relation_correlations_{tag}.csv"
    plot_png = out_dir / f"openml_emd_relation_scatter_{tag}.png"

    _save_csv(
        details_csv,
        details_rows,
        fieldnames=[
            "repeat_id",
            "context_size",
            "query_size",
            "rollout_size",
            "context_classes",
            "unseen_query_labels",
            "accuracy",
            "nll",
            "ece",
            "emd",
            "emd_std",
            "emd_coverage",
        ],
    )
    _save_csv(
        corr_csv,
        corr_rows,
        fieldnames=[
            "dataset_name",
            "openml_data_id",
            "n_repeats",
            "emd_vs_nll_pearson",
            "emd_vs_nll_spearman",
            "emd_vs_ece_pearson",
            "emd_vs_ece_spearman",
        ],
    )
    _plot_scatter(rows=rows, dataset_name=dataset_name, out_path=plot_png)

    print("\nSaved:")
    print(f"  details:      {details_csv}")
    print(f"  correlations: {corr_csv}")
    print(f"  plot:         {plot_png}")


if __name__ == "__main__":
    main()
