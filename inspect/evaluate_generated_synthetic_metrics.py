"""
evaluate_generated_synthetic_metrics.py
------------------------------------------------------------
Evaluate baseline TabPFN on synthetic datasets generated in-memory.

For each generated synthetic task:
  - sample context/query split (default: context=100, query=50)
  - compute Accuracy / NLL / ECE

Outputs:
  - per-task metrics CSV
  - failed-task CSV
  - summary figure (curves + distributions + scatter)


# Example :
python inspect/evaluate_generated_synthetic_metrics.py \
  --setup-group nonlinear_link_setups

python inspect/evaluate_generated_synthetic_metrics.py \
  --setup-group simple_linear
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval import compute_basic_metrics
from generate_synthetic import MixtureConfig, generate_mixture_tensors, make_mixture_config
from predictive_rule import ClassifierPredRule


@dataclass
class DatasetEvalResult:
    data_id: int
    name: str
    setup_name: str
    prior_type: str
    context_size: int
    query_size: int
    n_samples: int
    n_features: int
    n_classes: int
    context_classes: int
    unseen_query_labels: int
    accuracy: float
    nll: float
    ece: float


def _make_forced_prior_config(prior_type: str) -> MixtureConfig:
    cfg = make_mixture_config("mixed_full")
    cfg.p_gbdt = 1.0 if prior_type == "gbdt" else 0.0
    cfg.p_scm = 1.0 if prior_type == "scm" else 0.0
    cfg.p_smooth_mlp = 1.0 if prior_type == "smooth_mlp" else 0.0
    cfg.p_sparse_linear = 1.0 if prior_type == "sparse_linear" else 0.0
    cfg.p_nonlinear_link = 0.0
    return cfg


def _get_setup_specs(group_name: str) -> list[tuple[str, MixtureConfig]]:
    group_norm = str(group_name).strip().lower()
    if group_norm == "standard_priors":
        return [
            ("gbdt", _make_forced_prior_config("gbdt")),
            ("scm", _make_forced_prior_config("scm")),
            ("smooth_mlp", _make_forced_prior_config("smooth_mlp")),
            ("sparse_linear", _make_forced_prior_config("sparse_linear")),
        ]
    if group_norm == "nonlinear_link_setups":
        return [
            ("nonlinear_link_logistic", make_mixture_config("nonlinear_link_logistic")),
            ("nonlinear_link_gmm0", make_mixture_config("nonlinear_link_gmm0")),
            ("nonlinear_link_gmm_neg1", make_mixture_config("nonlinear_link_gmm_neg1")),
            ("nonlinear_link_gmm_neg2", make_mixture_config("nonlinear_link_gmm_neg2")),
        ]
    if group_norm == "simple_linear":
        return [
            ("simple_linear", make_mixture_config("simple_linear")),
        ]
    raise ValueError(
        f"Unknown setup group: {group_name}. "
        "Expected one of: standard_priors, nonlinear_link_setups, simple_linear."
    )


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """Load plain tensor state_dict from common checkpoint formats."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        raw = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        raw = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    state: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            state[str(k)] = v.detach().cpu().clone()
    if len(state) == 0:
        raise ValueError("No tensor state_dict entries found in checkpoint.")
    return state


def _parse_context_sizes(raw: str) -> list[int]:
    vals = []
    for part in str(raw).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(int(s))
    vals = sorted(set(vals))
    if len(vals) == 0:
        raise ValueError("context sizes must not be empty")
    if any(v <= 0 for v in vals):
        raise ValueError(f"context sizes must be positive, got {vals}")
    return vals


def _sample_shared_context_query_indices(
    *,
    y: np.ndarray,
    context_sizes: list[int],
    query_size: int,
    seed: int,
    max_tries: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    n = int(len(y))
    if n < 3:
        raise ValueError(f"Dataset too small: n={n}")
    max_ctx = int(max(context_sizes))
    n_ctx_max = int(np.clip(max_ctx, 1, n - 2))
    n_q = int(np.clip(query_size, 1, n - n_ctx_max - 1))
    clipped_context_sizes = [int(np.clip(v, 1, n - n_q - 1)) for v in context_sizes]

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        perm = np.asarray(rng.permutation(n), dtype=int)
        context_pool_idx = perm[:n_ctx_max]
        idx_q = perm[n_ctx_max:n_ctx_max + n_q]
        if all(len(np.unique(y[context_pool_idx[:ctx]])) >= 2 for ctx in clipped_context_sizes):
            return context_pool_idx, idx_q, clipped_context_sizes
    raise RuntimeError("Failed to sample shared context/query split with >=2 context classes.")


def _evaluate_dataset(
    *,
    data_id: int,
    x: np.ndarray,
    y: np.ndarray,
    setup_name: str,
    prior_type: str,
    context_idx: np.ndarray,
    query_idx: np.ndarray,
    n_estimators: int,
    locked_state_dict: dict[str, torch.Tensor] | None,
) -> DatasetEvalResult:
    x_ctx = np.asarray(x[context_idx], dtype=np.float32)
    y_ctx_global = np.asarray(y[context_idx], dtype=np.int64)
    x_q = np.asarray(x[query_idx], dtype=np.float32)
    y_q_global = np.asarray(y[query_idx], dtype=np.int64)

    local_classes, y_ctx_local = np.unique(y_ctx_global, return_inverse=True)
    y_ctx_local = y_ctx_local.astype(np.int64)
    c_global = int(np.max(y)) + 1

    rule = ClassifierPredRule([False] * int(x.shape[1]), n_estimators=n_estimators)
    if locked_state_dict is not None:
        rule._locked_state_dict = {k: v.clone().cpu() for k, v in locked_state_dict.items()}

    rule.fit(x_ctx, y_ctx_local)
    with torch.no_grad():
        probs_local = rule.get_belief_torch(x_q, x_ctx, y_ctx_local).cpu().numpy()

    probs_global = np.zeros((len(x_q), c_global), dtype=np.float64)
    probs_global[:, np.asarray(local_classes).astype(int)] = np.asarray(probs_local, dtype=np.float64)
    probs_global = probs_global / np.clip(probs_global.sum(axis=1, keepdims=True), 1e-12, None)
    m = compute_basic_metrics(probs_global, y_q_global)
    unseen = int(np.sum(~np.isin(y_q_global, local_classes)))

    return DatasetEvalResult(
        data_id=int(data_id),
        name=f"synthetic_{int(data_id):03d}",
        setup_name=str(setup_name),
        prior_type=str(prior_type),
        context_size=int(len(context_idx)),
        query_size=int(len(query_idx)),
        n_samples=int(len(y)),
        n_features=int(x.shape[1]),
        n_classes=int(c_global),
        context_classes=int(len(local_classes)),
        unseen_query_labels=unseen,
        accuracy=float(m.accuracy),
        nll=float(m.nll),
        ece=float(m.ece),
    )


def _plot_summary(
    *,
    rows_by_context: dict[int, list[DatasetEvalResult]],
    query_size: int,
    title: str,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    context_sizes = sorted(rows_by_context)
    colors = plt.cm.tab10(np.linspace(0, 1, len(context_sizes)))

    ax = axes[0, 0]
    for color, ctx in zip(colors, context_sizes):
        rows = rows_by_context[ctx]
        idx = np.arange(len(rows))
        acc = np.asarray([r.accuracy for r in rows], dtype=np.float64)
        ax.plot(idx, acc, marker="o", linewidth=1.2, markersize=3.5, color=color, alpha=0.9, label=f"ctx={ctx}")
        ax.axhline(float(np.mean(acc)), linestyle="--", color=color, alpha=0.35)
    ax.set_title("Accuracy by Dataset")
    ax.set_xlabel("Dataset rank")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for color, ctx in zip(colors, context_sizes):
        rows = rows_by_context[ctx]
        idx = np.arange(len(rows))
        nll = np.asarray([r.nll for r in rows], dtype=np.float64)
        ax.plot(idx, nll, marker="o", linewidth=1.2, markersize=3.5, color=color, alpha=0.9, label=f"ctx={ctx}")
        ax.axhline(float(np.mean(nll)), linestyle="--", color=color, alpha=0.35)
    ax.set_title("NLL by Dataset")
    ax.set_xlabel("Dataset rank")
    ax.set_ylabel("NLL")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    bins = min(14, max(5, len(next(iter(rows_by_context.values()))) // 3 if len(rows_by_context) > 0 else 5))
    for color, ctx in zip(colors, context_sizes):
        rows = rows_by_context[ctx]
        acc = np.asarray([r.accuracy for r in rows], dtype=np.float64)
        ax.hist(acc, bins=bins, alpha=0.35, color=color, label=f"ctx={ctx}")
    ax.set_title("Accuracy Distributions")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    for color, ctx in zip(colors, context_sizes):
        rows = rows_by_context[ctx]
        nll = np.asarray([r.nll for r in rows], dtype=np.float64)
        ax.hist(nll, bins=bins, alpha=0.35, color=color, label=f"ctx={ctx}")
    ax.set_title("NLL Distributions")
    ax.set_xlabel("NLL")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(
        f"{title} "
        f"(contexts={','.join(str(v) for v in context_sizes)}, query={query_size}, "
        f"n={len(next(iter(rows_by_context.values())))})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-datasets",
        type=int,
        default=100,
        help="Number of generated tasks per setup (total tasks = 4 * n-datasets)",
    )
    p.add_argument(
        "--setup-group",
        type=str,
        default="standard_priors",
        choices=["standard_priors", "nonlinear_link_setups", "simple_linear"],
        help="Which synthetic setup family to evaluate.",
    )
    p.add_argument(
        "--context-sizes",
        type=str,
        default="50,100,150",
        help="Comma-separated context sizes to compare, e.g. '50,100,150'",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=0,
        help="Legacy single context-size override. If >0, this overrides --context-sizes.",
    )
    p.add_argument("--query-size", type=int, default=50, help="Query size")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--n-estimators", type=int, default=4, help="TabPFN n_estimators")
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional merged model .pt path. If provided, evaluate with locked weights.",
    )
    p.add_argument("--save-dir", type=str, default="generated_synthetic_eval", help="Output dir")
    p.add_argument("--csv-name", type=str, default="generated_synthetic_metrics.csv", help="Metrics CSV filename")
    p.add_argument(
        "--fail-csv-name",
        type=str,
        default="generated_synthetic_failures.csv",
        help="Failure CSV filename",
    )
    p.add_argument(
        "--plot-name",
        type=str,
        default="generated_synthetic_summary.png",
        help="Summary plot filename",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    context_sizes = (
        [int(args.context_size)]
        if int(args.context_size) > 0
        else _parse_context_sizes(args.context_sizes)
    )

    setup_specs = _get_setup_specs(args.setup_group)
    setup_order = tuple(name for name, _ in setup_specs)
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    setup_names: list[str] = []
    per_prior = int(args.n_datasets)
    for setup_idx, (setup_name, cfg) in enumerate(setup_specs):
        x_block, y_block, metas_block = generate_mixture_tensors(
            n_tasks=per_prior,
            cfg=cfg,
            seed=int(args.seed + setup_idx * 100_003),
            return_metadata=True,
        )
        if metas_block is None:
            raise RuntimeError(f"Expected metadata from synthetic generator for setup={setup_name}.")
        x_blocks.append(np.asarray(x_block, dtype=np.float32))
        y_blocks.append(np.asarray(y_block, dtype=np.int64))
        metas.extend(list(metas_block))
        setup_names.extend([str(setup_name)] * int(x_block.shape[0]))

    x_tasks = np.concatenate(x_blocks, axis=0)
    y_tasks = np.concatenate(y_blocks, axis=0)

    print("── Generated Synthetic Diagnostics ──")
    print(
        f"setup_group={args.setup_group}, tasks={x_tasks.shape[0]} ({per_prior} per setup), points={x_tasks.shape[1]}, "
        f"dim={x_tasks.shape[2]}, contexts={context_sizes}, query={args.query_size}, "
        f"n_estimators={args.n_estimators}"
    )

    locked_state = None
    if args.model_path.strip():
        model_path = os.path.abspath(args.model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        locked_state = _extract_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model state: {model_path} ({len(locked_state)} tensors)")
    else:
        print("Model path not provided: evaluating baseline (untrained) TabPFN.")

    rows: list[DatasetEvalResult] = []
    rows_by_context: dict[int, list[DatasetEvalResult]] = {ctx: [] for ctx in context_sizes}
    failures: list[dict[str, str]] = []
    for i in range(int(x_tasks.shape[0])):
        try:
            seed_i = int(args.seed + i * 10007)
            context_pool_idx, query_idx, clipped_context_sizes = _sample_shared_context_query_indices(
                y=np.asarray(y_tasks[i], dtype=np.int64),
                context_sizes=context_sizes,
                query_size=int(args.query_size),
                seed=seed_i,
            )
            x_i = np.asarray(x_tasks[i], dtype=np.float32)
            y_i = np.asarray(y_tasks[i], dtype=np.int64)
            setup_name = str(setup_names[i])
            prior_type = str(metas[i].get("prior_type", "unknown"))

            for ctx in clipped_context_sizes:
                out = _evaluate_dataset(
                    data_id=i,
                    x=x_i,
                    y=y_i,
                    setup_name=setup_name,
                    prior_type=prior_type,
                    context_idx=context_pool_idx[:ctx],
                    query_idx=query_idx,
                    n_estimators=int(args.n_estimators),
                    locked_state_dict=locked_state,
                )
                rows.append(out)
                rows_by_context[ctx].append(out)
                print(
                    f"[{i+1:3d}/{len(x_tasks)}][ctx={ctx:3d}] did={out.data_id:<6d} "
                    f"name={out.name:<22} "
                    f"acc={out.accuracy:.4f} nll={out.nll:.4f} "
                    f"(setup={out.setup_name}, prior={out.prior_type}, cls={out.n_classes}, "
                    f"ctx_cls={out.context_classes}, unseen_q={out.unseen_query_labels})"
                )
        except Exception as e:
            failures.append(
                {
                    "data_id": str(int(i)),
                    "context_size": ",".join(str(v) for v in context_sizes),
                    "error": repr(e),
                }
            )
            print(f"[{i+1:3d}/{len(x_tasks)}] did={int(i):<6d} FAILED: {repr(e)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f"{os.path.splitext(args.csv_name)[0]}_{ts}.csv")
    fail_csv_path = os.path.join(args.save_dir, f"{os.path.splitext(args.fail_csv_name)[0]}_{ts}.csv")
    plot_base = os.path.join(args.save_dir, f"{os.path.splitext(args.plot_name)[0]}_{ts}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "data_id",
            "name",
            "setup_name",
            "prior_type",
            "context_size",
            "query_size",
            "n_samples",
            "n_features",
            "n_classes",
            "context_classes",
            "unseen_query_labels",
            "accuracy",
            "nll",
            "ece",
        ])
        for r in rows:
            writer.writerow([
                r.data_id,
                r.name,
                r.setup_name,
                r.prior_type,
                r.context_size,
                r.query_size,
                r.n_samples,
                r.n_features,
                r.n_classes,
                r.context_classes,
                r.unseen_query_labels,
                r.accuracy,
                r.nll,
                r.ece,
            ])
    print(f"Saved metrics CSV: {csv_path}")

    with open(fail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data_id", "context_size", "error"])
        for item in failures:
            writer.writerow([item["data_id"], item["context_size"], item["error"]])
    print(f"Saved failures CSV: {fail_csv_path}")

    if len(rows) > 0:
        print("\n── Summary ──")
        for setup_name in setup_order:
            setup_rows = [r for r in rows if r.setup_name == setup_name]
            if len(setup_rows) == 0:
                print(f"{setup_name}: no successful evaluations")
                continue

            setup_rows_by_context: dict[int, list[DatasetEvalResult]] = {}
            for ctx in context_sizes:
                ctx_rows = [r for r in setup_rows if int(r.context_size) == int(ctx)]
                if len(ctx_rows) > 0:
                    setup_rows_by_context[int(ctx)] = ctx_rows

            if len(setup_rows_by_context) == len(context_sizes):
                plot_path = f"{plot_base}_{setup_name}.png"
                _plot_summary(
                    rows_by_context=setup_rows_by_context,
                    query_size=int(args.query_size),
                    title=f"Synthetic generated-set diagnostics ({setup_name})",
                    save_path=plot_path,
                )
                print(f"Saved summary plot: {plot_path}")

            print(f"{setup_name}: success {len(setup_rows)}")
            for ctx in sorted(setup_rows_by_context):
                ctx_rows = setup_rows_by_context[ctx]
                acc = np.asarray([r.accuracy for r in ctx_rows], dtype=np.float64)
                nll = np.asarray([r.nll for r in ctx_rows], dtype=np.float64)
                print(f"  Context {ctx}: {len(ctx_rows)} evals")
                print(f"    Accuracy mean±std: {float(acc.mean()):.4f} ± {float(acc.std()):.4f}")
                print(f"    NLL mean±std:      {float(nll.mean()):.4f} ± {float(nll.std()):.4f}")
    else:
        print("\nNo successful dataset evaluations. Check failures CSV.")


if __name__ == "__main__":
    main()
