"""
run_synthetic_uncertainty_scaling.py

Synthetic-task version of run_uncertainty_scaling.py.

Each generated synthetic task is treated as one classification dataset:
  1) split task into train/test
  2) sample train subsets of size N0 repeatedly
  3) measure replicate disagreement on a fixed query bank
  4) aggregate uncertainty-vs-context scaling across tasks
"""
"""
(Synthetic -> Synthetic)
python run_synthetic_uncertainty_scaling.py \
  --tuned-merged-state /home/boreum/project/tabpfn-scloss/synthetic_model/merged_model_state_generated_train_800_seed42_20260316_053443.pt

"""

"""
(Synthetic -> Real)
python run_synthetic_uncertainty_scaling.py \
  --tuned-merged-state /home/boreum/project/tabpfn-scloss/openml_54/vehicle_model/merged_model_state_openml_54_20260305_152612.pt
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from predictive_rule import ClassifierPredRule
from run_classification import generate_synthetic_task_dataset, set_global_seeds


@dataclass
class ModelSpec:
    name: str
    locked_state: Optional[dict]


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
    return s if s else "dataset"


def _parse_n0_grid(s: str, n_train: int, include_full_train: bool) -> list[int]:
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        vals.append(int(tok))
    if include_full_train:
        vals.append(int(n_train))

    cleaned = sorted({v for v in vals if v > 0})
    cleaned = [min(v, int(n_train)) for v in cleaned]
    cleaned = sorted(set(cleaned))
    if len(cleaned) == 0:
        raise ValueError("N0 grid is empty after filtering.")
    return cleaned


def _entropy_probs(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    p = p / np.clip(p.sum(axis=-1, keepdims=True), eps, None)
    return -np.sum(p * np.log(p), axis=-1)


def js_disagreement_from_replicates(probs_rqc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probs_rqc, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / np.clip(p.sum(axis=-1, keepdims=True), eps, None)
    mean_p = p.mean(axis=0)
    h_mean = _entropy_probs(mean_p, eps=eps)
    h_each = _entropy_probs(p, eps=eps)
    u = h_mean - h_each.mean(axis=0)
    return np.clip(u, 0.0, None)


def load_locked_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError(
        f"Unsupported checkpoint format at {path}. "
        "Expected merged checkpoint with 'state_dict' or raw state_dict."
    )


def sample_query_bank_indices(n_test: int, query_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = n_test < query_count
    return np.array(rng.choice(n_test, size=query_count, replace=replace), dtype=int)


def sample_replicate_subsets(n_train: int, n0: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1009 * int(n0))
    subsets = []
    for _ in range(replicates):
        idx = rng.choice(n_train, size=n0, replace=False)
        subsets.append(np.asarray(idx, dtype=int))
    return np.stack(subsets, axis=0)


def summarize_u(u_q: np.ndarray) -> dict[str, float]:
    uq = np.asarray(u_q, dtype=np.float64)
    return {
        "u_js_median": float(np.median(uq)),
        "u_js_mean": float(np.mean(uq)),
        "u_js_q25": float(np.percentile(uq, 25.0)),
        "u_js_q75": float(np.percentile(uq, 75.0)),
    }


def save_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--query-count", type=int, default=24)
    p.add_argument("--replicates", type=int, default=50)
    p.add_argument("--n0-grid", type=str, default="16,32,64,128")
    p.add_argument("--include-full-train", action="store_true")
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--tuned-merged-state", type=str, default="")
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--output-dir", type=str, default="synthetic_uncertainty_scaling")
    p.add_argument("--show", action="store_true")

    p.add_argument(
        "--synthetic-tasks",
        type=int,
        default=10,
        help="Number of generated synthetic tasks. Each task is treated as one dataset.",
    )
    p.add_argument(
        "--synthetic-seed",
        type=int,
        default=43,
        help="Seed for synthetic task generation.",
    )
    p.add_argument(
        "--synthetic-split-name",
        type=str,
        default="uncertainty_eval",
        help="Split name passed to the synthetic generator.",
    )
    p.add_argument(
        "--task-indices",
        type=str,
        default="",
        help="Optional comma-separated subset of task indices to evaluate.",
    )
    return p.parse_args()


def _parse_optional_task_indices(text: str, n_tasks: int) -> list[int]:
    if str(text).strip() == "":
        return list(range(int(n_tasks)))
    vals = [int(tok.strip()) for tok in str(text).split(",") if tok.strip() != ""]
    if len(vals) == 0:
        raise ValueError("task-indices must be empty or a non-empty comma-separated list.")
    out = sorted({int(v) for v in vals})
    for v in out:
        if v < 0 or v >= int(n_tasks):
            raise ValueError(f"task index out of range: {v} for n_tasks={n_tasks}")
    return out


def _split_one_synthetic_task(
    *,
    x_task: np.ndarray,
    y_task: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            np.asarray(x_task, dtype=np.float32),
            np.asarray(y_task).astype(np.int64),
            test_size=float(test_size),
            random_state=int(seed),
            stratify=np.asarray(y_task).astype(np.int64),
        )
    except ValueError:
        return train_test_split(
            np.asarray(x_task, dtype=np.float32),
            np.asarray(y_task).astype(np.int64),
            test_size=float(test_size),
            random_state=int(seed),
            stratify=None,
        )


def collect_probs_for_model_synthetic(
    *,
    model_spec: ModelSpec,
    categorical_x: list[bool],
    n_estimators: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_queries: np.ndarray,
    subsets_rn: np.ndarray,
    n_classes_global: int,
) -> np.ndarray:
    """
    For one model and one N0 on one synthetic task:
      - fit each replicate subset
      - predict on a fixed query bank
      - map local class axis back to task-global class axis

    Returns:
      probs_rqc: (R, Q, C_global)
    """
    probs_runs = []
    for idx in subsets_rn:
        pred_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
        if model_spec.locked_state is not None:
            pred_rule._locked_state_dict = {
                k: v.clone().cpu() for k, v in model_spec.locked_state.items()
            }

        x_ctx = x_train[idx]
        y_ctx = y_train[idx]
        local_classes, y_local = np.unique(y_ctx.astype(int), return_inverse=True)
        pred_rule.fit(x_ctx, y_local.astype(np.int64))
        probs_local = np.asarray(pred_rule.get_belief(x_queries), dtype=np.float64)

        probs_global = np.zeros((probs_local.shape[0], int(n_classes_global)), dtype=np.float64)
        probs_global[:, local_classes] = probs_local
        probs_runs.append(probs_global)

    return np.stack(probs_runs, axis=0)


def _aggregate_pooled_rows(
    pooled: dict[tuple[str, int], list[np.ndarray]],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for (model_name, n0), arrays in sorted(pooled.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(arrays) == 0:
            continue
        u_all = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        s = summarize_u(u_all)
        rows.append(
            {
                "model": model_name,
                "n0": int(n0),
                "u_js_median": float(s["u_js_median"]),
                "u_js_mean": float(s["u_js_mean"]),
                "u_js_q25": float(s["u_js_q25"]),
                "u_js_q75": float(s["u_js_q75"]),
                "n_queries_total": int(len(u_all)),
                "n_tasks": int(len(arrays)),
            }
        )
    return rows


def _plot_ujs_rows(
    *,
    rows: list[dict],
    model_specs: list[ModelSpec],
    eps: float,
    title: str,
    output_path: str,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    baseline_ns: Optional[np.ndarray] = None
    baseline_med: Optional[np.ndarray] = None

    for model_spec in model_specs:
        model_name = model_spec.name
        model_rows = [r for r in rows if r["model"] == model_name]
        model_rows = sorted(model_rows, key=lambda r: int(r["n0"]))
        if len(model_rows) == 0:
            continue

        ns = np.array([int(r["n0"]) for r in model_rows], dtype=np.int64)
        med = np.array([float(r["u_js_median"]) for r in model_rows], dtype=np.float64)
        q25 = np.array([float(r["u_js_q25"]) for r in model_rows], dtype=np.float64)
        q75 = np.array([float(r["u_js_q75"]) for r in model_rows], dtype=np.float64)

        axes[0].plot(ns, med, marker="o", linewidth=1.8, label=model_name)
        axes[0].fill_between(ns, q25, q75, alpha=0.15)
        axes[1].plot(ns, ns * med, marker="o", linewidth=1.8, label=model_name)
        if model_name == "baseline":
            baseline_ns = ns
            baseline_med = med

    if baseline_ns is not None and baseline_med is not None and len(baseline_ns) > 0:
        n_ref_target = 128
        ref_i = int(np.argmin(np.abs(baseline_ns - n_ref_target)))
        n_ref = float(baseline_ns[ref_i])
        u_ref = float(max(baseline_med[ref_i], eps))
        u_refline = u_ref * (baseline_ns.astype(np.float64) / n_ref) ** (-1.0)
        axes[0].plot(
            baseline_ns,
            u_refline,
            linestyle="--",
            color="black",
            linewidth=1.4,
            label=f"slope=-1 ref (N_ref={int(n_ref)})",
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("N0 (context size)")
    axes[0].set_ylabel("U_JS")
    axes[0].set_title("(A) Log-log: U_JS vs N0")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlabel("N0 (context size)")
    axes[1].set_ylabel("N0 * U_JS(N0)")
    axes[1].set_title("(B) Compensated: N0 * U_JS")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seeds(args.seed)

    x_tasks, y_tasks, categorical_x, dataset_name = generate_synthetic_task_dataset(
        n_tasks=args.synthetic_tasks,
        seed=args.synthetic_seed,
        split_name=args.synthetic_split_name,
    )
    task_indices = _parse_optional_task_indices(args.task_indices, int(x_tasks.shape[0]))

    model_specs = [ModelSpec(name="baseline", locked_state=None)]
    if args.tuned_merged_state.strip() != "":
        tuned_state = load_locked_state(args.tuned_merged_state.strip())
        model_specs.append(ModelSpec(name="tuned", locked_state=tuned_state))

    print("\n── Synthetic Uncertainty Scaling Experiment (n=0; no rollout) ──")
    print(f"  Dataset batch: {dataset_name}")
    print(f"  Tasks used: {task_indices}")
    print(f"  Models: {[m.name for m in model_specs]}")

    per_task_rows: list[dict] = []
    pooled_u_by_model_n0: dict[tuple[str, int], list[np.ndarray]] = {}

    for task_index in task_indices:
        x_task = np.asarray(x_tasks[int(task_index)], dtype=np.float32)
        y_task = np.asarray(y_tasks[int(task_index)]).astype(np.int64)
        n_classes_global = int(np.max(y_task)) + 1

        X_train, X_test, y_train, y_test = _split_one_synthetic_task(
            x_task=x_task,
            y_task=y_task,
            test_size=args.test_size,
            seed=int(args.seed + 1000 * int(task_index)),
        )
        n0_grid = _parse_n0_grid(
            args.n0_grid,
            n_train=len(X_train),
            include_full_train=args.include_full_train,
        )
        query_indices = sample_query_bank_indices(
            n_test=len(X_test),
            query_count=args.query_count,
            seed=int(args.seed + 7000 * int(task_index)),
        )
        x_query_bank = np.asarray(X_test)[query_indices]
        subset_bank = {
            int(n0): sample_replicate_subsets(
                n_train=len(X_train),
                n0=int(n0),
                replicates=args.replicates,
                seed=int(args.seed + 10000 * int(task_index)),
            )
            for n0 in n0_grid
        }

        print(
            f"\n[task={int(task_index)}] train/test={len(X_train)}/{len(X_test)}, "
            f"classes={n_classes_global}, n0_grid={n0_grid}, queries={len(query_indices)}"
        )

        for model_spec in model_specs:
            print(f"  [model={model_spec.name}] collecting probabilities...")
            for n0 in n0_grid:
                probs_rqc = collect_probs_for_model_synthetic(
                    model_spec=model_spec,
                    categorical_x=categorical_x,
                    n_estimators=args.n_estimators,
                    x_train=X_train,
                    y_train=y_train,
                    x_queries=x_query_bank,
                    subsets_rn=subset_bank[int(n0)],
                    n_classes_global=n_classes_global,
                )
                u_q = js_disagreement_from_replicates(probs_rqc, eps=args.eps)
                s = summarize_u(u_q)
                per_task_rows.append(
                    {
                        "task_index": int(task_index),
                        "model": model_spec.name,
                        "n0": int(n0),
                        "u_js_median": float(s["u_js_median"]),
                        "u_js_mean": float(s["u_js_mean"]),
                        "u_js_q25": float(s["u_js_q25"]),
                        "u_js_q75": float(s["u_js_q75"]),
                        "n_train": int(len(X_train)),
                        "n_test": int(len(X_test)),
                        "n_classes": int(n_classes_global),
                        "query_count": int(len(query_indices)),
                    }
                )
                pooled_u_by_model_n0.setdefault((model_spec.name, int(n0)), []).append(u_q)
                print(
                    f"    N0={int(n0):4d} -> "
                    f"U_JS median={s['u_js_median']:.6e}, "
                    f"mean={s['u_js_mean']:.6e}"
                )

    summary_rows = _aggregate_pooled_rows(pooled_u_by_model_n0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = _slugify(dataset_name)
    per_task_csv = os.path.join(args.output_dir, f"synthetic_ujs_per_task_{dataset_tag}_{ts}.csv")
    summary_csv = os.path.join(args.output_dir, f"synthetic_ujs_summary_{dataset_tag}_{ts}.csv")
    save_csv(
        per_task_csv,
        per_task_rows,
        [
            "task_index",
            "model",
            "n0",
            "u_js_median",
            "u_js_mean",
            "u_js_q25",
            "u_js_q75",
            "n_train",
            "n_test",
            "n_classes",
            "query_count",
        ],
    )
    save_csv(
        summary_csv,
        summary_rows,
        [
            "model",
            "n0",
            "u_js_median",
            "u_js_mean",
            "u_js_q25",
            "u_js_q75",
            "n_queries_total",
            "n_tasks",
        ],
    )

    plot_path = os.path.join(
        args.output_dir, f"synthetic_ujs_scaling_plot_{dataset_tag}_{ts}.png"
    )
    _plot_ujs_rows(
        rows=summary_rows,
        model_specs=model_specs,
        eps=args.eps,
        title=f"Synthetic epistemic uncertainty scaling | tasks={len(task_indices)} | R={args.replicates}",
        output_path=plot_path,
        show=args.show,
    )

    print(f"[Saved] {per_task_csv}")
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {plot_path}")


if __name__ == "__main__":
    main()
