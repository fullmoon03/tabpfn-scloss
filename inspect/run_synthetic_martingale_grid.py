"""
run_synthetic_martingale_grid.py
------------------------------------------------------------
TabPFN martingale 성질 시각 점검용 스크립트 (synthetic task).

- inspect/generate_synthetic.py 에서 synthetic task batch를 생성한 뒤
- 그중 한 task를 선택해 context/query/rollout pool 분할
- query pool에서 query 12개 샘플
- query별 fixed-query mean belief across paths 계산
- 3x4 grid 한 장 이미지로 저장
"""
"""
python inspect/run_synthetic_martingale_grid.py

python inspect/run_synthetic_martingale_grid.py --model-path /absolute/path/to/model.pt
"""

import argparse
import os
import re
import sys
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make project-root modules importable when this script is run from inspect/.
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generate_synthetic import MixtureConfig, generate_mixture_tensors
from predictive_rule import ClassifierPredRule
from rollout import belief_at_depth_torch, rollout_one_trajectory_data_only


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return s if s else "synthetic"


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """Load a plain tensor state_dict from common checkpoint formats."""
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


def _make_pred_rule_factory(
    *,
    categorical_x: list[bool],
    n_estimators: int,
    locked_state_dict: dict[str, torch.Tensor] | None = None,
):
    """Return a factory that reuses a single pred_rule instance."""
    shared_pred = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    if locked_state_dict is not None:
        shared_pred._locked_state_dict = {
            k: v.clone().cpu() for k, v in locked_state_dict.items()
        }

    def factory():
        return shared_pred

    return factory


def _relabel_to_contiguous(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    class_ids, y_local = np.unique(y, return_inverse=True)
    return y_local.astype(np.int64), class_ids.astype(np.int64)


def _load_synthetic_rollout_pools(
    *,
    generation_seed: int,
    task_index: int,
    n_generated_tasks: int,
    query_pool_size: int,
    base_n: int,
    query_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[bool],
    np.ndarray,
    str,
]:
    cfg = MixtureConfig()
    x_tasks, y_tasks, metas = generate_mixture_tensors(
        n_tasks=int(n_generated_tasks),
        cfg=cfg,
        seed=int(generation_seed),
        return_metadata=True,
    )
    if metas is None:
        raise RuntimeError("Expected metadata from synthetic generator.")

    n_tasks, n_points, d = x_tasks.shape

    rng = np.random.default_rng(generation_seed)
    if task_index < 0:
        t_idx = int(rng.integers(0, n_tasks))
    else:
        t_idx = int(np.clip(task_index, 0, n_tasks - 1))

    x_task = np.asarray(x_tasks[t_idx], dtype=np.float32)
    y_task = np.asarray(y_tasks[t_idx], dtype=np.int64)
    meta = metas[t_idx]
    task_classes, task_counts = np.unique(y_task, return_counts=True)
    task_dist_text = ", ".join(
        [
            f"class {int(c)}: {int(cnt)} ({float(cnt) / float(len(y_task)):.1%})"
            for c, cnt in zip(task_classes.tolist(), task_counts.tolist())
        ]
    )
    print(
        f"Synthetic task class distribution: task={t_idx}, "
        f"n_samples={len(y_task)} -> {task_dist_text}"
    )

    required_points = int(base_n + query_count + 1)
    if n_points < required_points:
        raise ValueError(
            f"Synthetic task too small: n_points={n_points}, requires at least {required_points}."
        )

    n_ctx_pool = int(np.clip(base_n, 1, n_points - query_count - 1))
    rem_after_ctx = n_points - n_ctx_pool
    n_q_pool = int(np.clip(max(query_pool_size, query_count), query_count, rem_after_ctx - 1))

    perm = np.asarray(rng.permutation(n_points), dtype=int)
    idx_ctx = perm[:n_ctx_pool]
    idx_rest = perm[n_ctx_pool:]
    idx_q = np.asarray(rng.choice(idx_rest, size=n_q_pool, replace=False), dtype=int)
    q_mask = np.zeros(n_points, dtype=bool)
    q_mask[idx_q] = True
    idx_roll = idx_rest[~q_mask[idx_rest]]
    if len(idx_roll) == 0:
        idx_roll = idx_q

    x_context_pool = np.asarray(x_task[idx_ctx], dtype=np.float32)
    y_context_pool_raw = np.asarray(y_task[idx_ctx], dtype=np.int64)
    y_context_pool, class_ids = _relabel_to_contiguous(y_context_pool_raw)

    x_query_pool = np.asarray(x_task[idx_q], dtype=np.float32)
    y_query_pool = np.asarray(y_task[idx_q], dtype=np.int64)
    x_sampling_pool = np.asarray(x_task[idx_roll], dtype=np.float32)
    categorical_x = [False] * int(d)
    class_names = np.asarray([str(int(c)) for c in class_ids])

    dataset_name = (
        f"generated_synth_task{t_idx}_"
        f"{_slugify(str(meta.get('prior_type', 'synthetic')))}"
    )
    print(
        "Synthetic task split: "
        f"task={t_idx}, context={len(x_context_pool)}, "
        f"query_pool={len(x_query_pool)}, rollout_pool={len(x_sampling_pool)}, "
        f"local_classes={class_ids.tolist()}, prior={meta.get('prior_type', 'unknown')}"
    )

    return (
        x_context_pool,
        x_query_pool,
        y_context_pool,
        y_query_pool,
        x_sampling_pool,
        categorical_x,
        class_names,
        dataset_name,
    )


def _compute_fixed_query_rollout_stats(
    *,
    x_context_pool: np.ndarray,
    y_context_pool: np.ndarray,
    x_query_pool: np.ndarray,
    x_sampling_pool: np.ndarray,
    pred_rule_factory,
    depth: int,
    n_paths: int,
    query_index: int,
    seed: int,
) -> dict[str, Any]:
    q_idx = int(np.clip(query_index, 0, len(x_query_pool) - 1))
    x_q = np.atleast_2d(x_query_pool[q_idx])

    x0 = np.asarray(x_context_pool, dtype=np.float32)
    y0 = np.asarray(y_context_pool, dtype=np.int64)

    beliefs_runs: list[np.ndarray] = []
    for j in range(int(n_paths)):
        key = jax.random.PRNGKey(int(seed + j))
        pred_rule = pred_rule_factory()
        traj = rollout_one_trajectory_data_only(
            key=key,
            pred_rule=pred_rule,
            x0=x0,
            y0=y0,
            depth=int(depth),
            x_sampling_pool=x_sampling_pool,
            x_sample_without_replacement=False,
        )

        beliefs_per_depth: list[np.ndarray] = []
        for d in range(int(depth) + 1):
            with torch.no_grad():
                b = belief_at_depth_torch(pred_rule, traj, d, x_q)
            beliefs_per_depth.append(
                np.asarray(b.detach().cpu().numpy().squeeze(0), dtype=np.float64)
            )
        beliefs_runs.append(np.stack(beliefs_per_depth, axis=0))

    beliefs_runs_arr = np.stack(beliefs_runs, axis=0)  # (J, N+1, C)
    mean_beliefs = beliefs_runs_arr.mean(axis=0)
    if int(n_paths) > 1:
        std_beliefs = beliefs_runs_arr.std(axis=0, ddof=1)
    else:
        std_beliefs = np.zeros_like(mean_beliefs)
    max_abs_sum_error = float(np.max(np.abs(beliefs_runs_arr.sum(axis=2) - 1.0)))

    return {
        "q_idx": q_idx,
        "xs": np.arange(int(depth) + 1),
        "beliefs_runs": beliefs_runs_arr,
        "mean_beliefs": mean_beliefs,
        "std_beliefs": std_beliefs,
        "max_abs_sum_error": max_abs_sum_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--n-generated-tasks",
        type=int,
        default=100,
        help="Number of synthetic tasks to generate in-memory",
    )
    parser.add_argument(
        "--synthetic-task-index",
        type=int,
        default=-1,
        help="Task index in generated batch (-1: random by seed)",
    )
    parser.add_argument("--query-pool-size", type=int, default=20, help="Query pool size")

    parser.add_argument("--query-count", type=int, default=12, help="Number of sampled queries")
    parser.add_argument("--n-rows", type=int, default=3, help="Rows in single figure")
    parser.add_argument("--n-cols", type=int, default=4, help="Cols in single figure")
    parser.add_argument("--n-paths", type=int, default=8, help="Number of independent paths")
    parser.add_argument("--depth", type=int, default=30, help="Rollout depth")
    parser.add_argument("--base-n", type=int, default=100, help="Base context size")
    parser.add_argument("--n-estimators", type=int, default=4, help="TabPFN n_estimators")

    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional merged model .pt path. If set, rollout uses this locked state.",
    )
    parser.add_argument(
        "--save-dir", type=str, default="martingale_plots", help="Directory to save plot"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="synthetic_fixed_query_mean_belief_grid.png",
        help="Output image filename",
    )
    parser.add_argument("--show", action="store_true", help="Show figure interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    slots = int(args.n_rows) * int(args.n_cols)
    if int(args.query_count) > slots:
        raise ValueError(
            f"query-count({args.query_count}) must be <= n_rows*n_cols({slots}) for one image."
        )

    (
        x_context_pool,
        x_query_pool,
        y_context_pool,
        y_query_pool,
        x_sampling_pool,
        categorical_x,
        class_names,
        dataset_name,
    ) = _load_synthetic_rollout_pools(
        generation_seed=args.seed,
        task_index=args.synthetic_task_index,
        n_generated_tasks=args.n_generated_tasks,
        query_pool_size=args.query_pool_size,
        base_n=args.base_n,
        query_count=args.query_count,
    )

    if len(x_query_pool) == 0:
        raise ValueError("Empty query pool after split.")

    pred_rule_factory = None
    if args.model_path.strip():
        model_path = os.path.abspath(args.model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")
        locked_state = _extract_state_dict(ckpt)
        pred_rule_factory = _make_pred_rule_factory(
            categorical_x=categorical_x,
            n_estimators=args.n_estimators,
            locked_state_dict=locked_state,
        )
        print(f"Loaded model state: {model_path} ({len(locked_state)} tensors)")
    else:
        pred_rule_factory = _make_pred_rule_factory(
            categorical_x=categorical_x,
            n_estimators=args.n_estimators,
            locked_state_dict=None,
        )
        print("Model path not provided: using baseline (untrained) predictive rule.")

    rng = np.random.default_rng(args.seed)
    replace = len(x_query_pool) < int(args.query_count)
    query_indices = np.asarray(
        rng.choice(len(x_query_pool), size=int(args.query_count), replace=replace),
        dtype=int,
    )
    qpool_classes, qpool_counts = np.unique(y_query_pool, return_counts=True)
    qpool_dist = ", ".join(
        [f"class {int(c)}: {int(n)}" for c, n in zip(qpool_classes.tolist(), qpool_counts.tolist())]
    )
    print(f"Query pool class distribution (n={len(y_query_pool)}): {qpool_dist}")

    selected_query_labels = np.asarray(y_query_pool[query_indices], dtype=np.int64)
    sel_classes, sel_counts = np.unique(selected_query_labels, return_counts=True)
    sel_dist = ", ".join(
        [f"class {int(c)}: {int(n)}" for c, n in zip(sel_classes.tolist(), sel_counts.tolist())]
    )
    print(f"Selected {len(query_indices)} queries from query pool: {query_indices.tolist()}")
    print(f"Selected query class distribution (n={len(query_indices)}): {sel_dist}")
    print(f"Selected query classes by index: {selected_query_labels.tolist()}")

    print(
        "Context size: "
        f"{len(x_context_pool)} "
        f"(classes={len(np.unique(y_context_pool))})"
    )

    results = []
    for rank, q_idx in enumerate(query_indices):
        stats = _compute_fixed_query_rollout_stats(
            x_context_pool=x_context_pool,
            y_context_pool=y_context_pool,
            x_query_pool=x_query_pool,
            x_sampling_pool=x_sampling_pool,
            pred_rule_factory=pred_rule_factory,
            depth=args.depth,
            n_paths=args.n_paths,
            query_index=int(q_idx),
            seed=args.seed + rank * 1000,
        )
        results.append(
            {
                "query_index": int(q_idx),
                "xs": stats["xs"],
                "mean_beliefs": stats["mean_beliefs"],
                "max_abs_sum_error": float(stats["max_abs_sum_error"]),
            }
        )
        print(
            f"[{rank+1:2d}/{len(query_indices)}] q_idx={int(q_idx)} "
            f"shape={tuple(stats['mean_beliefs'].shape)} "
            f"max|sum-1|={float(stats['max_abs_sum_error']):.8f}"
        )

    fig, axes = plt.subplots(
        args.n_rows,
        args.n_cols,
        figsize=(args.n_cols * 5.0, args.n_rows * 3.7),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(results):
            ax.axis("off")
            continue

        item = results[ax_i]
        xs = item["xs"]
        mean_beliefs = item["mean_beliefs"]
        n_classes = mean_beliefs.shape[1]

        for c in range(n_classes):
            label = None
            if ax_i == 0:
                label = f"class {c} ({str(class_names[c])})"
            ax.plot(
                xs,
                mean_beliefs[:, c],
                marker="o",
                markersize=2.4,
                linewidth=1.2,
                label=label,
            )

        ax.set_title(f"q_idx={item['query_index']}", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        if ax_i % int(args.n_cols) == 0:
            ax.set_ylabel("Mean belief")
        if ax_i // int(args.n_cols) == int(args.n_rows) - 1:
            ax.set_xlabel("Depth n")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels, loc="upper right", fontsize=9)

    fig.suptitle(
        "Synthetic fixed-query mean belief across paths "
        f"(J={args.n_paths}, dataset={dataset_name})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])

    out_path = os.path.join(args.save_dir, args.output_name)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"[Saved] {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
