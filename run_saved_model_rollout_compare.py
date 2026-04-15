"""
run_saved_model_rollout_compare.py

Compare fixed-query rollout grids for:
- baseline model (before tuning)
- loaded merged model state (after tuning)

Supports two data sources:
- openml: evaluate synthetic-trained weights on OpenML context/query
- synthetic: evaluate on synthetic task from .h5
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from predictive_rule import ClassifierPredRule
from rollout import belief_at_depth_torch, rollout_one_trajectory_data_only


@dataclass
class RolloutCompareConfig:
    data_source: str = "openml"  # "openml" | "synthetic"

    # OpenML options
    openml_data_id: int = 54
    test_size: float = 0.25

    # Synthetic options
    synthetic_eval_h5: str = "tabicl_mixscm_40_400x5_c3.h5"
    synthetic_task_index: int = -1  # -1 => random by seed
    synthetic_context_pool_size: int = 100
    synthetic_query_pool_size: int = 20

    # Common rollout options
    seed: int = 42
    n_estimators: int = 4
    rollout_base_n: int = 100
    rollout_depth: int = 20
    rollout_n_paths: int = 8
    rollout_n_queries: int = 6
    rollout_grid_rows: int = 2
    rollout_grid_cols: int = 3
    rollout_save_dir: str = "rollout_plots"


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return s if s else "model"


def _extract_model_tag_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d{8}_\d{6})$", stem)
    if m is not None:
        return m.group(1)
    return _slugify(stem)


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
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
    locked_state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> Callable[[], Any]:
    # Reuse a single instance for deterministic parity with existing workflow.
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


def _choose_base_context_indices(y_context_pool: np.ndarray, base_n: int, seed: int) -> np.ndarray:
    y = np.asarray(y_context_pool).astype(int)
    n = len(y)
    m = int(np.clip(base_n, 1, n))
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    picked: list[int] = []
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        if len(idx_c) == 0:
            continue
        picked.append(int(rng.choice(idx_c)))

    picked_arr = np.unique(np.asarray(picked, dtype=int)) if len(picked) > 0 else np.empty(0, dtype=int)
    if len(picked_arr) >= m:
        return picked_arr[:m]

    remain = np.setdiff1d(np.arange(n, dtype=int), picked_arr, assume_unique=False)
    n_more = m - len(picked_arr)
    extra = rng.choice(remain, size=n_more, replace=False)
    return np.concatenate([picked_arr, np.asarray(extra, dtype=int)], axis=0)


def _load_openml_classification_dataset(
    *,
    data_id: int,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool], np.ndarray, str]:
    X, y_raw = fetch_openml(data_id=data_id, as_frame=False, return_X_y=True)
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    categorical_x = [False] * int(X_train.shape[1])
    dataset_name = f"openml_{int(data_id)}"
    return X_train, X_test, y_train, y_test, categorical_x, np.asarray(le.classes_), dataset_name


def _load_synthetic_rollout_pools(
    *,
    h5_path: str,
    seed: int,
    task_index: int,
    context_pool_size: int,
    query_pool_size: int,
    base_n: int,
    n_queries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool], np.ndarray, str]:
    import h5py

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Synthetic eval file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        x_tasks = np.asarray(f["X"], dtype=np.float32)
        y_tasks = np.asarray(f["y"])

    if x_tasks.ndim != 3 or y_tasks.ndim != 2:
        raise ValueError(f"Expected X:(T,N,d), y:(T,N), got X:{x_tasks.shape}, y:{y_tasks.shape}")

    y_tasks = np.rint(y_tasks).astype(np.int64)
    n_tasks, n_points, d = x_tasks.shape

    rng = np.random.default_rng(seed)
    if task_index < 0:
        t_idx = int(rng.integers(0, n_tasks))
    else:
        t_idx = int(np.clip(task_index, 0, n_tasks - 1))

    x_task = np.asarray(x_tasks[t_idx], dtype=np.float32)
    y_task = np.asarray(y_tasks[t_idx], dtype=np.int64)

    if n_points < (base_n + n_queries + 1):
        raise ValueError(
            f"Synthetic task too small: n_points={n_points}, requires at least {base_n + n_queries + 1}."
        )

    n_ctx_pool = int(np.clip(max(context_pool_size, base_n), base_n, n_points - n_queries - 1))
    rem_after_ctx = n_points - n_ctx_pool
    n_q_pool = int(np.clip(max(query_pool_size, n_queries), n_queries, rem_after_ctx - 1))

    perm = np.asarray(rng.permutation(n_points)).astype(int)
    idx_ctx = perm[:n_ctx_pool]
    idx_rest = perm[n_ctx_pool:]
    idx_q = idx_rest[:n_q_pool]
    idx_roll = idx_rest[n_q_pool:]
    if len(idx_roll) == 0:
        idx_roll = idx_q

    x_context_pool = np.asarray(x_task[idx_ctx], dtype=np.float32)
    y_context_pool_raw = np.asarray(y_task[idx_ctx], dtype=np.int64)
    y_context_pool, class_ids = _relabel_to_contiguous(y_context_pool_raw)

    x_query_pool = np.asarray(x_task[idx_q], dtype=np.float32)
    x_sampling_pool = np.asarray(x_task[idx_roll], dtype=np.float32)
    categorical_x = [False] * int(d)

    dataset_name = f"{_slugify(os.path.splitext(os.path.basename(h5_path))[0])}_task{t_idx}"
    class_names = np.asarray([str(int(c)) for c in class_ids])

    print(
        "Synthetic task split: "
        f"task={t_idx}, context_pool={len(x_context_pool)}, "
        f"query_pool={len(x_query_pool)}, rollout_pool={len(x_sampling_pool)}, "
        f"local_classes={class_ids.tolist()}"
    )

    return (
        x_context_pool,
        x_query_pool,
        y_context_pool,
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
    pred_rule_factory: Callable[[], Any],
    base_context_indices: np.ndarray,
    depth: int,
    n_paths: int,
    query_index: int,
    seed: int,
) -> dict[str, Any]:
    x_context_pool = np.asarray(x_context_pool)
    y_context_pool = np.asarray(y_context_pool).astype(np.int64)
    x_query_pool = np.asarray(x_query_pool)
    x_sampling_pool = np.asarray(x_sampling_pool)

    q_idx = int(np.clip(query_index, 0, len(x_query_pool) - 1))
    x_q = np.atleast_2d(x_query_pool[q_idx])

    x0 = np.asarray(x_context_pool[base_context_indices], dtype=np.float32)
    y0 = np.asarray(y_context_pool[base_context_indices], dtype=np.int64)

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
            beliefs_per_depth.append(np.asarray(b.detach().cpu().numpy().squeeze(0), dtype=np.float64))
        beliefs_runs.append(np.stack(beliefs_per_depth, axis=0))

    beliefs_runs_arr = np.stack(beliefs_runs, axis=0)  # (J, depth+1, C)
    mean_beliefs = beliefs_runs_arr.mean(axis=0)
    std_beliefs = beliefs_runs_arr.std(axis=0, ddof=1) if n_paths > 1 else np.zeros_like(mean_beliefs)
    max_abs_sum_error = float(np.max(np.abs(beliefs_runs_arr.sum(axis=2) - 1.0)))

    return {
        "q_idx": q_idx,
        "xs": np.arange(int(depth) + 1),
        "beliefs_runs": beliefs_runs_arr,
        "mean_beliefs": mean_beliefs,
        "std_beliefs": std_beliefs,
        "max_abs_sum_error": max_abs_sum_error,
    }


def _run_fixed_queries_rollout_grid_analysis(
    *,
    x_context_pool: np.ndarray,
    y_context_pool: np.ndarray,
    x_query_pool: np.ndarray,
    x_sampling_pool: np.ndarray,
    query_indices: Sequence[int],
    class_names: Optional[Sequence[Any]],
    pred_rule_factory: Callable[[], Any],
    base_n: int,
    depth: int,
    n_paths: int,
    seed: int,
    n_rows: int,
    n_cols: int,
    save_dir: str,
    tag: str,
    show: bool,
) -> dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    q_list = [int(q) for q in query_indices]
    if len(q_list) == 0:
        raise ValueError("query_indices must not be empty.")
    if len(q_list) > int(n_rows) * int(n_cols):
        raise ValueError("Too many query indices for requested grid.")

    base_indices = _choose_base_context_indices(y_context_pool, int(base_n), int(seed))
    stats_list = []
    for i, q_idx in enumerate(q_list):
        stats = _compute_fixed_query_rollout_stats(
            x_context_pool=x_context_pool,
            y_context_pool=y_context_pool,
            x_query_pool=x_query_pool,
            x_sampling_pool=x_sampling_pool,
            pred_rule_factory=pred_rule_factory,
            base_context_indices=base_indices,
            depth=int(depth),
            n_paths=int(n_paths),
            query_index=q_idx,
            seed=int(seed + i * 1000),
        )
        print(
            f"  [rollout-grid] q_idx={q_idx}, max |sum-1|={float(stats['max_abs_sum_error']):.8f}"
        )
        stats_list.append(stats)

    fig, axes = plt.subplots(
        int(n_rows),
        int(n_cols),
        figsize=(int(n_cols) * 5.0, int(n_rows) * 4.0),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(stats_list):
            ax.axis("off")
            continue

        stats = stats_list[ax_i]
        xs = stats["xs"]
        mean_beliefs = stats["mean_beliefs"]
        q_idx = int(stats["q_idx"])

        for c in range(mean_beliefs.shape[1]):
            label = None
            if ax_i == 0:
                cname = str(class_names[c]) if class_names is not None and c < len(class_names) else str(c)
                label = f"class {c} ({cname})"
            ax.plot(xs, mean_beliefs[:, c], marker="o", markersize=2.2, linewidth=1.2, label=label)

        ax.set_title(f"q_idx={q_idx}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if ax_i % int(n_cols) == 0:
            ax.set_ylabel(r"$\hat{\mathbb{E}}[\theta_n(x_q)]$")
        if ax_i // int(n_cols) == (int(n_rows) - 1):
            ax.set_xlabel("Depth n")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    title_tag = f", tag={tag}" if tag else ""
    fig.suptitle(f"Fixed-query mean belief across paths (J={n_paths}{title_tag})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])

    out_path = os.path.join(save_dir, f"fixed_queries_mean_belief_grid_{tag}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "query_indices": q_list,
        "paths": {"grid_plot_png": out_path},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to merged model .pt")
    parser.add_argument("--data-source", type=str, default="openml", choices=["openml", "synthetic"])

    parser.add_argument("--openml-data-id", type=int, default=54)
    parser.add_argument("--test-size", type=float, default=0.25)

    parser.add_argument("--synthetic-eval-h5", type=str, default="tabicl_mixscm_40_400x5_c3.h5")
    parser.add_argument("--synthetic-task-index", type=int, default=-1)
    parser.add_argument("--synthetic-context-pool-size", type=int, default=100)
    parser.add_argument("--synthetic-query-pool-size", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--rollout-base-n", type=int, default=100)
    parser.add_argument("--rollout-depth", type=int, default=20)
    parser.add_argument("--rollout-n-paths", type=int, default=8)
    parser.add_argument("--rollout-n-queries", type=int, default=6)
    parser.add_argument("--rollout-grid-rows", type=int, default=2)
    parser.add_argument("--rollout-grid-cols", type=int, default=3)
    parser.add_argument("--rollout-save-dir", type=str, default="rollout_plots")
    parser.add_argument("--tag", type=str, default="", help="Optional extra tag")
    parser.add_argument(
        "--skip-before",
        action="store_true",
        help="Skip baseline(before) rollout plot generation and only save after plot.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RolloutCompareConfig(
        data_source=args.data_source,
        openml_data_id=args.openml_data_id,
        test_size=args.test_size,
        synthetic_eval_h5=args.synthetic_eval_h5,
        synthetic_task_index=args.synthetic_task_index,
        synthetic_context_pool_size=args.synthetic_context_pool_size,
        synthetic_query_pool_size=args.synthetic_query_pool_size,
        seed=args.seed,
        n_estimators=args.n_estimators,
        rollout_base_n=args.rollout_base_n,
        rollout_depth=args.rollout_depth,
        rollout_n_paths=args.rollout_n_paths,
        rollout_n_queries=args.rollout_n_queries,
        rollout_grid_rows=args.rollout_grid_rows,
        rollout_grid_cols=args.rollout_grid_cols,
        rollout_save_dir=args.rollout_save_dir,
    )

    set_global_seeds(cfg.seed)

    if cfg.data_source == "openml":
        X_train, X_test, y_train, _, categorical_x, class_names, dataset_name = _load_openml_classification_dataset(
            data_id=cfg.openml_data_id,
            seed=cfg.seed,
            test_size=cfg.test_size,
        )
        x_context_pool = X_train
        y_context_pool = y_train
        x_query_pool = X_test
        x_sampling_pool = X_train
    else:
        (
            x_context_pool,
            x_query_pool,
            y_context_pool,
            x_sampling_pool,
            categorical_x,
            class_names,
            dataset_name,
        ) = _load_synthetic_rollout_pools(
            h5_path=cfg.synthetic_eval_h5,
            seed=cfg.seed,
            task_index=cfg.synthetic_task_index,
            context_pool_size=cfg.synthetic_context_pool_size,
            query_pool_size=cfg.synthetic_query_pool_size,
            base_n=cfg.rollout_base_n,
            n_queries=cfg.rollout_n_queries,
        )

    rng = np.random.default_rng(cfg.seed)
    replace = len(x_query_pool) < cfg.rollout_n_queries
    rollout_query_indices = np.asarray(
        rng.choice(len(x_query_pool), size=cfg.rollout_n_queries, replace=replace),
        dtype=int,
    )
    print(f"Data source: {cfg.data_source}")
    print(f"Dataset: {dataset_name}")
    print(f"Fixed rollout queries: {rollout_query_indices.tolist()}")

    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    loaded_state = _extract_state_dict(ckpt)
    print(f"Loaded model state: {model_path} ({len(loaded_state)} tensors)")

    dataset_tag = _slugify(dataset_name)
    model_tag = _extract_model_tag_from_filename(model_path)
    if args.tag.strip():
        model_tag = f"{model_tag}_{_slugify(args.tag)}"

    after_factory = _make_pred_rule_factory(
        categorical_x=categorical_x,
        n_estimators=cfg.n_estimators,
        locked_state_dict=loaded_state,
    )

    os.makedirs(cfg.rollout_save_dir, exist_ok=True)

    before_res = None
    if not args.skip_before:
        before_factory = _make_pred_rule_factory(
            categorical_x=categorical_x,
            n_estimators=cfg.n_estimators,
            locked_state_dict=None,
        )
        print("\nGenerating rollout grid: baseline (before)...")
        before_res = _run_fixed_queries_rollout_grid_analysis(
            x_context_pool=x_context_pool,
            y_context_pool=y_context_pool,
            x_query_pool=x_query_pool,
            x_sampling_pool=x_sampling_pool,
            query_indices=rollout_query_indices.tolist(),
            class_names=class_names,
            pred_rule_factory=before_factory,
            base_n=cfg.rollout_base_n,
            depth=cfg.rollout_depth,
            n_paths=cfg.rollout_n_paths,
            seed=cfg.seed,
            n_rows=cfg.rollout_grid_rows,
            n_cols=cfg.rollout_grid_cols,
            save_dir=cfg.rollout_save_dir,
            tag=f"{dataset_tag}_before_{model_tag}",
            show=args.show,
        )
        before_src = before_res["paths"]["grid_plot_png"]
        before_dst = os.path.join(cfg.rollout_save_dir, f"belief_{dataset_tag}_before_{model_tag}.png")
        if os.path.abspath(before_src) != os.path.abspath(before_dst):
            os.replace(before_src, before_dst)
            before_res["paths"]["grid_plot_png"] = before_dst

    print("\nGenerating rollout grid: loaded model (after)...")
    after_res = _run_fixed_queries_rollout_grid_analysis(
        x_context_pool=x_context_pool,
        y_context_pool=y_context_pool,
        x_query_pool=x_query_pool,
        x_sampling_pool=x_sampling_pool,
        query_indices=rollout_query_indices.tolist(),
        class_names=class_names,
        pred_rule_factory=after_factory,
        base_n=cfg.rollout_base_n,
        depth=cfg.rollout_depth,
        n_paths=cfg.rollout_n_paths,
        seed=cfg.seed,
        n_rows=cfg.rollout_grid_rows,
        n_cols=cfg.rollout_grid_cols,
        save_dir=cfg.rollout_save_dir,
        tag=f"{dataset_tag}_after",
        show=args.show,
    )
    after_src = after_res["paths"]["grid_plot_png"]
    after_dst = os.path.join(cfg.rollout_save_dir, f"belief_{dataset_tag}_after_{model_tag}.png")
    if os.path.abspath(after_src) != os.path.abspath(after_dst):
        os.replace(after_src, after_dst)
        after_res["paths"]["grid_plot_png"] = after_dst

    print("\nDone.")
    if before_res is not None:
        print(f"  Before plot: {before_res['paths']['grid_plot_png']}")
    else:
        print("  Before plot: skipped (--skip-before)")
    print(f"  After  plot: {after_res['paths']['grid_plot_png']}")


if __name__ == "__main__":
    main()
