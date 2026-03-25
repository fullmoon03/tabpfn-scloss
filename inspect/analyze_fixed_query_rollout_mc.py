"""
analyze_fixed_query_rollout_mc.py
------------------------------------------------------------
Plot fixed-query rollout with independent resampling paths.

Requested logic:
  - Fix a query x_q
  - For depths n=0..N, record theta_n^(j)(x_q) on each path j=1..J
  - Average over paths: E_hat[theta_n(x_q)] = (1/J) * sum_j theta_n^(j)(x_q)
"""

import argparse
import os
from typing import Callable, Optional, Sequence, Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from predictive_rule import make_predictive_rule
from rollout import rollout_one_trajectory


SAVE_DIR = "rollout_plots"


def _tagged_name(stem: str, ext: str, tag: str) -> str:
    tag = str(tag).strip()
    return f"{stem}.{ext}" if tag == "" else f"{stem}_{tag}.{ext}"


def compute_fixed_query_rollout_stats(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query_pool: np.ndarray,
    categorical_x: list[bool],
    pred_rule_factory: Optional[Callable[[], Any]] = None,
    n_estimators: int = 2,
    base_n: int = 50,
    depth: int = 30,
    n_paths: int = 8,
    query_index: int = 0,
    seed: int = 0,
) -> dict:
    """Run fixed-query rollout MC and return raw/aggregated belief statistics."""
    np.random.seed(seed)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train).astype(np.int64)
    x_query_pool = np.asarray(x_query_pool)

    n0 = int(np.clip(base_n, 1, len(x_train)))
    x0 = x_train[:n0]
    y0 = y_train[:n0]
    q_idx = int(np.clip(query_index, 0, len(x_query_pool) - 1))
    x_q = x_query_pool[q_idx : q_idx + 1]
    xs = np.arange(depth + 1)

    if pred_rule_factory is None:
        def pred_rule_factory():
            return make_predictive_rule(
                task_type="classification",
                categorical_x=categorical_x,
                n_estimators=n_estimators,
                average_before_softmax=False,
            )

    beliefs_runs = []
    for j in range(n_paths):
        key = jax.random.PRNGKey(seed + j)
        pred_rule = pred_rule_factory()
        traj = rollout_one_trajectory(
            key=key,
            pred_rule=pred_rule,
            x0=x0,
            y0=y0,
            depth=depth,
            x_q=x_q,
        )
        beliefs_j = np.asarray(traj.beliefs, dtype=np.float64)  # (N+1, C)
        beliefs_runs.append(beliefs_j)

    beliefs_runs = np.stack(beliefs_runs, axis=0)  # (J, N+1, C)
    mean_beliefs = beliefs_runs.mean(axis=0)  # (N+1, C)
    std_beliefs = beliefs_runs.std(axis=0, ddof=1)  # (N+1, C), unbiased (sample std)

    sums_all = beliefs_runs.sum(axis=2)  # (J, N+1)
    max_abs_err = float(np.max(np.abs(sums_all - 1.0)))

    return {
        "xs": xs,
        "q_idx": q_idx,
        "beliefs_runs": beliefs_runs,
        "mean_beliefs": mean_beliefs,
        "std_beliefs": std_beliefs,
        "max_abs_sum_error": max_abs_err,
    }


def run_fixed_query_rollout_analysis(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query_pool: np.ndarray,
    categorical_x: list[bool],
    class_names: Optional[Sequence[Any]] = None,
    pred_rule_factory: Optional[Callable[[], Any]] = None,
    n_estimators: int = 2,
    base_n: int = 50,
    depth: int = 30,
    n_paths: int = 8,
    query_index: int = 0,
    seed: int = 0,
    save_dir: str = SAVE_DIR,
    tag: str = "",
    show: bool = False,
) -> dict:
    """
    Fixed-query rollout 분석을 함수 형태로 실행 (run_classification.py에서 재사용).

    Returns:
        dict with summary stats and saved paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    stats = compute_fixed_query_rollout_stats(
        x_train=x_train,
        y_train=y_train,
        x_query_pool=x_query_pool,
        categorical_x=categorical_x,
        pred_rule_factory=pred_rule_factory,
        n_estimators=n_estimators,
        base_n=base_n,
        depth=depth,
        n_paths=n_paths,
        query_index=query_index,
        seed=seed,
    )
    xs = stats["xs"]
    q_idx = int(stats["q_idx"])
    beliefs_runs = stats["beliefs_runs"]
    mean_beliefs = stats["mean_beliefs"]
    std_beliefs = stats["std_beliefs"]
    max_abs_err = float(stats["max_abs_sum_error"])
    print("\n[A] Sum-to-one check for all paths/depths")
    print(f"    shape(beliefs_runs): {beliefs_runs.shape}  # (J, N+1, C)")
    print(f"    max |sum-1|: {max_abs_err:.8f}")
    if max_abs_err > 1e-4:
        print("    WARNING: normalization error larger than expected.")
    else:
        print("    OK: all beliefs are normalized.")

    plt.figure(figsize=(10, 5))
    c_count = mean_beliefs.shape[1]
    for c in range(c_count):
        cname = str(class_names[c]) if class_names is not None and c < len(class_names) else str(c)
        (line,) = plt.plot(
            xs,
            mean_beliefs[:, c],
            marker="o",
            linewidth=1.6,
            label=f"class {c} ({cname})",
        )
        color = line.get_color()
        lower = np.clip(mean_beliefs[:, c] - std_beliefs[:, c], 0.0, 1.0)
        upper = np.clip(mean_beliefs[:, c] + std_beliefs[:, c], 0.0, 1.0)
        plt.fill_between(xs, lower, upper, color=color, alpha=0.15)
    plt.xlabel("Depth n")
    plt.ylabel(r"$\hat{\mathbb{E}}[\theta_n(x_q)]$")
    title_tag = f", tag={tag}" if tag else ""
    plt.title(f"Fixed query mean belief across paths (J={n_paths}, q_idx={q_idx}{title_tag})")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(save_dir, _tagged_name("mean_belief_all_classes", "png", tag))
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    print(f"[Saved] {p1}")
    if show:
        plt.show()
    else:
        plt.close()

    target_class = int(np.argmax(mean_beliefs[0]))
    target_name = str(class_names[target_class]) if class_names is not None and target_class < len(class_names) else str(target_class)

    plt.figure(figsize=(10, 5))
    for j in range(n_paths):
        plt.plot(
            xs,
            beliefs_runs[j, :, target_class],
            marker="o",
            linewidth=1.0,
            alpha=0.55,
            label=f"path {j}",
        )
    plt.plot(
        xs,
        mean_beliefs[:, target_class],
        color="black",
        linewidth=3.0,
        label="mean over paths",
    )
    plt.fill_between(
        xs,
        mean_beliefs[:, target_class] - std_beliefs[:, target_class],
        mean_beliefs[:, target_class] + std_beliefs[:, target_class],
        color="black",
        alpha=0.15,
        label="±1 std",
    )
    plt.xlabel("Depth n")
    plt.ylabel(rf"$\theta_n(x_q)[\mathrm{{class}}\ {target_class}]$")
    plt.title(f"Independent resampling paths (class {target_class} = {target_name}{title_tag})")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    p2 = os.path.join(save_dir, _tagged_name("stochasticity_overlay_with_mean", "png", tag))
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    print(f"[Saved] {p2}")
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "max_abs_sum_error": max_abs_err,
        "beliefs_runs_shape": tuple(beliefs_runs.shape),
        "paths": {
            "mean_plot_png": p1,
            "overlay_plot_png": p2,
        },
    }


def run_fixed_queries_rollout_grid_analysis(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query_pool: np.ndarray,
    query_indices: Sequence[int],
    categorical_x: list[bool],
    class_names: Optional[Sequence[Any]] = None,
    pred_rule_factory: Optional[Callable[[], Any]] = None,
    n_estimators: int = 2,
    base_n: int = 50,
    depth: int = 30,
    n_paths: int = 8,
    seed: int = 0,
    n_rows: int = 2,
    n_cols: int = 3,
    save_dir: str = SAVE_DIR,
    tag: str = "",
    show: bool = False,
) -> dict:
    """
    여러 fixed query(예: 6개)에 대해 mean belief 곡선을 grid(2x3)로 저장.
    """
    os.makedirs(save_dir, exist_ok=True)
    q_list = [int(q) for q in query_indices]
    if len(q_list) == 0:
        raise ValueError("query_indices must not be empty.")
    if len(q_list) > n_rows * n_cols:
        raise ValueError(
            f"Too many queries for grid: len(query_indices)={len(q_list)} > "
            f"n_rows*n_cols={n_rows*n_cols}"
        )

    stats_list = []
    for i, q_idx in enumerate(q_list):
        stats = compute_fixed_query_rollout_stats(
            x_train=x_train,
            y_train=y_train,
            x_query_pool=x_query_pool,
            categorical_x=categorical_x,
            pred_rule_factory=pred_rule_factory,
            n_estimators=n_estimators,
            base_n=base_n,
            depth=depth,
            n_paths=n_paths,
            query_index=q_idx,
            seed=seed + i * 1000,
        )
        print(
            f"  [rollout-grid] q_idx={q_idx}, "
            f"max |sum-1|={float(stats['max_abs_sum_error']):.8f}"
        )
        stats_list.append(stats)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.0, n_rows * 4.0),
        sharex=True, sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(stats_list):
            ax.axis("off")
            continue
        stats = stats_list[ax_i]
        q_idx = int(stats["q_idx"])
        xs = stats["xs"]
        mean_beliefs = stats["mean_beliefs"]  # (N+1, C)
        c_count = mean_beliefs.shape[1]

        for c in range(c_count):
            label = None
            if ax_i == 0:
                cname = (
                    str(class_names[c])
                    if class_names is not None and c < len(class_names)
                    else str(c)
                )
                label = f"class {c} ({cname})"
            ax.plot(
                xs, mean_beliefs[:, c],
                marker="o", markersize=2.2, linewidth=1.2, label=label,
            )

        ax.set_title(f"q_idx={q_idx}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if ax_i % n_cols == 0:
            ax.set_ylabel(r"$\hat{\mathbb{E}}[\theta_n(x_q)]$")
        if ax_i // n_cols == (n_rows - 1):
            ax.set_xlabel("Depth n")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    title_tag = f", tag={tag}" if tag else ""
    fig.suptitle(
        f"Fixed-query mean belief across paths (J={n_paths}{title_tag})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])

    out_path = os.path.join(save_dir, _tagged_name("fixed_queries_mean_belief_grid", "png", tag))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "query_indices": q_list,
        "paths": {
            "grid_plot_png": out_path,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-n", type=int, default=50, help="Base context size N0")
    parser.add_argument("--depth", type=int, default=30, help="Rollout depth N")
    parser.add_argument("--n-paths", type=int, default=8, help="Number of independent paths J")
    parser.add_argument("--query-index", type=int, default=0, help="Fixed query index in X_test")
    parser.add_argument("--n-estimators", type=int, default=2, help="TabPFN n_estimators")
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.random.seed(args.seed)

    # 1) Dataset
    X, y_raw = fetch_openml(data_id=54, as_frame=False, return_X_y=True)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.int64)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Classes: {list(le.classes_)}")

    d = X_train.shape[1]
    categorical_x = [False] * d
    run_fixed_query_rollout_analysis(
        x_train=X_train,
        y_train=y_train,
        x_query_pool=X_test,
        categorical_x=categorical_x,
        class_names=le.classes_,
        n_estimators=args.n_estimators,
        base_n=args.base_n,
        depth=args.depth,
        n_paths=args.n_paths,
        query_index=args.query_index,
        seed=args.seed,
        save_dir=SAVE_DIR,
        tag="",
        show=args.show,
    )


if __name__ == "__main__":
    main()
