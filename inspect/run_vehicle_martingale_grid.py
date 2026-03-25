"""
run_vehicle_martingale_grid.py
------------------------------------------------------------
TabPFN martingale 성질 시각 점검용 스크립트.

- OpenML Vehicle test set에서 query 24개 샘플
- query별 fixed-query mean belief across paths 계산
- 기본 12개(3x4) query를 1장 이미지로 저장
"""

import argparse
import importlib.util
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Make project-root modules importable when this script is run from inspect/.
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predictive_rule import ClassifierPredRule


def _load_rollout_analyzer_module() -> Any:
    module_path = os.path.join(THIS_DIR, "analyze_fixed_query_rollout_mc.py")
    spec = importlib.util.spec_from_file_location("analyze_fixed_query_rollout_mc", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load analyzer module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vehicle_dataset(seed: int = 42):
    """Load OpenML Vehicle dataset and split train/test."""
    data = fetch_openml(data_id=54, as_frame=False, parser="auto")
    X, y_raw = data.data, data.target

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    categorical_x = [False] * X.shape[1]
    return X_train, X_test, y_train, y_test, categorical_x, le


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--query-count", type=int, default=12, help="Number of test queries")
    parser.add_argument("--n-rows", type=int, default=3, help="Rows per figure")
    parser.add_argument("--n-cols", type=int, default=4, help="Columns per figure")
    parser.add_argument("--n-paths", type=int, default=8, help="Number of independent paths")
    parser.add_argument("--depth", type=int, default=30, help="Rollout depth")
    parser.add_argument("--base-n", type=int, default=100, help="Base context size")
    parser.add_argument("--n-estimators", type=int, default=2, help="TabPFN n_estimators")
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional merged model .pt path. If set, rollout uses this locked state.",
    )
    parser.add_argument(
        "--save-dir", type=str, default="martingale_plots", help="Directory to save plot grids"
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    return parser.parse_args()


def _extract_state_dict(ckpt: Any) -> dict:
    """Load a plain tensor state_dict from common checkpoint formats."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        raw = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        raw = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")
    state: dict = {}
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
    locked_state_dict: dict | None = None,
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


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    analyzer = _load_rollout_analyzer_module()
    X_train, X_test, y_train, _, categorical_x, le = load_vehicle_dataset(seed=args.seed)

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
        print("Model path not provided: using baseline (untrained) predictive rule.")

    rng = np.random.default_rng(args.seed)
    replace = len(X_test) < args.query_count
    query_indices = np.array(
        rng.choice(len(X_test), size=args.query_count, replace=replace),
        dtype=int,
    )
    print(f"Selected {len(query_indices)} queries from test set: {query_indices.tolist()}")

    results = []
    for rank, q_idx in enumerate(query_indices):
        stats = analyzer.compute_fixed_query_rollout_stats(
            x_train=X_train,
            y_train=y_train,
            x_query_pool=X_test,
            categorical_x=categorical_x,
            pred_rule_factory=pred_rule_factory,
            n_estimators=args.n_estimators,
            base_n=args.base_n,
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
            }
        )
        print(
            f"[{rank+1:2d}/{len(query_indices)}] q_idx={int(q_idx)} "
            f"shape={tuple(stats['mean_beliefs'].shape)}"
        )

    plots_per_figure = args.n_rows * args.n_cols
    n_figures = int(np.ceil(len(results) / plots_per_figure))

    for fig_i in range(n_figures):
        start = fig_i * plots_per_figure
        end = min(len(results), start + plots_per_figure)
        chunk = results[start:end]

        fig, axes = plt.subplots(
            args.n_rows,
            args.n_cols,
            figsize=(args.n_cols * 5.0, args.n_rows * 3.7),
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes).reshape(-1)

        for ax_i, ax in enumerate(axes):
            if ax_i >= len(chunk):
                ax.axis("off")
                continue

            item = chunk[ax_i]
            xs = item["xs"]
            mean_beliefs = item["mean_beliefs"]
            n_classes = mean_beliefs.shape[1]

            for c in range(n_classes):
                label = None
                if fig_i == 0 and ax_i == 0:
                    label = f"class {c} ({str(le.classes_[c])})"
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
            if ax_i % args.n_cols == 0:
                ax.set_ylabel("Mean belief")
            if ax_i // args.n_cols == args.n_rows - 1:
                ax.set_xlabel("Depth n")

        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper right", fontsize=9)

        fig.suptitle(
            "Vehicle fixed-query mean belief across paths "
            f"({start+1}-{end}/{len(results)})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 0.98, 0.95])

        out_path = os.path.join(
            args.save_dir,
            f"vehicle_fixed_query_mean_belief_grid_{fig_i+1}_of_{n_figures}.png",
        )
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        print(f"[Saved] {out_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
