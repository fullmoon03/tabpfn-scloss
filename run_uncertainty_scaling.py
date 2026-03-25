"""
run_uncertainty_scaling.py
------------------------------------------------------------
Context size N0를 키울 때 epistemic uncertainty가 감소하는지 측정.

핵심 실험:
- rollout 없이(n=0) 고정 query x_q에서 예측분포 p^(r)(x_q; N0) 수집
- replicate r마다 train subset D0^(r)만 바꿔서 예측 분포 변동성 측정
- 지표: JS disagreement (BALD MI)
  U_JS(N0; x_q) = H(mean_r p^(r)) - mean_r H(p^(r))
- query-level U_JS를 얻은 뒤 median over queries로 집계

옵션:
- baseline 모델만 실행 (기본)
- --tuned-merged-state로 학습된 merged state_dict를 주면 baseline vs tuned 비교
"""

"""
(Real -> Real)
python run_uncertainty_scaling.py \
  --tuned-merged-state /home/boreum/project/tabpfn-scloss/openml_54/vehicle_model/merged_model_state_openml_54_20260305_152612.pt
"""

"""
(Synthetic -> Real)
python run_uncertainty_scaling.py \
  --tuned-merged-state /home/boreum/project/tabpfn-scloss/synthetic_model/merged_model_state_generated_train_800_seed42_20260316_053443.pt
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

from predictive_rule import ClassifierPredRule
from run_openml_classification import load_openml_dataset_split


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
    """
    Args:
        probs_rqc: (R, Q, C)

    Returns:
        U_JS per query: (Q,)
    """
    p = np.asarray(probs_rqc, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / np.clip(p.sum(axis=-1, keepdims=True), eps, None)

    mean_p = p.mean(axis=0)  # (Q, C)
    h_mean = _entropy_probs(mean_p, eps=eps)  # (Q,)
    h_each = _entropy_probs(p, eps=eps)  # (R, Q)
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
    """
    Return shape (R, N0) index matrix.
    seed를 n0별로 분리해 loop 순서가 바뀌어도 동일 subset이 나오도록 함.
    """
    rng = np.random.default_rng(seed + 1009 * int(n0))
    subsets = []
    for _ in range(replicates):
        idx = rng.choice(n_train, size=n0, replace=False)
        subsets.append(np.asarray(idx, dtype=int))
    return np.stack(subsets, axis=0)


def collect_probs_for_model(
    *,
    model_spec: ModelSpec,
    categorical_x: list[bool],
    n_estimators: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_queries: np.ndarray,
    subsets_rn: np.ndarray,
) -> np.ndarray:
    """
    For one model and one N0:
    - each replicate subset으로 fit
    - fixed query bank(Q개)에 대한 probs를 한 번에 예측

    Returns:
        probs_rqc: (R, Q, C)
    """
    # Global class axis for cross-replicate comparability.
    # y_train is label-encoded as 0..C-1 in load_openml_dataset_split.
    n_classes_global = int(np.max(y_train)) + 1
    probs_runs = []
    for idx in subsets_rn:
        pred_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
        if model_spec.locked_state is not None:
            # PredictiveRule.fit() 이후 locked state를 restore하도록 훅 사용.
            pred_rule._locked_state_dict = {
                k: v.clone().cpu() for k, v in model_spec.locked_state.items()
            }

        x_ctx = x_train[idx]
        y_ctx = y_train[idx]

        # Some subsets may miss classes (e.g., [0,1,3]).
        # Fit with local contiguous labels, then map probabilities back
        # to the global 0..C-1 axis.
        local_classes, y_local = np.unique(y_ctx.astype(int), return_inverse=True)
        pred_rule.fit(x_ctx, y_local.astype(np.int64))
        probs_local = np.asarray(pred_rule.get_belief(x_queries), dtype=np.float64)  # (Q, C_local)

        probs_global = np.zeros((probs_local.shape[0], n_classes_global), dtype=np.float64)
        probs_global[:, local_classes] = probs_local
        probs_runs.append(probs_global)

    return np.stack(probs_runs, axis=0)  # (R, Q, C)


def summarize_u(u_q: np.ndarray) -> dict[str, float]:
    uq = np.asarray(u_q, dtype=np.float64)
    return {
        "u_js_median": float(np.median(uq)),
        "u_js_mean": float(np.mean(uq)),
        "u_js_q25": float(np.percentile(uq, 25.0)),
        "u_js_q75": float(np.percentile(uq, 75.0)),
    }


def fit_loglog_alpha(ns: np.ndarray, us: np.ndarray, eps: float, n_min: int, n_max: int) -> dict[str, float]:
    mask = (ns >= n_min) & (ns <= n_max)
    if int(mask.sum()) < 2:
        return {
            "n_min": float(n_min),
            "n_max": float(n_max),
            "alpha": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "n_points": float(mask.sum()),
        }

    x = np.log(ns[mask].astype(np.float64))
    y = np.log(us[mask].astype(np.float64) + eps)
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n_min": float(n_min),
        "n_max": float(n_max),
        "alpha": float(-slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_points": float(mask.sum()),
    }


def save_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--openml-data-id", type=int, default=54)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--query-count", type=int, default=24)
    p.add_argument("--replicates", type=int, default=50)
    p.add_argument("--n0-grid", type=str, default="16,32,64,128,256,512")
    p.add_argument("--include-full-train", action="store_true")
    p.add_argument("--n-estimators", type=int, default=1)
    p.add_argument("--tuned-merged-state", type=str, default="")
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--output-dir", type=str, default="uncertainty_scaling")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    X_train, X_test, y_train, y_test, categorical_x, le, dataset_name = (
        load_openml_dataset_split(
            data_id=args.openml_data_id,
            seed=args.seed,
            test_size=args.test_size,
        )
    )
    _ = y_test, le

    dataset_tag = _slugify(dataset_name)
    n_train = len(X_train)
    n0_grid = _parse_n0_grid(args.n0_grid, n_train=n_train, include_full_train=args.include_full_train)

    query_indices = sample_query_bank_indices(
        n_test=len(X_test), query_count=args.query_count, seed=args.seed
    )
    x_query_bank = np.asarray(X_test)[query_indices]

    model_specs = [ModelSpec(name="baseline", locked_state=None)]
    if args.tuned_merged_state.strip() != "":
        tuned_state = load_locked_state(args.tuned_merged_state.strip())
        model_specs.append(ModelSpec(name="tuned", locked_state=tuned_state))

    print("\n── Uncertainty Scaling Experiment (n=0; no rollout) ──")
    print(f"  Dataset: {dataset_name} (OpenML ID={args.openml_data_id})")
    print(f"  Train/Test: {len(X_train)}/{len(X_test)}")
    print(f"  Models: {[m.name for m in model_specs]}")
    print(f"  N0 grid: {n0_grid}")
    print(f"  Replicates R: {args.replicates}")
    print(f"  Query count Q: {len(query_indices)}")
    print(f"  Query indices in X_test: {query_indices.tolist()}")

    # Precompute replicate subsets for each N0 (shared across models).
    subset_bank = {
        int(n0): sample_replicate_subsets(
            n_train=len(X_train),
            n0=int(n0),
            replicates=args.replicates,
            seed=args.seed,
        )
        for n0 in n0_grid
    }

    summary_rows: list[dict] = []

    for model_spec in model_specs:
        print(f"\n[model={model_spec.name}] collecting probabilities...")
        for n0 in n0_grid:
            subsets = subset_bank[int(n0)]  # (R, N0)
            probs_rqc = collect_probs_for_model(
                model_spec=model_spec,
                categorical_x=categorical_x,
                n_estimators=args.n_estimators,
                x_train=X_train,
                y_train=y_train,
                x_queries=x_query_bank,
                subsets_rn=subsets,
            )
            u_q = js_disagreement_from_replicates(probs_rqc, eps=args.eps)
            s = summarize_u(u_q)
            summary_rows.append(
                {
                    "model": model_spec.name,
                    "n0": int(n0),
                    **s,
                }
            )

            print(
                f"  N0={int(n0):4d} -> "
                f"U_JS median={s['u_js_median']:.6e}, "
                f"mean={s['u_js_mean']:.6e}"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_csv = os.path.join(
        args.output_dir, f"ujs_summary_{dataset_tag}_{ts}.csv"
    )
    save_csv(
        summary_csv,
        summary_rows,
        ["model", "n0", "u_js_median", "u_js_mean", "u_js_q25", "u_js_q75"],
    )

    # Save one png with two panels:
    # (A) log-log U_JS vs N0, (B) compensated N0*U_JS vs N0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    baseline_ns: Optional[np.ndarray] = None
    baseline_med: Optional[np.ndarray] = None
    for model_spec in model_specs:
        model_name = model_spec.name
        rows = [r for r in summary_rows if r["model"] == model_name]
        rows = sorted(rows, key=lambda r: int(r["n0"]))
        ns = np.array([int(r["n0"]) for r in rows], dtype=np.int64)
        med = np.array([float(r["u_js_median"]) for r in rows], dtype=np.float64)
        q25 = np.array([float(r["u_js_q25"]) for r in rows], dtype=np.float64)
        q75 = np.array([float(r["u_js_q75"]) for r in rows], dtype=np.float64)

        axes[0].plot(ns, med, marker="o", linewidth=1.8, label=model_name)
        axes[0].fill_between(ns, q25, q75, alpha=0.15)
        axes[1].plot(ns, ns * med, marker="o", linewidth=1.8, label=model_name)
        if model_name == "baseline":
            baseline_ns = ns
            baseline_med = med

    # Add slope=-1 reference line on panel (A), anchored at baseline near N_ref=128.
    if baseline_ns is not None and baseline_med is not None and len(baseline_ns) > 0:
        n_ref_target = 128
        ref_i = int(np.argmin(np.abs(baseline_ns - n_ref_target)))
        n_ref = float(baseline_ns[ref_i])
        u_ref = float(max(baseline_med[ref_i], args.eps))
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
    axes[0].set_ylabel("U_JS(N0) = median over queries")
    axes[0].set_title("(A) Log-log: U_JS vs N0")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlabel("N0 (context size)")
    axes[1].set_ylabel("N0 * U_JS(N0)")
    axes[1].set_title("(B) Compensated: N0 * U_JS")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"Epistemic uncertainty scaling | dataset={dataset_name} | Q={len(query_indices)} | R={args.replicates}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plot_path = os.path.join(
        args.output_dir, f"ujs_scaling_plot_{dataset_tag}_{ts}.png"
    )
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {plot_path}")


if __name__ == "__main__":
    main()
