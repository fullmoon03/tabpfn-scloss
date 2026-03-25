"""
run_classification.py
─────────────────────────────────────────────
Classification 실험: OpenML Vehicle dataset.

Pipeline:
  1. 데이터 로드 → LabelEncoder → train/test split
  2. 학습 전 baseline eval
  3. SC + CE training (train_and_merge)
  4. 학습 후 eval
  5. Before/After 비교 + SC metric + drift curve
"""

import time
import numpy as np
import jax
import torch

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from predictive_rule import ClassifierPredRule
from rollout import (
    build_prefix_batch,
    build_prefix_batch_data,
    sample_horizon_pair,
)
from train import TrainConfig, train_and_merge
from eval import (
    evaluate_basic,
    compute_sc_metric,
    compute_drift_curve,
    EvalComparison,
    print_comparison,
)


def set_global_seeds(seed: int) -> None:
    """Seed numpy/torch for reproducible experiment runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vehicle_dataset(subsample: int = 600, seed: int = 0):
    print("── Loading Vehicle dataset (OpenML #54) ──")
    data = fetch_openml(data_id=54, as_frame=False, parser="auto")
    X, y_raw = data.data, data.target

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Classes: {le.classes_} → {list(range(len(le.classes_)))}")
    print(f"  Full dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")

    if subsample and subsample < len(X):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=subsample, replace=False)
        X, y = X[idx], y[idx]
        print(f"  Subsampled to {subsample} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y,
    )
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")

    categorical_x = [False] * X.shape[1]
    return X_train, X_test, y_train, y_test, categorical_x, le


def run_experiment(
    X_train, X_test, y_train, y_test, categorical_x,
    config: TrainConfig,
    key=None,
):
    set_global_seeds(config.seed)
    if key is None:
        key = jax.random.PRNGKey(config.seed)

    key, subkey = jax.random.split(key)
    q_idx = int(jax.random.randint(subkey, (), 0, len(X_test)))
    x_q = X_test[q_idx:q_idx+1]

    # ════════════════════════════════════════════════════════
    # 1. BASELINE EVAL
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  BASELINE (before SC training)")
    print("=" * 60)

    pred_rule_baseline = ClassifierPredRule(
        categorical_x,
        n_estimators=config.n_estimators,
    )

    baseline_metrics = evaluate_basic(
        pred_rule_baseline,
        X_train, y_train,
        X_test, y_test,
        use_torch=True,
        task_type="classification",
    )
    print(f"  Accuracy: {baseline_metrics.accuracy:.4f}")
    print(f"  NLL:      {baseline_metrics.nll:.4f}")
    print(f"  ECE:      {baseline_metrics.ece:.4f}")

    print("\n  Computing baseline SC metric & drift curve...")
    key, subkey = jax.random.split(key)
    baseline_prefix = build_prefix_batch(
        key=subkey,
        pred_rule=pred_rule_baseline,
        x0=X_train, y0=y_train,
        prefix_depth=config.prefix_depth,
        continuation_depth=config.continuation_depth,
        x_q=x_q,
        n_continuations=config.n_continuations,
    )
    key, subkey = jax.random.split(key)
    baseline_horizon = sample_horizon_pair(
        subkey,
        total_depth=config.continuation_depth,
        k_max=config.k_max,
        prefix_depth=0,
    )
    baseline_sc = compute_sc_metric(
        pred_rule_baseline, baseline_prefix, baseline_horizon, x_q,
        use_torch=True,
    )
    baseline_drift = compute_drift_curve(
        pred_rule_baseline, baseline_prefix, x_q,
        use_torch=True,
    )
    print(f"  SC L2: {baseline_sc.l2_mean:.4f} ± {baseline_sc.l2_std:.4f}")
    print(f"  SC KL: {baseline_sc.kl_mean:.4f} ± {baseline_sc.kl_std:.4f}")
    print(f"  Drift (last depth var): {baseline_drift.points[-1].belief_var:.6f}")

    # ════════════════════════════════════════════════════════
    # 2. TRAINING
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SC + CE TRAINING")
    print("=" * 60)

    t0 = time.time()
    pred_rule_trained, train_state = train_and_merge(
        X_train, y_train, categorical_x,
        config=config,
        x_q_fixed=x_q,
        key=key,
    )
    train_time = time.time() - t0

    # ════════════════════════════════════════════════════════
    # 3. POST-TRAINING EVAL
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  POST-TRAINING EVAL")
    print("=" * 60)

    trained_metrics = evaluate_basic(
        pred_rule_trained,
        X_train, y_train,
        X_test, y_test,
        use_torch=True,
        task_type="classification",
    )
    print(f"  Accuracy: {trained_metrics.accuracy:.4f}")
    print(f"  NLL:      {trained_metrics.nll:.4f}")
    print(f"  ECE:      {trained_metrics.ece:.4f}")

    print("\n  Computing post-training SC metric & drift curve...")
    key, subkey = jax.random.split(key)
    trained_prefix = build_prefix_batch(
        key=subkey,
        pred_rule=pred_rule_trained,
        x0=X_train, y0=y_train,
        prefix_depth=config.prefix_depth,
        continuation_depth=config.continuation_depth,
        x_q=x_q,
        n_continuations=config.n_continuations,
    )
    trained_sc = compute_sc_metric(
        pred_rule_trained, trained_prefix, baseline_horizon, x_q,
        use_torch=True,
    )
    trained_drift = compute_drift_curve(
        pred_rule_trained, trained_prefix, x_q,
        use_torch=True,
    )
    print(f"  SC L2: {trained_sc.l2_mean:.4f} ± {trained_sc.l2_std:.4f}")
    print(f"  SC KL: {trained_sc.kl_mean:.4f} ± {trained_sc.kl_std:.4f}")
    print(f"  Drift (last depth var): {trained_drift.points[-1].belief_var:.6f}")

    # ════════════════════════════════════════════════════════
    # 4. COMPARISON
    # ════════════════════════════════════════════════════════
    comp = EvalComparison(
        before=baseline_metrics,
        after=trained_metrics,
        sc_before=baseline_sc,
        sc_after=trained_sc,
        drift_before=baseline_drift,
        drift_after=trained_drift,
    )
    print_comparison(comp)

    print("\n── Drift Curve (belief variance by depth) ──")
    print(f"  {'depth':>5} {'before':>10} {'after':>10} {'Δ':>10}")
    print(f"  {'─'*37}")
    for pb, pa in zip(baseline_drift.points, trained_drift.points):
        delta = pa.belief_var - pb.belief_var
        print(f"  {pb.depth:>5} {pb.belief_var:>10.6f} {pa.belief_var:>10.6f} {delta:>+10.6f}")

    print(f"\n── Training Summary ──")
    print(f"  Total time: {train_time:.1f}s")
    print(f"  Final loss: {train_state.losses[-1]:.4f}")
    print(f"  Loss curve (first/mid/last): "
          f"{train_state.losses[0]:.4f} → "
          f"{train_state.losses[len(train_state.losses)//2]:.4f} → "
          f"{train_state.losses[-1]:.4f}")

    return comp, train_state


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    config = TrainConfig(
        task_type="classification",
        n_estimators=1,
        seed=42,
        prefix_depth=3,
        continuation_depth=10,
        n_continuations=4,
        k_max=3,
        lam=0.3,
        query_ratio=0.2,
        num_steps=30,
        log_every=5,
        lr=1e-4,
        grad_clip=1.0,
        device="cuda",
    )

    set_global_seeds(config.seed)
    X_train, X_test, y_train, y_test, categorical_x, le = load_vehicle_dataset(
        subsample=400,
        seed=config.seed,
    )

    comp, state = run_experiment(
        X_train, X_test, y_train, y_test, categorical_x,
        config=config,
    )