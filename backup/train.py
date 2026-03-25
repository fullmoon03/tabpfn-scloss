"""
train.py  (Step 7)
─────────────────────────────────────────────
SC + Supervised CE 학습 루프.

핵심 원칙:
  - PrefixBatch 1개 = optimizer step 1번
  - Rollout (데이터 생성)은 pred_rule_sampling (sklearn, no-grad)
  - Belief 계산은 pred_rule_train (torch, autograd)
  - 두 pred_rule은 반드시 별도 인스턴스

학습 루프 구조:
  1. Base dataset D0 선택
  2. Prefix depth n, horizon pair (k1, k2) 샘플링
  3. Data-only prefix batch 생성 (sklearn path)
  4. Torch belief로 SC loss 계산 (autograd path)
  5. Supervised CE loss 계산 (torch path)
  6. Combined loss → backward → optimizer step
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import jax
import numpy as np
import torch
from jaxtyping import PRNGKeyArray

from predictive_rule import ClassifierPredRule, RegressorPredRule, PredictiveRule
from rollout import (
    build_prefix_batch_data,
    compute_mc_marginals_torch,
    sample_horizon_pair,
)
from loss import (
    sc_loss,
    supervised_ce_loss_torch,
    supervised_nll_regression_torch,
    combined_loss,
    LossResult,
)
from lora import (
    LoRAConfig,
    get_tabpfn_model,
    inject_lora,
    get_lora_params,
    merge_lora,
    print_lora_summary,
)


# ── Training Config ─────────────────────────────────────────
@dataclass
class TrainConfig:
    """학습 하이퍼파라미터."""
    # Task
    task_type: str = "classification"  # "classification" or "regression"

    # Rollout
    n_estimators: int = 1        # TabPFN ensemble size
    prefix_depth: int = 5          # n: prefix rollout depth
    continuation_depth: int = 15   # T: continuation rollout depth
    n_continuations: int = 8       # B: MC continuations per prefix
    k_max: int = 5                 # Kmax: horizon pair upper bound

    # Loss
    lam: float = 0.3               # λ: SC weight in combined loss
    query_ratio: float = 0.2       # fraction of D0 to hold out as query set

    # LoRA
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_target_layers: Optional[tuple] = None  # None = auto (last 4 blocks)

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.01
    num_steps: int = 100
    grad_clip: float = 1.0         # max gradient norm (0 = disabled)

    # Logging
    seed: int = 42
    log_every: int = 10
    device: str = "cpu"


# ── Training State ──────────────────────────────────────────
@dataclass
class TrainState:
    """학습 상태 추적."""
    step: int = 0
    losses: list[float] = field(default_factory=list)
    sc_losses: list[float] = field(default_factory=list)
    ce_losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)


# ── Dataset Split ───────────────────────────────────────────
def split_context_query(
    key: PRNGKeyArray,
    x: np.ndarray,
    y: np.ndarray,
    query_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    n_query = max(1, int(n * query_ratio))
    n_context = n - n_query

    perm = jax.random.permutation(key, n)
    perm = np.array(perm)

    idx_ctx = perm[:n_context]
    idx_q = perm[n_context:]

    return x[idx_ctx], y[idx_ctx], x[idx_q], y[idx_q]


# ── Core Training Function ──────────────────────────────────
def train(
    x0: np.ndarray,
    y0: np.ndarray,
    categorical_x: list[bool],
    config: TrainConfig = TrainConfig(),
    x_q_fixed: Optional[np.ndarray] = None,
    key: Optional[PRNGKeyArray] = None,
) -> tuple[PredictiveRule, TrainState]:
    # Unified reproducibility seed for numpy/torch/jax entrypoint.
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if key is None:
        key = jax.random.PRNGKey(config.seed)

    device = torch.device(config.device)

    # ── 1. Create two separate pred_rule instances ───────
    if config.task_type == "classification":
        pred_rule_sampling = ClassifierPredRule(categorical_x, n_estimators=config.n_estimators)
        pred_rule_train = ClassifierPredRule(categorical_x, n_estimators=config.n_estimators)
    elif config.task_type == "regression":
        pred_rule_sampling = RegressorPredRule(categorical_x, n_estimators=config.n_estimators)
        pred_rule_train = RegressorPredRule(categorical_x, n_estimators=config.n_estimators)
    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")

    # Initialize both with base dataset
    pred_rule_sampling.fit(x0, y0)
    pred_rule_train.fit(x0, y0)

    # Move sampling model to device
    sampling_model = get_tabpfn_model(pred_rule_sampling)
    sampling_model.to(device)

    assert pred_rule_sampling is not pred_rule_train, \
        "pred_rule_sampling and pred_rule_train must be different instances"

    # ── 2. Inject LoRA into training model ───────────────
    train_model = get_tabpfn_model(pred_rule_train)
    lora_config = LoRAConfig(
        r=config.lora_r,
        alpha=config.lora_alpha,
        target_layers=config.lora_target_layers,
    )
    lora_modules = inject_lora(train_model, lora_config)
    lora_params = get_lora_params(lora_modules)
    train_model.to(device)
    summary = print_lora_summary(train_model, lora_modules)

    # ── 3. Optimizer ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # ── 4. Fixed query point ─────────────────────────────
    if x_q_fixed is None:
        key, subkey = jax.random.split(key)
        idx = int(jax.random.randint(subkey, (), 0, x0.shape[0]))
        x_q_fixed = x0[idx:idx+1]
    x_q_fixed = np.atleast_2d(x_q_fixed)

    # ── 5. Fixed query split ─────────────────────────────
    key, subkey = jax.random.split(key)
    x_ctx_base, y_ctx_base, x_query, y_query = split_context_query(
        subkey, x0, y0, config.query_ratio,
    )

    # ── 6. Training loop ─────────────────────────────────
    state = TrainState()
    train_model.train()

    print(f"\n── Training ({config.num_steps} steps, {config.task_type}) ──")
    print(f"  λ={config.lam}, B={config.n_continuations}, "
          f"prefix={config.prefix_depth}, cont={config.continuation_depth}")
    print(f"  context={len(y_ctx_base)}, query={len(y_query)} (fixed)")
    print()

    for step in range(config.num_steps):
        t0 = time.time()

        # (1) Build data-only prefix batch (sklearn path, no-grad)
        key, subkey = jax.random.split(key)
        prefix_batch = build_prefix_batch_data(
            key=subkey,
            pred_rule_sampling=pred_rule_sampling,
            x0=x_ctx_base,
            y0=y_ctx_base,
            prefix_depth=config.prefix_depth,
            continuation_depth=config.continuation_depth,
            n_continuations=config.n_continuations,
        )

        # (2) Sample horizon pair
        key, subkey = jax.random.split(key)
        horizon = sample_horizon_pair(
            key=subkey,
            total_depth=config.continuation_depth,
            k_max=config.k_max,
            prefix_depth=0,
        )

        # (3) Torch beliefs → SC loss (autograd path)
        p_early_list, p_late_mean = compute_mc_marginals_torch(
            pred_rule_train=pred_rule_train,
            prefix_batch=prefix_batch,
            horizon=horizon,
            x_q=x_q_fixed,
        )
        L_sc = sc_loss(p_early_list, p_late_mean)

        # (4) Supervised loss (torch path, no fit())
        if config.task_type == "classification":
            L_ce = supervised_ce_loss_torch(
                pred_rule=pred_rule_train,
                x_context=x_ctx_base,
                y_context=y_ctx_base,
                x_query=x_query,
                y_query=y_query,
            )
        else:
            L_ce = supervised_nll_regression_torch(
                pred_rule=pred_rule_train,
                x_context=x_ctx_base,
                y_context=y_ctx_base,
                x_query=x_query,
                y_query=y_query,
            )

        # (5) Combined loss → backward → step
        result = combined_loss(L_sc, L_ce, lam=config.lam)

        if not torch.isfinite(result.total):
            print(
                f"  [warn] non-finite loss at step {step+1}: "
                f"L={result.total.item()}, SC={result.sc_loss.item()}, CE={result.ce_loss.item()} "
                "→ skipping step"
            )
            continue

        optimizer.zero_grad(set_to_none=True)
        result.total.backward()

        # Skip optimizer step if any LoRA grad is non-finite.
        has_non_finite_grad = False
        for p in lora_params:
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                has_non_finite_grad = True
                break
        if has_non_finite_grad:
            print(f"  [warn] non-finite gradient at step {step+1} → skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Gradient clipping
        grad_norm = 0.0
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                lora_params, config.grad_clip,
            ).item()
        else:
            grad_norm = sum(
                p.grad.norm().item() ** 2 for p in lora_params if p.grad is not None
            ) ** 0.5

        optimizer.step()

        # ── Logging ──────────────────────────────────────
        elapsed = time.time() - t0
        state.step = step + 1
        state.losses.append(result.total.item())
        state.sc_losses.append(result.sc_loss.item())
        state.ce_losses.append(result.ce_loss.item())
        state.grad_norms.append(grad_norm)
        state.elapsed.append(elapsed)

        if (step + 1) % config.log_every == 0 or step == 0:
            avg_loss = np.mean(state.losses[-config.log_every:])
            avg_sc = np.mean(state.sc_losses[-config.log_every:])
            avg_ce = np.mean(state.ce_losses[-config.log_every:])
            avg_gn = np.mean(state.grad_norms[-config.log_every:])
            print(
                f"  [{step+1:4d}/{config.num_steps}] "
                f"L={avg_loss:.4f} (SC={avg_sc:.4f}, CE={avg_ce:.4f}) "
                f"|g|={avg_gn:.4f}  {elapsed:.2f}s"
            )

    print(f"\n── Training complete ({sum(state.elapsed):.1f}s total) ──")

    return pred_rule_train, state


# ── Convenience: Train + Merge ──────────────────────────────
def train_and_merge(
    x0: np.ndarray,
    y0: np.ndarray,
    categorical_x: list[bool],
    config: TrainConfig = TrainConfig(),
    **kwargs,
) -> tuple[PredictiveRule, TrainState]:
    pred_rule_train, state = train(x0, y0, categorical_x, config, **kwargs)

    train_model = get_tabpfn_model(pred_rule_train)
    n_merged = merge_lora(train_model)
    print(f"  LoRA merged: {n_merged} modules into base model.")

    pred_rule_train.lock_weights()
    print("  Weights locked (will survive fit() calls).")

    return pred_rule_train, state


# ── Main (example usage) ────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    data = fetch_openml(data_id=54, as_frame=False, parser="auto")
    X, y_raw = data.data, data.target.astype(int)

    np.random.seed(0)
    idx = np.random.choice(len(X), size=200, replace=False)
    X, y_raw = X[idx], y_raw[idx]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    categorical_x = [False] * X.shape[1]

    config = TrainConfig(
        prefix_depth=3,
        continuation_depth=8,
        n_continuations=4,
        k_max=3,
        lam=0.3,
        num_steps=20,
        log_every=5,
        lr=1e-4,
    )

    pred_rule, state = train_and_merge(X, y, categorical_x, config)
    print(f"\nFinal losses: {state.losses[-5:]}")