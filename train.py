"""
SC 학습 루프.

Rollout 데이터 생성은 sampling pred_rule(sklearn path),
loss 계산은 train pred_rule(torch path)로 분리한다.
"""

import time
import os
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

import numpy as np
import torch
from jaxtyping import PRNGKeyArray

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from predictive_rule import ClassifierPredRule, RegressorPredRule, PredictiveRule
from rollout import (
    build_prefix_batch_data,
    belief_at_depth_torch_batched,
    horizon_k_to_depth,
)
from loss import (
    sc_loss,
)
from eval import (
    build_fixed_synthetic_anchor_suite,
    _compute_emd_fixed_anchor_suite,
    _compute_fixed_anchor_suite_eval_with_current_lora,
)
from lora import (
    LoRAConfig,
    LoRALinear,
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
    continuation_depth: int = 15   # T: continuation rollout depth
    n_continuations: int = 8       # B: MC continuations per prefix
    k_max: int = 5                 # Kmax: horizon pair upper bound

    # SC sampling
    sc_context_size: int = 50      # per-episode context size sampled from each task
    sc_task_query_pool_size: int = 20  # synthetic mode: fixed per-task query-pool size
    sc_num_pairs_per_query: int = 4  # sampled (k1, k2) pair count per query
    sc_k1_range: tuple[int, int] = (1, 9)  # inclusive student horizon range
    sc_k2_range: tuple[int, int] = (10, 15)  # inclusive teacher horizon range
    # Legacy SC config kept for compatibility with older entrypoints; ignored by
    # the current synthetic pairwise SC implementation.
    sc_anchor_ks: tuple[int, ...] = (3, 5, 7)
    sc_anchor_weights: tuple[float, ...] = (1.0, 1.5, 2.0)
    sc_enable_chain: bool = False
    sc_chain_beta_warmup: float = 0.1
    sc_chain_beta_warmup_steps: int = 10
    sc_chain_beta: float = 0.3
    sc_episodes_per_step: int = 3  # independent SC episodes per optimization step
    sc_queries_per_episode: int = 4  # sampled queries per episode from fixed SC query pool
    sc_prefix_depths: tuple[int, ...] = (0, 2, 4)  # SC prefix-depth set n

    # EMD monitoring (fixed anchor suite on held-out synthetic tasks)
    emd_anchor_count: int = 4      # number of fixed anchors
    emd_context_size: int = 100    # fixed context size per anchor
    emd_queries_per_anchor: int = 5  # fixed query count per anchor
    emd_prefix_depths: tuple[int, ...] = (0, 2, 4)
    emd_k_values: tuple[int, ...] = (3, 7, 11, 15)  # p^(1) 대비 EMD 계산 대상 k
    enable_emd: bool = True        # False면 EMD 계산/기록 비활성화
    emd_fill_every: int = 5        # EMD 계산 주기 (step 단위, step 0은 항상 계산)
    emd_fixed_rollout_paths: bool = True  # (anchor, n) rollout 경로 고정
    emd_rng_fold_in: int = 202     # fold_in value for deterministic EMD RNG stream

    # LoRA
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_target_layers: Optional[tuple] = None  # None = auto (last 4 blocks)
    lora_include_decoder: bool = False  # False => encoder-only LoRA

    # Optimizer
    lr: float = 1e-4
    lr_decay_after_step: Optional[int] = None  # zero-based step threshold; decay applies for step > threshold
    lr_decay_factor: float = 1.0  # multiplier applied after decay threshold
    weight_decay: float = 0.01
    num_steps: int = 100
    grad_clip: float = 1.0         # max gradient norm (0 = disabled)

    # Logging
    seed: int = 42
    device: str = "cpu"


# ── Training State ──────────────────────────────────────────
@dataclass
class TrainState:
    """학습 상태 추적."""
    step: int = 0
    losses: list[float] = field(default_factory=list)
    sc_losses: list[float] = field(default_factory=list)
    p1_tuned_conf: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    # Step-wise EMD history
    emd_steps: list[int] = field(default_factory=list)
    emd_values: list[float] = field(default_factory=list)
    emd_stds: list[float] = field(default_factory=list)
    emd_coverage: list[int] = field(default_factory=list)
    met_acc_values: list[float] = field(default_factory=list)
    met_nll_values: list[float] = field(default_factory=list)
    met_ece_values: list[float] = field(default_factory=list)
    best_nll_step: Optional[int] = None
    best_nll_value: Optional[float] = None
    # LoRA adapter snapshot (filled in train_and_merge before merge)
    lora_adapter_state: Optional[dict] = None


# ── Dataset Split ───────────────────────────────────────────
def _sample_indices_without_replacement(
    key: PRNGKeyArray,
    n_total: int,
    sample_size: int,
) -> np.ndarray:
    """Sample indices uniformly without replacement."""
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    size = int(np.clip(sample_size, 1, n_total))
    perm = np.array(jax.random.permutation(key, n_total))
    return perm[:size].astype(int)


def _snapshot_lora_adapters(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Snapshot only LoRA adapter tensors to CPU."""
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            snapshot[name] = {
                "lora_A": module.lora_A.detach().cpu().clone(),
                "lora_B": module.lora_B.detach().cpu().clone(),
            }
    return snapshot


def _load_lora_adapters(
    model: torch.nn.Module,
    snapshot: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Restore LoRA adapter tensors from CPU snapshot."""
    name_to_module = dict(model.named_modules())
    with torch.no_grad():
        for name, tensors in snapshot.items():
            module = name_to_module.get(name)
            if not isinstance(module, LoRALinear):
                continue
            module.lora_A.copy_(tensors["lora_A"].to(module.lora_A.device, dtype=module.lora_A.dtype))
            module.lora_B.copy_(tensors["lora_B"].to(module.lora_B.device, dtype=module.lora_B.dtype))


def _relabel_to_contiguous(y: np.ndarray) -> np.ndarray:
    """Relabel labels to contiguous 0..C_local-1."""
    y_arr = np.asarray(y).astype(int)
    _, y_local = np.unique(y_arr, return_inverse=True)
    return y_local.astype(np.int64)


def _split_synthetic_task_pools(
    *,
    key: PRNGKeyArray,
    x_task: np.ndarray,
    y_task: np.ndarray,
    context_size: int,
    query_pool_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split one synthetic task into:
      - context D0
      - query pool
      - rollout feature pool
    """
    x_task = np.asarray(x_task)
    y_task = np.asarray(y_task).astype(int)
    idx_ctx, idx_qpool, idx_roll = _split_synthetic_task_indices(
        key=key,
        y_task=y_task,
        context_size=context_size,
        query_pool_size=query_pool_size,
    )
    if len(idx_roll) == 0:
        raise ValueError(
            f"Rollout pool is empty after split: n_points={len(x_task)}, "
            f"context_size={context_size}, query_pool_size={query_pool_size}"
        )

    x_ctx = x_task[idx_ctx]
    y_ctx_local = _relabel_to_contiguous(y_task[idx_ctx])
    x_qpool = x_task[idx_qpool]
    x_roll_pool = x_task[idx_roll]
    return x_ctx, y_ctx_local, x_qpool, x_roll_pool


def _split_synthetic_task_indices(
    *,
    key: PRNGKeyArray,
    y_task: np.ndarray,
    context_size: int,
    query_pool_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one synthetic task indices into context/query-pool/rollout-pool."""
    n_points = int(len(np.asarray(y_task)))
    if n_points < 3:
        raise ValueError(f"Synthetic task must have >=3 points, got {n_points}")
    n_ctx = int(np.clip(context_size, 1, n_points - 2))
    n_qpool = int(np.clip(query_pool_size, 1, n_points - n_ctx - 1))

    perm = np.asarray(jax.random.permutation(key, n_points)).astype(int)
    idx_ctx = perm[:n_ctx]
    idx_rem = perm[n_ctx:]
    idx_qpool = idx_rem[:n_qpool]
    idx_roll = idx_rem[n_qpool:]
    return idx_ctx, idx_qpool, idx_roll


def _build_prefix_continuation_map(
    *,
    key: PRNGKeyArray,
    pred_rule_sampling: PredictiveRule,
    x_context: np.ndarray,
    y_context: np.ndarray,
    prefix_depths: tuple[int, ...],
    continuation_depth: int,
    n_continuations: int,
    fixed_rollout_keys: Optional[dict[int, PRNGKeyArray]] = None,
    x_sampling_pool: Optional[np.ndarray] = None,
    x_sample_without_replacement: bool = False,
) -> tuple[dict[int, list], PRNGKeyArray]:
    """
    Build continuations once per prefix depth and return a prefix->continuations map.
    """
    continuations_by_prefix: dict[int, list] = {}
    for n in prefix_depths:
        n_int = int(n)
        if fixed_rollout_keys is not None:
            subkey = fixed_rollout_keys[n_int]
        else:
            key, subkey = jax.random.split(key)
        prefix_batch = build_prefix_batch_data(
            key=subkey,
            pred_rule_sampling=pred_rule_sampling,
            x0=x_context,
            y0=y_context,
            prefix_depth=n_int,
            continuation_depth=continuation_depth,
            n_continuations=n_continuations,
            x_sampling_pool=x_sampling_pool,
            x_sample_without_replacement=x_sample_without_replacement,
        )
        continuations_by_prefix[n_int] = prefix_batch.continuations
    return continuations_by_prefix, key


def _compute_query_marginals_for_ks(
    *,
    pred_rule_train: PredictiveRule,
    continuations: list,
    ks: tuple[int, ...],
    x_query: np.ndarray,
    no_grad_keys: set[int],
) -> dict[int, torch.Tensor]:
    """
    Compute MC-averaged beliefs p_hat[k] for a single query over a continuation set.
    """
    p_hat: dict[int, torch.Tensor] = {}
    for k in ks:
        depth = horizon_k_to_depth(int(k))
        if int(k) in no_grad_keys:
            with torch.no_grad():
                b_batch = belief_at_depth_torch_batched(
                    pred_rule_train, continuations, depth, x_query
                )  # (B, M, C)
        else:
            b_batch = belief_at_depth_torch_batched(
                pred_rule_train, continuations, depth, x_query
            )  # (B, M, C)
        p_hat[int(k)] = b_batch.mean(dim=0).squeeze(0)  # (C,)
    return p_hat


# ── Synthetic Task-Batch Training ──────────────────────────
def train_synthetic(
    x_tasks: np.ndarray,
    y_tasks: np.ndarray,
    categorical_x: list[bool],
    config: TrainConfig = TrainConfig(),
    key: Optional[PRNGKeyArray] = None,
    initial_lora_adapter_modules: Optional[dict[str, dict[str, Any]]] = None,
    emd_anchor_tasks_x: Optional[np.ndarray] = None,
    emd_anchor_tasks_y: Optional[np.ndarray] = None,
    emd_anchor_contexts: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    emd_anchor_contexts_global: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    emd_anchor_query_banks: Optional[list[np.ndarray]] = None,
    emd_anchor_rollout_pools: Optional[list[np.ndarray]] = None,
    emd_fixed_rollout_keys: Optional[dict[tuple[int, int], PRNGKeyArray]] = None,
    emd_anchor_context_class_ids: Optional[list[np.ndarray]] = None,
    emd_anchor_query_labels: Optional[list[np.ndarray]] = None,
    emd_global_num_classes: Optional[int] = None,
    step_callback: Optional[Callable[[int, PredictiveRule, PredictiveRule], None]] = None,
) -> tuple[PredictiveRule, TrainState]:
    """
    Synthetic task-batch SC training.

    - step = consume fresh synthetic task episodes in sequential order and optimize SC loss
    - rollout x sampling uses held-out task rollout pool (path-wise no replacement)
    - EMD monitors on fixed held-out synthetic anchor tasks
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    x_tasks = np.asarray(x_tasks)
    y_tasks = np.asarray(y_tasks).astype(int)
    if x_tasks.ndim != 3 or y_tasks.ndim != 2:
        raise ValueError(
            f"Expected synthetic tasks x:(N_task,N_point,d), y:(N_task,N_point), "
            f"got x:{x_tasks.shape}, y:{y_tasks.shape}"
        )
    if x_tasks.shape[0] != y_tasks.shape[0] or x_tasks.shape[1] != y_tasks.shape[1]:
        raise ValueError(
            f"Synthetic task shape mismatch: x:{x_tasks.shape}, y:{y_tasks.shape}"
        )
    n_tasks, n_points, _ = x_tasks.shape
    if n_tasks < 1 or n_points < 3:
        raise ValueError(f"Invalid synthetic task tensor shape: {x_tasks.shape}")

    if key is None:
        key = jax.random.PRNGKey(config.seed)

    device = torch.device(config.device)

    # Create separate sampling/train rules.
    if config.task_type == "classification":
        pred_rule_sampling = ClassifierPredRule(categorical_x, n_estimators=config.n_estimators)
        pred_rule_train = ClassifierPredRule(categorical_x, n_estimators=config.n_estimators)
    elif config.task_type == "regression":
        pred_rule_sampling = RegressorPredRule(categorical_x, n_estimators=config.n_estimators)
        pred_rule_train = RegressorPredRule(categorical_x, n_estimators=config.n_estimators)
    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")

    # Initialize inference state from a fixed eval anchor so the pre-EMD fit
    # path is reproducible across resumed runs.
    if not (config.enable_emd and emd_anchor_contexts is not None and len(emd_anchor_contexts) > 0):
        raise ValueError(
            "train_synthetic now requires fixed eval anchors when enable_emd=True."
        )
    x_ctx0, y_ctx0 = emd_anchor_contexts[0]
    pred_rule_sampling.fit(x_ctx0, y_ctx0)
    pred_rule_train.fit(x_ctx0, y_ctx0)

    # Inject LoRA.
    train_model = get_tabpfn_model(pred_rule_train)
    lora_config = LoRAConfig(
        r=config.lora_r,
        alpha=config.lora_alpha,
        target_layers=config.lora_target_layers,
        include_decoder=bool(config.lora_include_decoder),
    )
    lora_modules = inject_lora(train_model, lora_config)
    if initial_lora_adapter_modules is not None:
        _load_lora_adapters(train_model, initial_lora_adapter_modules)
    lora_params = get_lora_params(lora_modules)
    train_model.to(device)
    print_lora_summary(train_model, lora_modules)

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if config.sc_context_size < 1:
        raise ValueError(f"sc_context_size must be >=1, got {config.sc_context_size}")
    if config.sc_task_query_pool_size < 1:
        raise ValueError(
            f"sc_task_query_pool_size must be >=1, got {config.sc_task_query_pool_size}"
        )
    if config.sc_queries_per_episode < 1:
        raise ValueError(
            f"sc_queries_per_episode must be >=1, got {config.sc_queries_per_episode}"
        )
    if config.sc_episodes_per_step < 1:
        raise ValueError(
            f"sc_episodes_per_step must be >=1, got {config.sc_episodes_per_step}"
        )

    if int(config.sc_num_pairs_per_query) < 1:
        raise ValueError(
            f"sc_num_pairs_per_query must be >=1, got {config.sc_num_pairs_per_query}"
        )
    sc_k1_lo, sc_k1_hi = (int(config.sc_k1_range[0]), int(config.sc_k1_range[1]))
    sc_k2_lo, sc_k2_hi = (int(config.sc_k2_range[0]), int(config.sc_k2_range[1]))
    if sc_k1_lo < 1 or sc_k1_hi < sc_k1_lo:
        raise ValueError(f"Invalid sc_k1_range: {config.sc_k1_range}")
    if sc_k2_lo < 1 or sc_k2_hi < sc_k2_lo:
        raise ValueError(f"Invalid sc_k2_range: {config.sc_k2_range}")
    if sc_k1_hi > config.continuation_depth:
        raise ValueError(
            f"sc_k1_range must be <= continuation_depth={config.continuation_depth}, "
            f"got {config.sc_k1_range}"
        )
    if sc_k2_hi > config.continuation_depth:
        raise ValueError(
            f"sc_k2_range must be <= continuation_depth={config.continuation_depth}, "
            f"got {config.sc_k2_range}"
        )
    sc_k1_values = tuple(range(sc_k1_lo, sc_k1_hi + 1))
    sc_k2_values = tuple(range(sc_k2_lo, sc_k2_hi + 1))
    sc_prefix_depths = tuple(int(v) for v in config.sc_prefix_depths)
    if len(sc_prefix_depths) == 0 or min(sc_prefix_depths) < 0:
        raise ValueError(f"Invalid sc_prefix_depths: {sc_prefix_depths}")
    sc_ks_all = tuple(sorted({*sc_k1_values, *sc_k2_values}))
    sc_needed_depth = max(horizon_k_to_depth(k) for k in sc_ks_all)
    if sc_needed_depth > config.continuation_depth:
        raise ValueError(
            f"SC needs continuation depth {sc_needed_depth}, "
            f"but continuation_depth={config.continuation_depth}"
        )
    if config.sc_context_size + config.sc_task_query_pool_size + sc_needed_depth > n_points:
        raise ValueError(
            "Synthetic task too small for requested split/depth: "
            f"points={n_points}, context={config.sc_context_size}, "
            f"query_pool={config.sc_task_query_pool_size}, depth={sc_needed_depth}"
        )

    # Fixed synthetic EMD anchor suite from held-out tasks.
    key_emd: Optional[PRNGKeyArray] = None
    emd_anchor_contexts_local: list[tuple[np.ndarray, np.ndarray]] = []
    emd_anchor_context_class_ids_local: list[np.ndarray] = []
    emd_anchor_query_banks_local: list[np.ndarray] = []
    emd_anchor_query_labels_local: list[np.ndarray] = []
    emd_anchor_rollout_pools_local: list[np.ndarray] = []
    emd_fixed_rollout_keys_local: Optional[dict[tuple[int, int], PRNGKeyArray]] = None
    emd_global_num_classes_local: Optional[int] = None
    emd_prefix_depths = tuple(int(v) for v in config.emd_prefix_depths)
    emd_k_values = tuple(
        int(v) for v in config.emd_k_values
        if (int(v) > 1 and int(v) <= config.continuation_depth)
    )
    if config.enable_emd:
        if len(emd_k_values) == 0:
            raise ValueError(
                f"Filtered emd_k_values is empty: {config.emd_k_values}"
            )
        # If precomputed fixed anchors are supplied, use them as-is.
        if (
            emd_anchor_contexts is not None
            and emd_anchor_context_class_ids is not None
            and emd_anchor_query_banks is not None
            and emd_anchor_query_labels is not None
            and emd_anchor_rollout_pools is not None
            and emd_global_num_classes is not None
        ):
            # The fixed eval suite already determines the EMD randomness; keep
            # this key independent from train-side seeds.
            key_emd = jax.random.PRNGKey(0)
            emd_anchor_contexts_local = list(emd_anchor_contexts)
            emd_anchor_context_class_ids_local = list(emd_anchor_context_class_ids)
            emd_anchor_query_banks_local = list(emd_anchor_query_banks)
            emd_anchor_query_labels_local = list(emd_anchor_query_labels)
            emd_anchor_rollout_pools_local = list(emd_anchor_rollout_pools)
            emd_fixed_rollout_keys_local = emd_fixed_rollout_keys
            emd_global_num_classes_local = int(emd_global_num_classes)
        else:
            key_emd = jax.random.fold_in(
                jax.random.PRNGKey(config.seed),
                int(config.emd_rng_fold_in),
            )
            if emd_anchor_tasks_x is None or emd_anchor_tasks_y is None:
                raise ValueError(
                    "train_synthetic requires held-out synthetic anchors "
                    "(emd_anchor_tasks_x / emd_anchor_tasks_y) when enable_emd=True."
                )
            (
                emd_anchor_contexts_local,
                _,
                emd_anchor_context_class_ids_local,
                emd_anchor_query_banks_local,
                emd_anchor_query_labels_local,
                emd_anchor_rollout_pools_local,
                emd_fixed_rollout_keys_local,
                key_emd,
            ) = build_fixed_synthetic_anchor_suite(
                key=key_emd,
                x_tasks=np.asarray(emd_anchor_tasks_x),
                y_tasks=np.asarray(emd_anchor_tasks_y),
                anchor_count=config.emd_anchor_count,
                context_size=config.emd_context_size,
                query_pool_size=config.sc_task_query_pool_size,
                queries_per_anchor=config.emd_queries_per_anchor,
                prefix_depths=emd_prefix_depths,
                fixed_rollout_paths=config.emd_fixed_rollout_paths,
            )
            emd_global_num_classes_local = int(np.max(np.asarray(emd_anchor_tasks_y))) + 1

    emd_pairs_total = (
        sum(len(qb) for qb in emd_anchor_query_banks_local) * len(emd_prefix_depths)
        if config.enable_emd
        else 0
    )

    print(f"\n── Training ({config.num_steps} steps, synthetic {config.task_type}) ──")
    print(f"  tasks={n_tasks}, points/task={n_points}, B={config.n_continuations}, cont={config.continuation_depth}")
    print(
        f"  Synthetic split/task: context={config.sc_context_size}, "
        f"query_pool={config.sc_task_query_pool_size}, "
        f"queries/episode={config.sc_queries_per_episode}, "
        f"rollout_pool={n_points - config.sc_context_size - config.sc_task_query_pool_size}"
    )
    print(
        f"  SC: episodes/step={config.sc_episodes_per_step}, prefix n={sc_prefix_depths}, "
        f"pairs/query={config.sc_num_pairs_per_query}, "
        f"k1_range={config.sc_k1_range}, k2_range={config.sc_k2_range}"
    )
    if config.enable_emd:
        print(
            f"  EMD anchors={len(emd_anchor_contexts_local)}, context/anchor={config.emd_context_size}, "
            f"queries/anchor={config.emd_queries_per_anchor}, pairs/eval={emd_pairs_total}, "
            f"k={emd_k_values}, "
            f"every {config.emd_fill_every} step(s)"
        )
    else:
        print("  EMD=off")
    print()

    state = TrainState()
    train_model.train()
    best_nll_value = float("inf")
    best_nll_step: Optional[int] = None
    best_lora_snapshot: Optional[dict[str, dict[str, torch.Tensor]]] = None

    if config.enable_emd:
        if config.task_type == "classification":
            init_emd, init_std, init_cov, init_acc, init_nll, init_ece, key_emd = _compute_fixed_anchor_suite_eval_with_current_lora(
                key=key_emd,
                train_model=train_model,
                categorical_x=categorical_x,
                n_estimators=config.n_estimators,
                lora_config=lora_config,
                base_model_state_dict=None,
                anchor_contexts=emd_anchor_contexts_local,
                anchor_context_class_ids=emd_anchor_context_class_ids_local,
                anchor_query_banks_x=emd_anchor_query_banks_local,
                anchor_query_banks_y=emd_anchor_query_labels_local,
                global_num_classes=int(emd_global_num_classes_local),
                prefix_depths=emd_prefix_depths,
                k_values=emd_k_values,
                continuation_depth=config.continuation_depth,
                n_continuations=config.n_continuations,
                fixed_rollout_keys=emd_fixed_rollout_keys_local,
                anchor_rollout_pools=emd_anchor_rollout_pools_local,
            )
        else:
            init_emd, init_std, init_cov, key_emd = _compute_emd_fixed_anchor_suite(
                key=key_emd,
                pred_rule_train=pred_rule_train,
                pred_rule_sampling=pred_rule_sampling,
                anchor_contexts=emd_anchor_contexts_local,
                anchor_query_banks=emd_anchor_query_banks_local,
                prefix_depths=emd_prefix_depths,
                k_values=emd_k_values,
                continuation_depth=config.continuation_depth,
                n_continuations=config.n_continuations,
                fixed_rollout_keys=emd_fixed_rollout_keys_local,
                anchor_rollout_pools=emd_anchor_rollout_pools_local,
            )
        state.emd_steps.append(0)
        state.emd_values.append(init_emd)
        state.emd_stds.append(init_std)
        state.emd_coverage.append(init_cov)
        print(
            f"  [EMD {0:4d}] mean={init_emd:.6f} (std={init_std:.6f}, cov={init_cov}/{emd_pairs_total}) "
            f"(before optimization)"
        )
        if config.task_type == "classification":
            state.met_acc_values.append(init_acc)
            state.met_nll_values.append(init_nll)
            state.met_ece_values.append(init_ece)
            if np.isfinite(init_nll):
                best_nll_value = float(init_nll)
                best_nll_step = 0
                best_lora_snapshot = _snapshot_lora_adapters(train_model)
            print(
                f"  [MET {0:4d}] acc={init_acc:.4f}, nll={init_nll:.4f}, ece={init_ece:.4f}"
            )
        elif np.isfinite(init_emd):
            best_lora_snapshot = _snapshot_lora_adapters(train_model)

    # Task sampling without replacement:
    # consume a random permutation of task indices, then reshuffle when exhausted.
    key, k_task_perm = jax.random.split(key)
    task_perm = np.asarray(jax.random.permutation(k_task_perm, n_tasks)).astype(int)
    task_cursor = 0

    for step in range(config.num_steps):
        t0 = time.time()

        lr_this_step = config.lr
        if (
            config.lr_decay_after_step is not None
            and step > int(config.lr_decay_after_step)
            and config.lr_decay_factor < 1.0
        ):
            lr_this_step = config.lr * config.lr_decay_factor
        for group in optimizer.param_groups:
            group["lr"] = lr_this_step

        optimizer.zero_grad(set_to_none=True)

        episode_loss_vals: list[float] = []
        episode_sc_loss_vals: list[float] = []
        episode_tuned_conf: list[float] = []

        for _ in range(config.sc_episodes_per_step):
            if task_cursor >= n_tasks:
                key, k_task_perm = jax.random.split(key)
                task_perm = np.asarray(
                    jax.random.permutation(k_task_perm, n_tasks)
                ).astype(int)
                task_cursor = 0
            task_idx = int(task_perm[task_cursor])
            task_cursor += 1
            x_task = x_tasks[task_idx]
            y_task = y_tasks[task_idx]

            key, k_split = jax.random.split(key)
            x_ctx_episode, y_ctx_episode, x_qpool, x_roll_pool = _split_synthetic_task_pools(
                key=k_split,
                x_task=x_task,
                y_task=y_task,
                context_size=config.sc_context_size,
                query_pool_size=config.sc_task_query_pool_size,
            )

            key, k_q = jax.random.split(key)
            if len(x_qpool) >= config.sc_queries_per_episode:
                q_local = _sample_indices_without_replacement(
                    k_q, n_total=len(x_qpool), sample_size=config.sc_queries_per_episode
                )
                q_episode = x_qpool[q_local]
            else:
                pos = np.asarray(
                    jax.random.choice(
                        k_q,
                        len(x_qpool),
                        shape=(config.sc_queries_per_episode,),
                        replace=True,
                    )
                ).astype(int)
                q_episode = x_qpool[pos]

            per_pair_loss_vals: list[float] = []
            per_pair_sc_loss_vals: list[float] = []
            per_pair_tuned_conf: list[float] = []

            conts_by_prefix, key = _build_prefix_continuation_map(
                key=key,
                pred_rule_sampling=pred_rule_sampling,
                x_context=x_ctx_episode,
                y_context=y_ctx_episode,
                prefix_depths=sc_prefix_depths,
                continuation_depth=sc_needed_depth,
                n_continuations=config.n_continuations,
                fixed_rollout_keys=None,
                x_sampling_pool=x_roll_pool,
                x_sample_without_replacement=True,
            )

            for sc_prefix_depth in sc_prefix_depths:
                conts = conts_by_prefix[int(sc_prefix_depth)]
                for q_val in q_episode:
                    key, k_pair_k1 = jax.random.split(key)
                    key, k_pair_k2 = jax.random.split(key)
                    sampled_k1 = np.asarray(
                        jax.random.randint(
                            k_pair_k1,
                            shape=(int(config.sc_num_pairs_per_query),),
                            minval=sc_k1_lo,
                            maxval=sc_k1_hi + 1,
                        )
                    ).astype(int)
                    sampled_k2 = np.asarray(
                        jax.random.randint(
                            k_pair_k2,
                            shape=(int(config.sc_num_pairs_per_query),),
                            minval=sc_k2_lo,
                            maxval=sc_k2_hi + 1,
                        )
                    ).astype(int)
                    sampled_pairs = tuple(
                        (int(k1), int(k2)) for k1, k2 in zip(sampled_k1.tolist(), sampled_k2.tolist())
                    )
                    ks_for_query = tuple(
                        sorted({int(k) for pair in sampled_pairs for k in pair})
                    )
                    no_grad_keys_for_query: set[int] = (
                        {int(k2) for _, k2 in sampled_pairs}
                        .difference({int(k1) for k1, _ in sampled_pairs})
                    )
                    sc_query = np.atleast_2d(q_val)
                    p_hat_tuned = _compute_query_marginals_for_ks(
                        pred_rule_train=pred_rule_train,
                        continuations=conts,
                        ks=ks_for_query,
                        x_query=sc_query,
                        no_grad_keys=no_grad_keys_for_query,
                    )
                    l_sc_pair = sc_loss(
                        p_by_k=p_hat_tuned,
                        sampled_pairs=sampled_pairs,
                    )
                    micro_loss = l_sc_pair / (
                        float(config.sc_episodes_per_step)
                        * float(len(sc_prefix_depths))
                        * float(len(q_episode))
                    )
                    micro_loss.backward()
                    tuned_conf_k = min(int(k1) for k1, _ in sampled_pairs)
                    p1_tuned = p_hat_tuned[tuned_conf_k]
                    p1_tuned_prob = p1_tuned / p1_tuned.sum().clamp_min(1e-8)
                    per_pair_tuned_conf.append(float(p1_tuned_prob.max().detach().cpu().item()))
                    per_pair_sc_loss_vals.append(float(l_sc_pair.detach().cpu().item()))
                    per_pair_loss_vals.append(float(l_sc_pair.detach().cpu().item()))

            if len(per_pair_loss_vals) == 0:
                raise RuntimeError("No SC losses were produced in episode.")
            episode_loss_vals.append(float(np.mean(per_pair_loss_vals)))
            episode_sc_loss_vals.append(float(np.mean(per_pair_sc_loss_vals)))
            episode_tuned_conf.append(float(np.mean(per_pair_tuned_conf)))

        L_total = float(np.mean(episode_loss_vals))
        L_sc = float(np.mean(episode_sc_loss_vals))
        p1_tuned_conf_step = float(np.mean(episode_tuned_conf))

        has_non_finite_grad = False
        for p in lora_params:
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                has_non_finite_grad = True
                break
        if has_non_finite_grad:
            print(f"  [warn] non-finite gradient at step {step+1} → skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, config.grad_clip).item()
        else:
            grad_norm = sum(
                p.grad.norm().item() ** 2 for p in lora_params if p.grad is not None
            ) ** 0.5
        optimizer.step()

        elapsed = time.time() - t0
        state.step = step + 1
        state.losses.append(L_total)
        state.sc_losses.append(L_sc)
        state.p1_tuned_conf.append(float(p1_tuned_conf_step))
        state.grad_norms.append(grad_norm)
        state.elapsed.append(elapsed)

        print(
            f"  [{step+1:4d}/{config.num_steps}] "
            f"L={L_total:.4f} "
            f"(SC={L_sc:.4f}, conf_tuned={p1_tuned_conf_step:.3f}) "
            f"|g|={grad_norm:.4f}  {elapsed:.2f}s"
        )

        if config.enable_emd and ((step + 1) % config.emd_fill_every == 0):
            if config.task_type == "classification":
                emd_mean, emd_std, emd_cov, met_acc, met_nll, met_ece, key_emd = _compute_fixed_anchor_suite_eval_with_current_lora(
                    key=key_emd,
                    train_model=train_model,
                    categorical_x=categorical_x,
                    n_estimators=config.n_estimators,
                    lora_config=lora_config,
                    base_model_state_dict=None,
                    anchor_contexts=emd_anchor_contexts_local,
                    anchor_context_class_ids=emd_anchor_context_class_ids_local,
                    anchor_query_banks_x=emd_anchor_query_banks_local,
                    anchor_query_banks_y=emd_anchor_query_labels_local,
                    global_num_classes=int(emd_global_num_classes_local),
                    prefix_depths=emd_prefix_depths,
                    k_values=emd_k_values,
                    continuation_depth=config.continuation_depth,
                    n_continuations=config.n_continuations,
                    fixed_rollout_keys=emd_fixed_rollout_keys_local,
                    anchor_rollout_pools=emd_anchor_rollout_pools_local,
                )
            else:
                emd_mean, emd_std, emd_cov, key_emd = _compute_emd_fixed_anchor_suite(
                    key=key_emd,
                    pred_rule_train=pred_rule_train,
                    pred_rule_sampling=pred_rule_sampling,
                    anchor_contexts=emd_anchor_contexts_local,
                    anchor_query_banks=emd_anchor_query_banks_local,
                    prefix_depths=emd_prefix_depths,
                    k_values=emd_k_values,
                    continuation_depth=config.continuation_depth,
                    n_continuations=config.n_continuations,
                    fixed_rollout_keys=emd_fixed_rollout_keys_local,
                    anchor_rollout_pools=emd_anchor_rollout_pools_local,
                )
            state.emd_steps.append(step + 1)
            state.emd_values.append(emd_mean)
            state.emd_stds.append(emd_std)
            state.emd_coverage.append(emd_cov)
            print(
                f"  [EMD {step+1:4d}] mean={emd_mean:.6f} (std={emd_std:.6f}, cov={emd_cov}/{emd_pairs_total})"
            )
            if config.task_type == "classification":
                state.met_acc_values.append(met_acc)
                state.met_nll_values.append(met_nll)
                state.met_ece_values.append(met_ece)
                if np.isfinite(met_nll) and float(met_nll) < best_nll_value:
                    best_nll_value = float(met_nll)
                    best_nll_step = step + 1
                    best_lora_snapshot = _snapshot_lora_adapters(train_model)
                print(
                    f"  [MET {step+1:4d}] acc={met_acc:.4f}, nll={met_nll:.4f}, ece={met_ece:.4f}"
                )
            elif np.isfinite(emd_mean) and best_lora_snapshot is None:
                best_lora_snapshot = _snapshot_lora_adapters(train_model)

        if step_callback is not None:
            step_callback(step + 1, pred_rule_train, pred_rule_sampling)

    if config.enable_emd and best_lora_snapshot is not None:
        _load_lora_adapters(train_model, best_lora_snapshot)
        if config.task_type == "classification":
            state.best_nll_step = best_nll_step
            state.best_nll_value = best_nll_value
            print(
                f"  Restored best NLL checkpoint: step={best_nll_step}, "
                f"nll={best_nll_value:.6f}"
            )
        else:
            print("  Restored best checkpoint snapshot.")

    print(f"\n── Training complete ({sum(state.elapsed):.1f}s total) ──")
    return pred_rule_train, state


def train_and_merge_synthetic(
    x_tasks: np.ndarray,
    y_tasks: np.ndarray,
    categorical_x: list[bool],
    config: TrainConfig = TrainConfig(),
    **kwargs,
) -> tuple[PredictiveRule, TrainState]:
    """Synthetic task-batch training + LoRA merge."""
    pred_rule_train, state = train_synthetic(x_tasks, y_tasks, categorical_x, config, **kwargs)

    train_model = get_tabpfn_model(pred_rule_train)
    adapter_modules: dict[str, dict] = {}
    for name, module in train_model.named_modules():
        if isinstance(module, LoRALinear):
            adapter_modules[name] = {
                "lora_A": module.lora_A.detach().cpu(),
                "lora_B": module.lora_B.detach().cpu(),
                "r": int(module.r),
                "alpha": float(module.alpha),
                "scaling": float(module.scaling),
            }
    state.lora_adapter_state = {
        "format_version": 1,
        "modules": adapter_modules,
    }

    n_merged = merge_lora(train_model)
    print(f"  LoRA merged: {n_merged} modules into base model.")
    pred_rule_train.lock_weights()
    print("  Weights locked (will survive fit() calls).")
    return pred_rule_train, state
