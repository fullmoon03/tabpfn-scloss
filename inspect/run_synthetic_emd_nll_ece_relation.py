"""
run_synthetic_emd_nll_ece_relation.py

Relation experiment between EMD and NLL/ECE on synthetic tasks.

Design:
  - one task per prior family
  - multiple independently sampled query/context/rollout splits per task
  - context size is fixed
  - query-sampling variation induces EMD variation
  - baseline vs 1-step SC-tuned model overlay

Notes:
  - SC tuning uses the same pairwise SC loss structure as run_classification.py,
    but runs only a few optimizer steps with resampled tuning splits.
  - EMD semantics match the current run_classification.py convention:
      base sampling + belief-model evaluation
"""

# Example:
# python inspect/run_synthetic_emd_nll_ece_relation.py --setup-group standard_priors --no-include-tuned
# python inspect/run_synthetic_emd_nll_ece_relation.py --setup-group simple_linear --no-include-tuned
# python inspect/run_synthetic_emd_nll_ece_relation.py --setup-group simple_linear_ablations --no-include-tuned
# python inspect/run_synthetic_emd_nll_ece_relation.py --setup-group scm_variants --no-include-tuned
# python inspect/run_synthetic_emd_nll_ece_relation.py --setup-group nonlinear_link_setups --no-include-tuned
# python inspect/run_synthetic_emd_nll_ece_relation.py --setup-group nonlinear_link_setups --n-splits 50 --tuning-steps 5


from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import os
import sys

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from eval import (
    _compute_fixed_anchor_suite_eval_for_model_pair,
    compute_basic_metrics,
)
from generate_synthetic import MixtureConfig, generate_mixture_task, make_mixture_config
from lora import LoRAConfig, get_lora_params, get_tabpfn_model, inject_lora, merge_lora
from loss import sc_loss
from predictive_rule import ClassifierPredRule
from train import _build_prefix_continuation_map, _compute_query_marginals_for_ks


STANDARD_PRIOR_ORDER = ("gbdt", "scm", "smooth_mlp", "sparse_linear")
NONLINEAR_LINK_SETUP_ORDER = (
    "nonlinear_link_logistic",
    "nonlinear_link_gmm0",
    "nonlinear_link_gmm_neg1",
    "nonlinear_link_gmm_neg2",
)
SIMPLE_LINEAR_ABLATION_ORDER = (
    "sl_lower_margin",
    "sl_softer_labels",
    "sl_clean_linear",
    "sl_two_informative",
    "sl_six_informative",
    "sl_mild_imbalance",
    "sl_strong_imbalance",
    "sl_tiny_pairwise",
    "sl_corr_gaussian",
    "sl_corr_low_margin_clean",
)
SCM_VARIANT_ORDER = (
    "scm_parent1",
    "scm_parent3",
    "scm_alpha2_4",
)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Expected comma-separated integers, got: {text!r}")
    return tuple(int(p) for p in parts)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _clone_state_dict_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _relabel_to_contiguous(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, y_local = np.unique(np.asarray(y).astype(int), return_inverse=True)
    return classes.astype(np.int64), y_local.astype(np.int64)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _make_forced_prior_config(mode: str, prior_type: str) -> MixtureConfig:
    cfg = make_mixture_config(mode)
    cfg.p_gbdt = 1.0 if prior_type == "gbdt" else 0.0
    cfg.p_scm = 1.0 if prior_type == "scm" else 0.0
    cfg.p_smooth_mlp = 1.0 if prior_type == "smooth_mlp" else 0.0
    cfg.p_sparse_linear = 1.0 if prior_type == "sparse_linear" else 0.0
    cfg.p_nonlinear_link = 0.0
    return cfg


def _make_simple_linear_ablation_config(setup_name: str) -> MixtureConfig:
    cfg = make_mixture_config("simple_linear")
    cfg.mode_name = str(setup_name)

    if setup_name == "sl_lower_margin":
        cfg.logit_scale_min = 1.6
        cfg.logit_scale_max = 1.6
        return cfg

    if setup_name == "sl_softer_labels":
        cfg.temperature_min = 0.95
        cfg.temperature_max = 0.95
        cfg.deterministic_label_prob = 0.90
        cfg.label_noise_min = 0.0
        cfg.label_noise_max = 0.0
        return cfg

    if setup_name == "sl_clean_linear":
        cfg.label_noise_min = 0.0
        cfg.label_noise_max = 0.0
        cfg.deterministic_label_prob = 1.0
        cfg.logit_scale_min = 2.0
        cfg.logit_scale_max = 2.0
        return cfg

    if setup_name == "sl_two_informative":
        cfg.informative_min = 2
        cfg.informative_max = 2
        cfg.logit_scale_min = 2.0
        cfg.logit_scale_max = 2.0
        return cfg

    if setup_name == "sl_six_informative":
        cfg.informative_min = 6
        cfg.informative_max = 6
        return cfg

    if setup_name == "sl_mild_imbalance":
        cfg.class_prior_mode = "dirichlet"
        cfg.dirichlet_alpha_choices = (1.5, 2.0)
        return cfg

    if setup_name == "sl_strong_imbalance":
        cfg.class_prior_mode = "dirichlet"
        cfg.dirichlet_alpha_choices = (0.5, 0.8)
        return cfg

    if setup_name == "sl_tiny_pairwise":
        cfg.pairwise_interaction_prob = 0.03
        return cfg

    if setup_name == "sl_corr_gaussian":
        cfg.feature_mode = "mixed"
        cfg.p_correlated_gaussian = 1.0
        cfg.p_independent_hetero = 0.0
        cfg.p_heavy_tail = 0.0
        return cfg

    if setup_name == "sl_corr_low_margin_clean":
        cfg.feature_mode = "mixed"
        cfg.p_correlated_gaussian = 1.0
        cfg.p_independent_hetero = 0.0
        cfg.p_heavy_tail = 0.0
        cfg.informative_min = 3
        cfg.informative_max = 3
        cfg.logit_scale_min = 1.7
        cfg.logit_scale_max = 1.7
        cfg.temperature_min = 0.85
        cfg.temperature_max = 0.85
        cfg.label_noise_min = 0.0
        cfg.label_noise_max = 0.0
        cfg.deterministic_label_prob = 1.0
        cfg.class_prior_mode = "uniform"
        return cfg

    raise ValueError(f"Unknown simple_linear ablation setup: {setup_name}")


def _make_scm_variant_config(setup_name: str, synthetic_mode: str) -> MixtureConfig:
    cfg = _make_forced_prior_config(synthetic_mode, "scm")
    cfg.mode_name = str(setup_name)

    if setup_name == "scm_parent1":
        cfg.scm_max_parents = 3
        cfg.class_prior_mode = "uniform"
        return cfg

    if setup_name == "scm_parent3":
        cfg.scm_max_parents = 3
        cfg.dirichlet_alpha_choices = (2.0, 4.0)
        return cfg

    if setup_name == "scm_alpha2_4":
        cfg.scm_max_parents = 4
        return cfg

    raise ValueError(f"Unknown scm variant setup: {setup_name}")


def _get_setup_specs(
    *,
    setup_group: str,
    synthetic_mode: str,
    priors_raw: str,
) -> list[tuple[str, MixtureConfig]]:
    group = str(setup_group).strip().lower()
    if group == "standard_priors":
        priors = tuple(p.strip() for p in priors_raw.split(",") if p.strip())
        for prior in priors:
            if prior not in STANDARD_PRIOR_ORDER:
                raise ValueError(f"Unknown prior: {prior}")
        return [(prior, _make_forced_prior_config(synthetic_mode, prior)) for prior in priors]
    if group == "simple_linear":
        return [("simple_linear", make_mixture_config("simple_linear"))]
    if group == "simple_linear_ablations":
        return [(name, _make_simple_linear_ablation_config(name)) for name in SIMPLE_LINEAR_ABLATION_ORDER]
    if group == "scm_variants":
        return [(name, _make_scm_variant_config(name, synthetic_mode)) for name in SCM_VARIANT_ORDER]
    if group == "nonlinear_link_setups":
        return [(name, make_mixture_config(name)) for name in NONLINEAR_LINK_SETUP_ORDER]
    raise ValueError(
        f"Unknown setup_group: {setup_group}. "
        "Expected one of: standard_priors, simple_linear, simple_linear_ablations, scm_variants, nonlinear_link_setups."
    )


def _generate_setup_task(
    *,
    setup_name: str,
    cfg: MixtureConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    x, y, meta = generate_mixture_task(cfg, rng)
    if setup_name in STANDARD_PRIOR_ORDER and str(meta.get("prior_type")) != setup_name:
        raise RuntimeError(
            f"Forced prior generation failed: expected {setup_name}, got {meta.get('prior_type')}"
        )
    if setup_name == "simple_linear" or setup_name in SIMPLE_LINEAR_ABLATION_ORDER:
        if str(meta.get("prior_type")) != "sparse_linear":
            raise RuntimeError(
                f"Expected sparse_linear prior for setup {setup_name}, got {meta.get('prior_type')}"
            )
    if setup_name in SCM_VARIANT_ORDER:
        if str(meta.get("prior_type")) != "scm":
            raise RuntimeError(
                f"Expected scm prior for setup {setup_name}, got {meta.get('prior_type')}"
            )
    if setup_name in NONLINEAR_LINK_SETUP_ORDER:
        if str(meta.get("prior_type")) != "nonlinear_link":
            raise RuntimeError(
                f"Expected nonlinear_link prior for setup {setup_name}, got {meta.get('prior_type')}"
            )
    return x.astype(np.float32), y.astype(np.int64), meta


def _sample_valid_split(
    *,
    x_task: np.ndarray,
    y_task: np.ndarray,
    rng: np.random.Generator,
    context_size: int,
    query_pool_size: int,
    require_all_classes_in_context: bool,
    max_tries: int = 256,
) -> dict[str, np.ndarray]:
    n = len(y_task)
    rollout_size = n - context_size - query_pool_size
    if rollout_size <= 0:
        raise ValueError(
            f"Invalid split sizes for task with n={n}: context={context_size}, query={query_pool_size}"
        )
    global_classes = np.unique(y_task)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        idx_ctx = perm[:context_size]
        idx_query = perm[context_size : context_size + query_pool_size]
        idx_roll = perm[context_size + query_pool_size :]
        y_ctx = y_task[idx_ctx]
        if require_all_classes_in_context and len(np.unique(y_ctx)) < len(global_classes):
            continue
        if len(np.unique(y_ctx)) < 2:
            continue
        return {
            "idx_context": idx_ctx.astype(int),
            "idx_query": idx_query.astype(int),
            "idx_rollout": idx_roll.astype(int),
        }
    raise RuntimeError(
        "Failed to sample a valid query/context/rollout split satisfying context class constraints."
    )


def _build_eval_splits(
    *,
    x_task: np.ndarray,
    y_task: np.ndarray,
    n_splits: int,
    seed: int,
    context_size: int,
    query_pool_size: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    splits: list[dict[str, Any]] = []
    for split_id in range(int(n_splits)):
        split_idx = _sample_valid_split(
            x_task=x_task,
            y_task=y_task,
            rng=rng,
            context_size=context_size,
            query_pool_size=query_pool_size,
            require_all_classes_in_context=False,
        )
        splits.append(
            {
                "query_sample_id": split_id,
                "idx_context": split_idx["idx_context"],
                "idx_query": split_idx["idx_query"],
                "idx_rollout": split_idx["idx_rollout"],
            }
        )
    return splits


def _build_tuning_split(
    *,
    x_task: np.ndarray,
    y_task: np.ndarray,
    seed: int,
    context_size: int,
    query_pool_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return _sample_valid_split(
        x_task=x_task,
        y_task=y_task,
        rng=rng,
        context_size=context_size,
        query_pool_size=query_pool_size,
        require_all_classes_in_context=True,
    )


def _evaluate_anchor_metrics(
    *,
    model_state_dict: dict[str, torch.Tensor] | None,
    categorical_x: list[bool],
    n_estimators: int,
    x_context: np.ndarray,
    y_context_global: np.ndarray,
    x_query: np.ndarray,
    y_query_global: np.ndarray,
    global_num_classes: int,
) -> dict[str, float]:
    local_classes, y_ctx_local = _relabel_to_contiguous(y_context_global)
    pred_rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    if model_state_dict is not None:
        pred_rule._locked_state_dict = _clone_state_dict_cpu(model_state_dict)
    pred_rule.fit(x_context, y_ctx_local)
    with torch.no_grad():
        probs_local = pred_rule.get_belief_torch(x_query, x_context, y_ctx_local).cpu().numpy()
    probs_global = np.zeros((len(x_query), int(global_num_classes)), dtype=np.float64)
    probs_global[:, local_classes] = probs_local[:, : len(local_classes)]
    probs_global = probs_global / np.clip(probs_global.sum(axis=1, keepdims=True), 1e-12, None)
    m = compute_basic_metrics(probs_global, np.asarray(y_query_global).astype(int))
    return {
        "accuracy": float(m.accuracy),
        "nll": float(m.nll),
        "ece": float(m.ece),
    }


def _evaluate_anchor_emd(
    *,
    belief_model_state_dict: dict[str, torch.Tensor] | None,
    categorical_x: list[bool],
    n_estimators: int,
    x_context: np.ndarray,
    y_context_global: np.ndarray,
    x_query_emd: np.ndarray,
    y_query_emd_global: np.ndarray,
    x_rollout_pool: np.ndarray,
    global_num_classes: int,
    prefix_depths: tuple[int, ...],
    k_values: tuple[int, ...],
    continuation_depth: int,
    n_continuations: int,
    fixed_rollout_key_seed: int,
) -> float:
    local_classes, y_ctx_local = _relabel_to_contiguous(y_context_global)
    key = jax.random.PRNGKey(int(fixed_rollout_key_seed))
    fixed_rollout_keys: dict[tuple[int, int], jax.Array] = {}
    for prefix_depth in prefix_depths:
        key, subkey = jax.random.split(key)
        fixed_rollout_keys[(0, int(prefix_depth))] = subkey

    emd_mean, _emd_std, _cov, _acc, _nll, _ece, _ = _compute_fixed_anchor_suite_eval_for_model_pair(
        key=key,
        categorical_x=categorical_x,
        n_estimators=n_estimators,
        sampling_model_state_dict=None,
        belief_model_state_dict=belief_model_state_dict,
        anchor_contexts=[(x_context, y_ctx_local)],
        anchor_context_class_ids=[local_classes],
        anchor_query_banks_x=[x_query_emd],
        anchor_query_banks_y=[np.asarray(y_query_emd_global).astype(int)],
        global_num_classes=int(global_num_classes),
        prefix_depths=prefix_depths,
        k_values=k_values,
        continuation_depth=int(continuation_depth),
        n_continuations=int(n_continuations),
        fixed_rollout_keys=fixed_rollout_keys,
        anchor_rollout_pools=[np.asarray(x_rollout_pool, dtype=np.float32)],
    )
    return float(emd_mean)


def _run_single_step_sc_tuning(
    *,
    x_task: np.ndarray,
    y_task: np.ndarray,
    categorical_x: list[bool],
    device: str,
    n_estimators: int,
    lr: float,
    weight_decay: float,
    lora_r: int,
    lora_alpha: float,
    n_continuations: int,
    continuation_depth: int,
    queries_per_episode: int,
    num_pairs_per_query: int,
    k1_range: tuple[int, int],
    k2_range: tuple[int, int],
    prefix_depths: tuple[int, ...],
    seed: int,
    tuning_steps: int,
    context_size: int,
    query_pool_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    split_rng = np.random.default_rng(int(seed))
    init_split = _sample_valid_split(
        x_task=x_task,
        y_task=y_task,
        rng=split_rng,
        context_size=int(context_size),
        query_pool_size=int(query_pool_size),
        require_all_classes_in_context=True,
    )
    x_ctx_init = np.asarray(x_task[init_split["idx_context"]], dtype=np.float32)
    y_ctx_init_global = np.asarray(y_task[init_split["idx_context"]]).astype(int)
    _, y_ctx_init_local = _relabel_to_contiguous(y_ctx_init_global)
    pred_rule_sampling = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    pred_rule_train = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    pred_rule_sampling.fit(x_ctx_init, y_ctx_init_local)
    pred_rule_train.fit(x_ctx_init, y_ctx_init_local)

    train_model = get_tabpfn_model(pred_rule_train)
    lora_config = LoRAConfig(
        r=int(lora_r),
        alpha=float(lora_alpha),
        target_layers=None,
        include_decoder=False,
    )
    lora_modules = inject_lora(train_model, lora_config)
    optimizer = torch.optim.AdamW(
        get_lora_params(lora_modules),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    sc_k1_lo, sc_k1_hi = int(k1_range[0]), int(k1_range[1])
    sc_k2_lo, sc_k2_hi = int(k2_range[0]), int(k2_range[1])
    key = jax.random.PRNGKey(int(seed))
    device_obj = torch.device(device)
    train_model.to(device_obj)
    train_model.train()
    pair_losses: list[float] = []
    tuned_conf_vals: list[float] = []
    for _ in range(int(tuning_steps)):
        step_split = _sample_valid_split(
            x_task=x_task,
            y_task=y_task,
            rng=split_rng,
            context_size=int(context_size),
            query_pool_size=int(query_pool_size),
            require_all_classes_in_context=True,
        )
        x_ctx = np.asarray(x_task[step_split["idx_context"]], dtype=np.float32)
        y_ctx_global = np.asarray(y_task[step_split["idx_context"]]).astype(int)
        x_qpool = np.asarray(x_task[step_split["idx_query"]], dtype=np.float32)
        x_roll_pool = np.asarray(x_task[step_split["idx_rollout"]], dtype=np.float32)
        _, y_ctx_local = _relabel_to_contiguous(y_ctx_global)
        pred_rule_sampling.fit(x_ctx, y_ctx_local)
        pred_rule_train.fit(x_ctx, y_ctx_local)

        optimizer.zero_grad(set_to_none=True)

        key, k_q = jax.random.split(key)
        if len(x_qpool) >= int(queries_per_episode):
            q_idx = np.asarray(
                jax.random.choice(k_q, len(x_qpool), shape=(int(queries_per_episode),), replace=False)
            ).astype(int)
        else:
            q_idx = np.asarray(
                jax.random.choice(k_q, len(x_qpool), shape=(int(queries_per_episode),), replace=True)
            ).astype(int)
        q_episode = x_qpool[q_idx]

        conts_by_prefix, key = _build_prefix_continuation_map(
            key=key,
            pred_rule_sampling=pred_rule_sampling,
            x_context=x_ctx,
            y_context=y_ctx_local,
            prefix_depths=tuple(int(v) for v in prefix_depths),
            continuation_depth=int(continuation_depth),
            n_continuations=int(n_continuations),
            fixed_rollout_keys=None,
            x_sampling_pool=x_roll_pool,
            x_sample_without_replacement=True,
        )

        for prefix_depth in prefix_depths:
            conts = conts_by_prefix[int(prefix_depth)]
            for q_val in q_episode:
                key, k_pair_k1 = jax.random.split(key)
                key, k_pair_k2 = jax.random.split(key)
                sampled_k1 = np.asarray(
                    jax.random.randint(
                        k_pair_k1,
                        shape=(int(num_pairs_per_query),),
                        minval=sc_k1_lo,
                        maxval=sc_k1_hi + 1,
                    )
                ).astype(int)
                sampled_k2 = np.asarray(
                    jax.random.randint(
                        k_pair_k2,
                        shape=(int(num_pairs_per_query),),
                        minval=sc_k2_lo,
                        maxval=sc_k2_hi + 1,
                    )
                ).astype(int)
                sampled_pairs = tuple(
                    (int(k1), int(k2))
                    for k1, k2 in zip(sampled_k1.tolist(), sampled_k2.tolist())
                )
                ks_for_query = tuple(sorted({int(k) for pair in sampled_pairs for k in pair}))
                no_grad_keys_for_query = (
                    {int(k2) for _, k2 in sampled_pairs}
                    .difference({int(k1) for k1, _ in sampled_pairs})
                )
                p_hat = _compute_query_marginals_for_ks(
                    pred_rule_train=pred_rule_train,
                    continuations=conts,
                    ks=ks_for_query,
                    x_query=np.atleast_2d(q_val),
                    no_grad_keys=no_grad_keys_for_query,
                )
                l_sc = sc_loss(p_by_k=p_hat, sampled_pairs=sampled_pairs)
                micro_loss = l_sc / (float(len(prefix_depths)) * float(len(q_episode)))
                micro_loss.backward()
                pair_losses.append(float(l_sc.detach().cpu().item()))
                conf_k = min(int(k1) for k1, _ in sampled_pairs)
                p_conf = p_hat[conf_k] / p_hat[conf_k].sum().clamp_min(1e-8)
                tuned_conf_vals.append(float(p_conf.max().detach().cpu().item()))

        optimizer.step()
    merged_count = merge_lora(train_model)
    if merged_count <= 0:
        raise RuntimeError("Expected merged LoRA modules after single-step tuning.")
    merged_state = _clone_state_dict_cpu(train_model.state_dict())
    return merged_state, {
        "sc_loss_mean": float(np.mean(pair_losses)) if pair_losses else float("nan"),
        "conf_tuned_mean": float(np.mean(tuned_conf_vals)) if tuned_conf_vals else float("nan"),
        "tuning_steps": int(tuning_steps),
    }


def _plot_scatter(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    setup_names: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(len(setup_names), 2, figsize=(12, 4 * len(setup_names)), squeeze=False)
    color_map = {"baseline": "#1f77b4", "tuned": "#d62728"}
    marker_map = {"baseline": "o", "tuned": "s"}
    for row_i, setup_name in enumerate(setup_names):
        task_rows = [r for r in rows if r["setup_name"] == setup_name]
        for col_i, metric_name in enumerate(("nll", "ece")):
            ax = axes[row_i][col_i]
            for model_name in ("baseline", "tuned"):
                model_rows = [r for r in task_rows if r["model_name"] == model_name]
                if not model_rows:
                    continue
                ax.scatter(
                    [r["emd"] for r in model_rows],
                    [r[metric_name] for r in model_rows],
                    s=50,
                    alpha=0.82,
                    c=color_map[model_name],
                    marker=marker_map[model_name],
                    edgecolors="black",
                    linewidths=0.4,
                    label=model_name if row_i == 0 and col_i == 0 else None,
                )
            ax.set_title(f"{setup_name}: EMD vs {metric_name.upper()}")
            ax.set_xlabel("EMD")
            ax.set_ylabel(metric_name.upper())
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=min(4, len(handles)), frameon=False)
    fig.suptitle("Synthetic EMD vs NLL/ECE relation\nunder query-sampling variation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["setup_name"]), str(row["model_name"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (setup_name, model_name), group_rows in sorted(grouped.items()):
        emd = np.asarray([float(r["emd"]) for r in group_rows], dtype=np.float64)
        nll = np.asarray([float(r["nll"]) for r in group_rows], dtype=np.float64)
        ece = np.asarray([float(r["ece"]) for r in group_rows], dtype=np.float64)
        summary.append(
            {
                "setup_name": setup_name,
                "prior_type": str(group_rows[0]["prior_type"]) if group_rows else "",
                "model_name": model_name,
                "n_points": len(group_rows),
                "emd_mean": float(np.mean(emd)),
                "nll_mean": float(np.mean(nll)),
                "ece_mean": float(np.mean(ece)),
                "emd_vs_nll_spearman": _spearman_corr(emd, nll),
                "emd_vs_nll_pearson": _pearson_corr(emd, nll),
                "emd_vs_ece_spearman": _spearman_corr(emd, ece),
                "emd_vs_ece_pearson": _pearson_corr(emd, ece),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic experiment: relation between EMD and NLL/ECE."
    )
    parser.add_argument(
        "--setup-group",
        type=str,
        default="standard_priors",
        choices=("standard_priors", "simple_linear", "simple_linear_ablations", "scm_variants", "nonlinear_link_setups"),
    )
    parser.add_argument("--synthetic-mode", type=str, default="mixed_full")
    parser.add_argument("--priors", type=str, default="gbdt,scm,smooth_mlp,sparse_linear")
    parser.add_argument("--task-seed-base", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=200)
    parser.add_argument("--train-seed-offset", type=int, default=10_000)
    parser.add_argument("--split-seed-offset", type=int, default=1_000)
    parser.add_argument("--context-size", type=int, default=100)
    parser.add_argument("--query-pool-size", type=int, default=20)
    parser.add_argument(
        "--emd-query-count",
        type=int,
        default=20,
        help="Deprecated in current script; EMD now uses the full query pool.",
    )
    parser.add_argument(
        "--prefix-depths",
        type=str,
        default="0",
        help="Deprecated in current script; EMD prefix depth is fixed to 0.",
    )
    parser.add_argument("--k-values", type=str, default="3,5,7,9")
    parser.add_argument("--n-continuations", type=int, default=8)
    parser.add_argument("--continuation-depth", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute and save baseline points as well as tuned points.",
    )
    parser.add_argument(
        "--include-tuned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run SC tuning and compute tuned points.",
    )
    parser.add_argument("--tuning-lr", type=float, default=1e-3)
    parser.add_argument("--tuning-steps", type=int, default=5)
    parser.add_argument("--tuning-weight-decay", type=float, default=0.01)
    parser.add_argument("--tuning-queries-per-episode", type=int, default=3)
    parser.add_argument("--tuning-num-pairs-per-query", type=int, default=4)
    parser.add_argument("--tuning-k1-range", type=str, default="1,5")
    parser.add_argument("--tuning-k2-range", type=str, default="6,10")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--save-dir", type=str, default="synthetic_emd_relation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_specs = _get_setup_specs(
        setup_group=str(args.setup_group),
        synthetic_mode=str(args.synthetic_mode),
        priors_raw=str(args.priors),
    )
    setup_names = tuple(name for name, _ in setup_specs)
    prefix_depths = (0,)
    k_values = _parse_int_tuple(args.k_values)
    tuning_k1_range = _parse_int_tuple(args.tuning_k1_range)
    tuning_k2_range = _parse_int_tuple(args.tuning_k2_range)
    if len(tuning_k1_range) != 2 or len(tuning_k2_range) != 2:
        raise ValueError("tuning-k1-range and tuning-k2-range must have two integers.")

    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    results_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for setup_idx, (setup_name, cfg) in enumerate(setup_specs):
        task_seed = int(args.task_seed_base) + setup_idx
        x_task, y_task, meta = _generate_setup_task(
            setup_name=setup_name,
            cfg=cfg,
            seed=task_seed,
        )
        prior_type = str(meta.get("prior_type", "unknown"))
        d = int(x_task.shape[1])
        categorical_x = [False] * d
        global_num_classes = int(np.max(y_task)) + 1

        eval_splits = _build_eval_splits(
            x_task=x_task,
            y_task=y_task,
            n_splits=int(args.n_splits),
            seed=task_seed + int(args.split_seed_offset),
            context_size=int(args.context_size),
            query_pool_size=int(args.query_pool_size),
        )
        print("\n" + "=" * 80)
        print(f"TASK {setup_idx}: setup={setup_name}, prior={prior_type}, mode={cfg.mode_name}, seed={task_seed}")
        print("=" * 80)
        print(
            f"  Shape: n={len(y_task)}, d={d}, classes={global_num_classes}, "
            f"label_noise={meta.get('label_noise')}, temperature={meta.get('temperature')}"
        )
        print(f"  Baseline evaluation: {'on' if args.include_baseline else 'off'}")
        print(f"  Tuned evaluation: {'on' if args.include_tuned else 'off'}")

        tuned_state: dict[str, torch.Tensor] | None = None
        if args.include_tuned:
            tuned_state, tuning_info = _run_single_step_sc_tuning(
                x_task=x_task,
                y_task=y_task,
                categorical_x=categorical_x,
                device="cuda" if torch.cuda.is_available() else "cpu",
                n_estimators=int(args.n_estimators),
                lr=float(args.tuning_lr),
                weight_decay=float(args.tuning_weight_decay),
                lora_r=int(args.lora_r),
                lora_alpha=float(args.lora_alpha),
                n_continuations=int(args.n_continuations),
                continuation_depth=int(args.continuation_depth),
                queries_per_episode=int(args.tuning_queries_per_episode),
                num_pairs_per_query=int(args.tuning_num_pairs_per_query),
                k1_range=(int(tuning_k1_range[0]), int(tuning_k1_range[1])),
                k2_range=(int(tuning_k2_range[0]), int(tuning_k2_range[1])),
                prefix_depths=prefix_depths,
                seed=task_seed + int(args.train_seed_offset) + 1,
                tuning_steps=int(args.tuning_steps),
                context_size=int(args.context_size),
                query_pool_size=int(args.query_pool_size),
            )
            tuning_rows.append(
                {
                    "setup_name": setup_name,
                    "prior_type": prior_type,
                    "task_seed": task_seed,
                    "synthetic_mode": cfg.mode_name,
                    "sc_loss_mean": tuning_info["sc_loss_mean"],
                    "conf_tuned_mean": tuning_info["conf_tuned_mean"],
                    "tuning_steps": tuning_info["tuning_steps"],
                }
            )
            print(
                f"  {tuning_info['tuning_steps']}-step tuning: "
                f"sc_loss={tuning_info['sc_loss_mean']:.4f}, "
                f"conf={tuning_info['conf_tuned_mean']:.4f}"
            )

        for split in eval_splits:
            split_id = int(split["query_sample_id"])
            idx_ctx = split["idx_context"]
            idx_query = split["idx_query"]
            idx_roll = split["idx_rollout"]
            x_ctx = np.asarray(x_task[idx_ctx], dtype=np.float32)
            y_ctx = np.asarray(y_task[idx_ctx]).astype(int)
            x_query_full = np.asarray(x_task[idx_query], dtype=np.float32)
            y_query_full = np.asarray(y_task[idx_query]).astype(int)
            x_query_emd = x_query_full
            y_query_emd = y_query_full
            x_roll_pool = np.asarray(x_task[idx_roll], dtype=np.float32)
            if args.include_baseline:
                baseline_metrics = _evaluate_anchor_metrics(
                    model_state_dict=None,
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_ctx,
                    y_context_global=y_ctx,
                    x_query=x_query_full,
                    y_query_global=y_query_full,
                    global_num_classes=global_num_classes,
                )
                baseline_emd = _evaluate_anchor_emd(
                    belief_model_state_dict=None,
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_ctx,
                    y_context_global=y_ctx,
                    x_query_emd=x_query_emd,
                    y_query_emd_global=y_query_emd,
                    x_rollout_pool=x_roll_pool,
                    global_num_classes=global_num_classes,
                    prefix_depths=prefix_depths,
                    k_values=k_values,
                    continuation_depth=int(args.continuation_depth),
                    n_continuations=int(args.n_continuations),
                    fixed_rollout_key_seed=task_seed * 1000 + split_id,
                )
                results_rows.append(
                    {
                        "setup_name": setup_name,
                        "prior_type": prior_type,
                        "task_seed": task_seed,
                        "synthetic_mode": cfg.mode_name,
                        "model_name": "baseline",
                        "query_sample_id": split_id,
                        "emd": baseline_emd,
                        "accuracy": baseline_metrics["accuracy"],
                        "nll": baseline_metrics["nll"],
                        "ece": baseline_metrics["ece"],
                    }
                )

            if args.include_tuned and tuned_state is not None:
                tuned_metrics = _evaluate_anchor_metrics(
                    model_state_dict=tuned_state,
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_ctx,
                    y_context_global=y_ctx,
                    x_query=x_query_full,
                    y_query_global=y_query_full,
                    global_num_classes=global_num_classes,
                )
                tuned_emd = _evaluate_anchor_emd(
                    belief_model_state_dict=tuned_state,
                    categorical_x=categorical_x,
                    n_estimators=int(args.n_estimators),
                    x_context=x_ctx,
                    y_context_global=y_ctx,
                    x_query_emd=x_query_emd,
                    y_query_emd_global=y_query_emd,
                    x_rollout_pool=x_roll_pool,
                    global_num_classes=global_num_classes,
                    prefix_depths=prefix_depths,
                    k_values=k_values,
                    continuation_depth=int(args.continuation_depth),
                    n_continuations=int(args.n_continuations),
                    fixed_rollout_key_seed=task_seed * 1000 + split_id,
                )
                results_rows.append(
                    {
                        "setup_name": setup_name,
                        "prior_type": prior_type,
                        "task_seed": task_seed,
                        "synthetic_mode": cfg.mode_name,
                        "model_name": "tuned",
                        "query_sample_id": split_id,
                        "emd": tuned_emd,
                        "accuracy": tuned_metrics["accuracy"],
                        "nll": tuned_metrics["nll"],
                        "ece": tuned_metrics["ece"],
                    }
                )

        setup_rows = [r for r in results_rows if r["setup_name"] == setup_name]
        for model_name in ("baseline", "tuned"):
            model_rows = [r for r in setup_rows if r["model_name"] == model_name]
            if model_rows:
                print(
                    f"  [{model_name}] mean EMD={np.mean([r['emd'] for r in model_rows]):.4f}, "
                    f"NLL={np.mean([r['nll'] for r in model_rows]):.4f}, "
                    f"ECE={np.mean([r['ece'] for r in model_rows]):.4f}"
                )

    correlation_rows = _summarize_correlations(results_rows)
    details_csv = out_dir / f"synthetic_emd_relation_details_{tag}.csv"
    corr_csv = out_dir / f"synthetic_emd_relation_correlations_{tag}.csv"
    tuning_csv = out_dir / f"synthetic_emd_relation_tuning_{tag}.csv"
    plot_png = out_dir / f"synthetic_emd_relation_scatter_{tag}.png"

    _write_csv(details_csv, results_rows)
    _write_csv(corr_csv, correlation_rows)
    if tuning_rows:
        _write_csv(tuning_csv, tuning_rows)
    _plot_scatter(
        rows=results_rows,
        out_path=plot_png,
        setup_names=setup_names,
    )

    print("\nSaved:")
    print(f"  details:      {details_csv}")
    print(f"  correlations: {corr_csv}")
    if tuning_rows:
        print(f"  tuning:       {tuning_csv}")
    print(f"  plot:         {plot_png}")


if __name__ == "__main__":
    main()
