"""
run_classification.py
Synthetic task-batch SC training entrypoint.
"""
"""
python -u run_classification.py \
  --train-seed 42 \
  --eval-seed 43 \
  --adapter-path /home/boreum/project/tabpfn-scloss/synthetic_model/lora_adapter_generated_train_800_seed42_20260319_154103.pt \
  | tee 20260321_2e-4_train42_eval43.log


python run_classification.py \
  --synthetic-mode simple_linear \
  --save-steps 25,50,100

python run_classification.py \
  --synthetic-mode linear_mix \
  --train-seed 42 \
  --eval-seed 43

python run_classification.py \
  --synthetic-mode scm_mix \
  --train-seed 42 \
  --eval-seed 43

python run_classification.py \
  --synthetic-mode nonlinear_link_mix \
  --train-seed 42 \
  --eval-seed 43


"""


import argparse
import copy
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(__file__)
INSPECT_DIR = os.path.join(THIS_DIR, "inspect")
if INSPECT_DIR not in sys.path:
    sys.path.insert(0, INSPECT_DIR)

from eval import build_fixed_synthetic_anchor_suite, compute_basic_metrics
from generate_synthetic import MixtureConfig, generate_mixture_tensors, make_mixture_config
from predictive_rule import ClassifierPredRule
from lora import LoRALinear, get_tabpfn_model, merge_lora
from train import TrainConfig, train_and_merge_synthetic


@dataclass
class DatasetConfig:
    """Synthetic generator config."""
    synthetic_train_tasks: int = 800
    synthetic_eval_tasks: int = 10
    synthetic_train_seed: int = 42
    synthetic_eval_seed: int = 43


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in str(text).split(",") if v.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one integer value.")
    return tuple(vals)


def _parse_int_pair(text: str) -> tuple[int, int]:
    vals = _parse_int_tuple(text)
    if len(vals) != 2:
        raise ValueError(f"Expected exactly two integers, got {vals}")
    return int(vals[0]), int(vals[1])


def _load_adapter_modules(path: str) -> dict[str, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported adapter checkpoint format.")
    modules = ckpt.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Adapter checkpoint must contain dict field 'modules'.")
    return modules


def _extract_base_state_from_fitted_rule(pred_rule: ClassifierPredRule) -> dict[str, torch.Tensor]:
    model = get_tabpfn_model(pred_rule)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _build_alpha_scaled_state(
    *,
    base_state: dict[str, torch.Tensor],
    adapter_modules: dict[str, dict],
    alpha_scale: float,
) -> tuple[dict[str, torch.Tensor], int, int]:
    out = {k: v.clone() for k, v in base_state.items()}
    n_applied = 0
    n_skipped = 0

    for name, info in adapter_modules.items():
        weight_key = f"{name}.weight"
        if weight_key not in out:
            n_skipped += 1
            continue
        lora_A = info.get("lora_A")
        lora_B = info.get("lora_B")
        if not torch.is_tensor(lora_A) or not torch.is_tensor(lora_B):
            n_skipped += 1
            continue
        scaling = info.get("scaling")
        if scaling is None:
            r = float(info.get("r", max(1, lora_A.shape[1])))
            alpha = float(info.get("alpha", r))
            scaling = alpha / max(r, 1.0)
        delta = (lora_A.detach().cpu().float() @ lora_B.detach().cpu().float()).T
        delta = delta * float(scaling) * float(alpha_scale)
        base_w = out[weight_key]
        out[weight_key] = base_w + delta.to(dtype=base_w.dtype)
        n_applied += 1
    return out, n_applied, n_skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Train seed for synthetic train-task generation and training order.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=43,
        help="Eval seed for synthetic eval-task generation and fixed eval suite.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="",
        help="Optional LoRA adapter checkpoint to resume training from.",
    )
    parser.add_argument(
        "--synthetic-mode",
        type=str,
        default="mixed_full",
        choices=(
            "linear_mix",
            "scm_mix",
            "nonlinear_link_mix",
            "scm",
            "simple_linear",
            "mixed_full",
            "nonlinear_link",
            "nonlinear_link_logistic",
            "nonlinear_link_gmm0",
            "nonlinear_link_gmm_neg1",
            "nonlinear_link_gmm_neg2",
        ),
        help="Synthetic generator mode for train/eval dataset creation.",
    )
    parser.add_argument(
        "--sc-num-pairs-per-query",
        type=int,
        default=None,
        help="Number of (k1, k2) pairs sampled per query.",
    )
    parser.add_argument(
        "--sc-k1-range",
        type=str,
        default="",
        help="Inclusive k1 range as 'lo,hi'.",
    )
    parser.add_argument(
        "--sc-k2-range",
        type=str,
        default="",
        help="Inclusive k2 range as 'lo,hi'.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override.",
    )
    parser.add_argument(
        "--lr-decay-after-step",
        type=int,
        default=None,
        help="Step index after which LR decay applies.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=None,
        help="Learning rate decay factor.",
    )
    parser.add_argument(
        "--save-steps",
        type=str,
        default="",
        help="Comma-separated training steps to save as additional checkpoints.",
    )
    parser.add_argument(
        "--save-best-emd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the restored best-EMD checkpoint after training (default: enabled).",
    )
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    """Seed numpy/torch for reproducible experiment runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slugify_dataset_name(name: str) -> str:
    """Dataset name -> filesystem-friendly slug."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
    return s if s else "dataset"


def generate_synthetic_task_dataset(
    *,
    n_tasks: int,
    seed: int,
    split_name: str,
    mode: str = "mixed_full",
):
    """
    Generate synthetic task dataset in-memory.

    Returns:
      X: (N_task, N_point, d)
      y: (N_task, N_point)
    """
    cfg = make_mixture_config(mode)
    x_tasks, y_tasks, metas = generate_mixture_tensors(
        n_tasks=int(n_tasks),
        cfg=cfg,
        seed=int(seed),
        return_metadata=True,
    )
    if metas is None:
        raise RuntimeError("Expected metadata from synthetic generator.")

    if x_tasks.ndim != 3 or y_tasks.ndim != 2:
        raise ValueError(
            f"Expected X:(N_task,N_point,d), y:(N_task,N_point), "
            f"got X:{x_tasks.shape}, y:{y_tasks.shape}"
        )
    if x_tasks.shape[0] != y_tasks.shape[0] or x_tasks.shape[1] != y_tasks.shape[1]:
        raise ValueError(f"X/y shape mismatch: X:{x_tasks.shape}, y:{y_tasks.shape}")

    y_tasks_i = np.rint(y_tasks).astype(np.int64)
    n_tasks_out, n_points, d = x_tasks.shape
    n_classes = int(np.max(y_tasks_i)) + 1
    categorical_x = [False] * int(d)

    dataset_name = _slugify_dataset_name(
        f"generated_{split_name}_{cfg.mode_name}_{n_tasks_out}_seed{seed}"
    )
    prior_counts: dict[str, int] = {}
    setting_counts: dict[str, int] = {}
    for meta in metas:
        key = str(meta.get("prior_type", "unknown"))
        prior_counts[key] = prior_counts.get(key, 0) + 1
        setting_name = meta.get("setting_name")
        if setting_name is not None:
            skey = str(setting_name)
            setting_counts[skey] = setting_counts.get(skey, 0) + 1
    prior_text = ", ".join(f"{k}={v}" for k, v in sorted(prior_counts.items()))
    setting_text = ", ".join(f"{k}={v}" for k, v in sorted(setting_counts.items()))

    print("── Generating Synthetic task dataset ──")
    print(f"  Split: {split_name}")
    print(f"  Mode: {cfg.mode_name}")
    print(f"  Seed: {seed}")
    print(f"  Shape: tasks={n_tasks_out}, points/task={n_points}, dim={d}, classes={n_classes}")
    print(f"  Prior mix: {prior_text}")
    if setting_counts:
        print(f"  Setting mix: {setting_text}")
    return x_tasks, y_tasks_i, categorical_x, dataset_name


def _evaluate_basic_on_fixed_synthetic_anchors(
    *,
    pred_rule: ClassifierPredRule,
    anchor_contexts: list[tuple[np.ndarray, np.ndarray]],
    anchor_context_class_ids: list[np.ndarray],
    anchor_query_banks_x: list[np.ndarray],
    anchor_query_banks_y: list[np.ndarray],
    global_num_classes: int,
) -> dict[str, float]:
    """
    Evaluate Acc/NLL/ECE on fixed synthetic anchor suite.

    - each anchor uses fixed context C_a
    - metrics are computed on fixed query bank Q_a
    - final metrics are mean/std over anchors
    """
    n_anchor = len(anchor_contexts)
    if not (
        len(anchor_context_class_ids) == n_anchor
        and len(anchor_query_banks_x) == n_anchor
        and len(anchor_query_banks_y) == n_anchor
    ):
        raise ValueError("Fixed synthetic anchor suite length mismatch.")

    acc_vals: list[float] = []
    nll_vals: list[float] = []
    ece_vals: list[float] = []
    for a in range(n_anchor):
        x_ctx, y_ctx_local = anchor_contexts[a]
        local_classes = np.asarray(anchor_context_class_ids[a]).astype(int)
        x_q = np.asarray(anchor_query_banks_x[a], dtype=np.float32)
        y_q = np.asarray(anchor_query_banks_y[a]).astype(int)
        if len(x_q) == 0:
            continue

        pred_rule.fit(x_ctx, y_ctx_local)
        with torch.no_grad():
            probs_local = pred_rule.get_belief_torch(x_q, x_ctx, y_ctx_local).cpu().numpy()

        probs_global = np.zeros((len(x_q), int(global_num_classes)), dtype=np.float64)
        probs_global[:, local_classes] = probs_local
        probs_global = probs_global / np.clip(
            probs_global.sum(axis=1, keepdims=True), 1e-12, None
        )
        m = compute_basic_metrics(probs_global, y_q)
        acc_vals.append(float(m.accuracy))
        nll_vals.append(float(m.nll))
        ece_vals.append(float(m.ece))

    if len(acc_vals) == 0:
        raise ValueError("No anchor/query samples available for fixed-anchor evaluation.")

    acc = np.asarray(acc_vals, dtype=np.float64)
    nll = np.asarray(nll_vals, dtype=np.float64)
    ece = np.asarray(ece_vals, dtype=np.float64)
    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "nll_mean": float(nll.mean()),
        "nll_std": float(nll.std()),
        "ece_mean": float(ece.mean()),
        "ece_std": float(ece.std()),
        "n_anchors": float(len(acc_vals)),
    }


def _print_metrics(title: str, m: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Accuracy: {m['accuracy_mean']:.4f} ± {m['accuracy_std']:.4f}")
    print(f"  NLL:      {m['nll_mean']:.4f} ± {m['nll_std']:.4f}")
    print(f"  ECE:      {m['ece_mean']:.4f} ± {m['ece_std']:.4f}")
    print(f"  Anchors:  {int(m['n_anchors'])}")


def _snapshot_adapter_modules(model: torch.nn.Module) -> dict[str, dict]:
    modules: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            modules[name] = {
                "lora_A": module.lora_A.detach().cpu(),
                "lora_B": module.lora_B.detach().cpu(),
                "r": int(module.r),
                "alpha": float(module.alpha),
                "scaling": float(module.scaling),
            }
    return {
        "format_version": 1,
        "modules": modules,
    }


def _build_merged_state_dict(pred_rule: ClassifierPredRule) -> dict[str, torch.Tensor]:
    model_copy = copy.deepcopy(get_tabpfn_model(pred_rule)).cpu()
    merge_lora(model_copy)
    return {
        k: v.detach().cpu()
        for k, v in model_copy.state_dict().items()
    }


def _export_checkpoint_pair(
    *,
    pred_rule: ClassifierPredRule,
    save_dir: str,
    dataset_name: str,
    dataset_tag: str,
    suffix: str,
    adapter_state: dict | None = None,
) -> tuple[str, str]:
    model = get_tabpfn_model(pred_rule)
    adapter_path = os.path.join(save_dir, f"lora_adapter_{dataset_tag}_{suffix}.pt")
    torch.save(
        _snapshot_adapter_modules(model) if adapter_state is None else adapter_state,
        adapter_path,
    )

    merged_path = os.path.join(save_dir, f"merged_model_state_{dataset_tag}_{suffix}.pt")
    torch.save(
        {
            "format_version": 1,
            "dataset_name": str(dataset_name),
            "dataset_tag": dataset_tag,
            "state_dict": _build_merged_state_dict(pred_rule),
        },
        merged_path,
    )
    return adapter_path, merged_path


def run_experiment_synthetic(
    x_train_tasks: np.ndarray,
    y_train_tasks: np.ndarray,
    x_eval_tasks: np.ndarray,
    y_eval_tasks: np.ndarray,
    categorical_x: list[bool],
    config: TrainConfig,
    train_dataset_name: str = "synthetic_train",
    eval_dataset_name: str = "synthetic_eval",
    eval_task_seed: int = 43,
    initial_lora_adapter_modules: dict[str, dict] | None = None,
    save_steps: tuple[int, ...] = (),
    save_best_emd: bool = True,
    key=None,
):
    """Synthetic task-batch training entrypoint."""
    set_global_seeds(config.seed)
    dataset_tag = _slugify_dataset_name(train_dataset_name)
    if key is None:
        key = jax.random.PRNGKey(config.seed)

    # Build one fixed synthetic anchor suite for both:
    #   - EMD monitoring (inside training)
    #   - Acc/NLL/ECE evaluation (baseline + tuned)
    emd_prefix_depths = tuple(int(v) for v in config.emd_prefix_depths)
    suite_key = jax.random.PRNGKey(int(eval_task_seed))
    (
        anchor_contexts,
        anchor_contexts_global,
        anchor_context_class_ids,
        anchor_query_banks_x,
        anchor_query_banks_y,
        anchor_rollout_pools,
        anchor_fixed_rollout_keys,
        _,
    ) = build_fixed_synthetic_anchor_suite(
        key=suite_key,
        x_tasks=np.asarray(x_eval_tasks),
        y_tasks=np.asarray(y_eval_tasks),
        anchor_count=config.emd_anchor_count,
        context_size=config.emd_context_size,
        query_pool_size=config.sc_task_query_pool_size,
        queries_per_anchor=config.emd_queries_per_anchor,
        prefix_depths=emd_prefix_depths,
        fixed_rollout_paths=config.emd_fixed_rollout_paths,
    )
    global_num_classes = int(np.max(y_eval_tasks)) + 1

    print("\n" + "=" * 60)
    print("  BASELINE (before SC training) on fixed synthetic anchors")
    print("=" * 60)
    pred_rule_baseline = ClassifierPredRule(categorical_x, n_estimators=config.n_estimators)
    if initial_lora_adapter_modules is not None:
        x_ctx0, y_ctx0 = anchor_contexts[0]
        pred_rule_base_fit = ClassifierPredRule(categorical_x, n_estimators=config.n_estimators)
        pred_rule_base_fit.fit(x_ctx0, y_ctx0)
        base_state = _extract_base_state_from_fitted_rule(pred_rule_base_fit)
        merged_state, _n_applied, _n_skipped = _build_alpha_scaled_state(
            base_state=base_state,
            adapter_modules=initial_lora_adapter_modules,
            alpha_scale=1.0,
        )
        pred_rule_baseline._locked_state_dict = {
            k: v.clone().cpu() for k, v in merged_state.items()
        }
    baseline_metrics = _evaluate_basic_on_fixed_synthetic_anchors(
        pred_rule=pred_rule_baseline,
        anchor_contexts=anchor_contexts,
        anchor_context_class_ids=anchor_context_class_ids,
        anchor_query_banks_x=anchor_query_banks_x,
        anchor_query_banks_y=anchor_query_banks_y,
        global_num_classes=global_num_classes,
    )
    _print_metrics(f"BASELINE @ {eval_dataset_name}", baseline_metrics)

    print("\n" + "=" * 60)
    print("  SC TRAINING (SYNTHETIC TASK-BATCH)")
    print("=" * 60)

    key, train_key = jax.random.split(key)
    t0 = time.time()
    model_save_dir = "synthetic_model"
    os.makedirs(model_save_dir, exist_ok=True)
    model_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_steps_set = {int(step) for step in save_steps if int(step) > 0}
    saved_step_exports: list[tuple[int, str, str]] = []

    def _step_export_callback(step_i: int, pred_rule_train, _pred_rule_sampling) -> None:
        if int(step_i) not in save_steps_set:
            return
        adapter_path, merged_path = _export_checkpoint_pair(
            pred_rule=pred_rule_train,
            save_dir=model_save_dir,
            dataset_name=train_dataset_name,
            dataset_tag=dataset_tag,
            suffix=f"{model_ts}_step{int(step_i)}",
        )
        saved_step_exports.append((int(step_i), adapter_path, merged_path))
        print(f"  Saved step checkpoint: step={int(step_i)}")
        print(f"    adapter={adapter_path}")
        print(f"    merged={merged_path}")

    pred_rule_trained, train_state = train_and_merge_synthetic(
        x_train_tasks,
        y_train_tasks,
        categorical_x,
        config=config,
        key=train_key,
        emd_anchor_tasks_x=x_eval_tasks,
        emd_anchor_tasks_y=y_eval_tasks,
        emd_anchor_contexts=anchor_contexts,
        emd_anchor_contexts_global=anchor_contexts_global,
        emd_anchor_query_banks=anchor_query_banks_x,
        emd_anchor_rollout_pools=anchor_rollout_pools,
        emd_fixed_rollout_keys=anchor_fixed_rollout_keys,
        emd_anchor_context_class_ids=anchor_context_class_ids,
        emd_anchor_query_labels=anchor_query_banks_y,
        emd_global_num_classes=global_num_classes,
        initial_lora_adapter_modules=initial_lora_adapter_modules,
        step_callback=_step_export_callback if len(save_steps_set) > 0 else None,
    )
    train_time = time.time() - t0

    print("\n" + "=" * 60)
    print("  POST-TRAINING EVAL on fixed synthetic anchors")
    print("=" * 60)
    trained_metrics = _evaluate_basic_on_fixed_synthetic_anchors(
        pred_rule=pred_rule_trained,
        anchor_contexts=anchor_contexts,
        anchor_context_class_ids=anchor_context_class_ids,
        anchor_query_banks_x=anchor_query_banks_x,
        anchor_query_banks_y=anchor_query_banks_y,
        global_num_classes=global_num_classes,
    )
    _print_metrics(f"LOADED MODEL @ {eval_dataset_name}", trained_metrics)
    print("\n" + "-" * 60)
    print("  Delta (loaded - baseline)")
    print("-" * 60)
    print(
        f"  Accuracy: "
        f"{trained_metrics['accuracy_mean'] - baseline_metrics['accuracy_mean']:+.4f}"
    )
    print(
        f"  NLL:      "
        f"{trained_metrics['nll_mean'] - baseline_metrics['nll_mean']:+.4f}"
    )
    print(
        f"  ECE:      "
        f"{trained_metrics['ece_mean'] - baseline_metrics['ece_mean']:+.4f}"
    )

    print("\n── Training Summary ──")
    print(f"  Total time: {train_time:.1f}s")
    if len(train_state.losses) > 0:
        print(f"  Final loss: {train_state.losses[-1]:.4f}")
        print(
            f"  Loss curve (first/mid/last): "
            f"{train_state.losses[0]:.4f} → "
            f"{train_state.losses[len(train_state.losses)//2]:.4f} → "
            f"{train_state.losses[-1]:.4f}"
        )
    if train_state.best_emd_step is not None and train_state.best_emd_value is not None:
        print(
            f"  Best EMD checkpoint: step={train_state.best_emd_step}, "
            f"emd_mean={train_state.best_emd_value:.6f}"
        )

    print("\n── Model Export ──")
    if save_best_emd:
        final_adapter_state = getattr(train_state, "lora_adapter_state", None)
        adapter_path, merged_path = _export_checkpoint_pair(
            pred_rule=pred_rule_trained,
            save_dir=model_save_dir,
            dataset_name=train_dataset_name,
            dataset_tag=dataset_tag,
            suffix=model_ts,
            adapter_state=final_adapter_state,
        )
        print("  Best-EMD export: enabled")
        print(f"  Saved LoRA adapter: {adapter_path}")
        print(f"  Saved merged model state_dict: {merged_path}")
    else:
        print("  Best-EMD export: disabled")
    if len(saved_step_exports) > 0:
        print("  Step exports:")
        for step_i, adapter_path, merged_path in saved_step_exports:
            print(f"    step={step_i}:")
            print(f"      adapter={adapter_path}")
            print(f"      merged={merged_path}")

    if config.enable_emd:
        emd_csv_path = f"emd_history_{dataset_tag}.csv"
        with open(emd_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "emd_mean",
                "emd_std",
                "emd_coverage",
            ])
            for step_i, emd_mean, emd_std, cov in zip(
                train_state.emd_steps,
                train_state.emd_values,
                train_state.emd_stds,
                train_state.emd_coverage,
            ):
                writer.writerow([step_i, emd_mean, emd_std, cov])
        print("\n── EMD History ──")
        print(f"  Saved: {emd_csv_path}")

        if len(train_state.emd_steps) > 0:
            emd_curve_dir = "emd_curve"
            os.makedirs(emd_curve_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            emd_png_path = os.path.join(emd_curve_dir, f"emd_curve_{dataset_tag}_{ts}.png")
            plt.figure(figsize=(7, 4))
            plt.plot(
                train_state.emd_steps,
                train_state.emd_values,
                marker="o",
                linewidth=1.6,
                label=f"EMD k={tuple(int(v) for v in config.emd_k_values)}",
            )
            plt.xlabel("Training step")
            plt.ylabel("EMD")
            plt.title("EMD over training")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(emd_png_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Plot : {emd_png_path}")

    return train_state


if __name__ == "__main__":
    args = parse_args()
    train_seed = int(args.train_seed)
    dataset_config = DatasetConfig(
        synthetic_train_tasks=800,
        synthetic_eval_tasks=10,
        synthetic_train_seed=train_seed,
        synthetic_eval_seed=int(args.eval_seed),
    )

    config = TrainConfig(
        task_type="classification",
        n_estimators=1,
        seed=train_seed,
        continuation_depth=30,
        n_continuations=8,
        k_max=15,
        sc_context_size=100,
        sc_task_query_pool_size=20,
        sc_num_pairs_per_query=(
            int(args.sc_num_pairs_per_query) if args.sc_num_pairs_per_query is not None else 4
        ),
        sc_k1_range=_parse_int_pair(args.sc_k1_range) if args.sc_k1_range.strip() else (1, 9),
        sc_k2_range=_parse_int_pair(args.sc_k2_range) if args.sc_k2_range.strip() else (10, 15),
        sc_episodes_per_step=1,
        sc_queries_per_episode=3,
        enable_emd=True,
        emd_fill_every=5,
        emd_fixed_rollout_paths=True,
        emd_anchor_count=3,
        emd_context_size=100,
        emd_queries_per_anchor=4,
        emd_k_values=(3, 7, 11, 15),
        num_steps=100,
        lr=float(args.lr) if args.lr is not None else 2e-4,
        lr_decay_after_step=int(args.lr_decay_after_step) if args.lr_decay_after_step is not None else 20,
        lr_decay_factor=float(args.lr_decay_factor) if args.lr_decay_factor is not None else 1.0,
        grad_clip=1.0,
        lora_include_decoder=False,
        device="cuda",
    )
    save_steps = _parse_int_tuple(args.save_steps) if args.save_steps.strip() else tuple()

    set_global_seeds(config.seed)
    initial_lora_adapter_modules = None
    if args.adapter_path.strip():
        adapter_path = os.path.abspath(args.adapter_path.strip())
        if not os.path.isfile(adapter_path):
            raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
        initial_lora_adapter_modules = _load_adapter_modules(adapter_path)
        print(f"── Loaded resume adapter: {adapter_path}")
    print(
        f"── Seed setup: train_seed={train_seed}, "
        f"eval_task_seed={dataset_config.synthetic_eval_seed}"
    )
    print(f"── Synthetic mode: {args.synthetic_mode}")

    (
        x_train_tasks,
        y_train_tasks,
        categorical_x,
        train_dataset_name,
    ) = generate_synthetic_task_dataset(
        n_tasks=dataset_config.synthetic_train_tasks,
        seed=dataset_config.synthetic_train_seed,
        split_name="train",
        mode=args.synthetic_mode,
    )
    x_eval_tasks, y_eval_tasks, categorical_x_eval, eval_dataset_name = (
        generate_synthetic_task_dataset(
            n_tasks=dataset_config.synthetic_eval_tasks,
            seed=dataset_config.synthetic_eval_seed,
            split_name="eval",
            mode=args.synthetic_mode,
        )
    )
    if len(categorical_x_eval) != len(categorical_x):
        raise ValueError(
            "Train/eval synthetic feature dimension mismatch: "
            f"{len(categorical_x)} vs {len(categorical_x_eval)}"
        )

    run_experiment_synthetic(
        x_train_tasks=x_train_tasks,
        y_train_tasks=y_train_tasks,
        x_eval_tasks=x_eval_tasks,
        y_eval_tasks=y_eval_tasks,
        categorical_x=categorical_x,
        config=config,
        train_dataset_name=train_dataset_name,
        eval_dataset_name=eval_dataset_name,
        eval_task_seed=dataset_config.synthetic_eval_seed,
        initial_lora_adapter_modules=initial_lora_adapter_modules,
        save_steps=save_steps,
        save_best_emd=bool(args.save_best_emd),
        key=jax.random.PRNGKey(train_seed),
    )
