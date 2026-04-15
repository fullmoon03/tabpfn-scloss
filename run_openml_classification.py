from __future__ import annotations

"""
run_classification.py
OpenML Vehicle dataset 기준 분류 실험 엔트리포인트.
"""

import time
import csv
import os
import importlib.util
import re
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from predictive_rule import ClassifierPredRule


@dataclass
class RunConfig:
    """실험 실행/시각화 옵션."""
    # inspect/analyze_fixed_query_rollout_mc.py 호출 여부
    enable_rollout_compare: bool = False
    rollout_plot_before: bool = True  # enable/disable "before tuning" rollout grid
    rollout_base_n: int = 50
    rollout_depth: int = 30
    rollout_n_paths: int = 8
    rollout_n_queries: int = 6
    rollout_grid_rows: int = 2
    rollout_grid_cols: int = 3
    rollout_save_dir: str = "rollout_plots"
    eval_context_size: int = 100
    eval_context_repeats: int = 4


@dataclass
class DatasetConfig:
    """입력 데이터셋 로딩 설정."""
    openml_data_id: int = 54
    test_size: float = 0.25


def set_openml_global_seeds(seed: int) -> None:
    """Seed numpy/torch for reproducible experiment runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slugify_dataset_name(name: str) -> str:
    """Dataset name -> filesystem-friendly slug."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
    return s if s else "dataset"


def _encode_features_for_tabpfn(X: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    """
    OpenML feature matrix를 TabPFN 입력 형식으로 변환.

    - numeric column: float32 유지, categorical=False
    - non-numeric column: 문자열 범주 인덱스로 인코딩, categorical=True
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={X_arr.shape}")

    n_samples, n_features = X_arr.shape
    encoded_cols = []
    categorical_x: list[bool] = []
    for j in range(n_features):
        col = X_arr[:, j]
        if np.issubdtype(col.dtype, np.number):
            encoded_cols.append(col.astype(np.float32))
            categorical_x.append(False)
        else:
            # 문자열 기반 범주 인덱스 인코딩
            _, inv = np.unique(col.astype(str), return_inverse=True)
            encoded_cols.append(inv.astype(np.float32))
            categorical_x.append(True)

    X_enc = np.column_stack(encoded_cols).astype(np.float32)
    if X_enc.shape[0] != n_samples:
        raise RuntimeError("Feature encoding produced invalid sample count.")
    return X_enc, categorical_x


def load_openml_dataset_split(
    data_id: int,
    seed: int = 0,
    test_size: float = 0.25,
):
    """
    OpenML 분류 데이터셋 로드.

    Args:
        data_id: OpenML dataset id
        seed: random seed
        test_size: test split ratio

    Returns:
        X_train, X_test, y_train, y_test, categorical_x, le, dataset_name
    """
    print(f"── Loading OpenML dataset (ID={data_id}) ──")
    data = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    X, y_raw = data.data, data.target
    dataset_name = getattr(data, "name", f"openml_{data_id}")
    X_enc, categorical_x = _encode_features_for_tabpfn(X)

    # LabelEncoder: string labels → 0..C-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Classes: {le.classes_} → {list(range(len(le.classes_)))}")
    print(
        f"  Dataset: {dataset_name}, "
        f"{X_enc.shape[0]} samples, {X_enc.shape[1]} features, {len(le.classes_)} classes"
    )
    print(f"  Categorical features: {int(sum(categorical_x))}/{len(categorical_x)}")

    # Train/test split (stratified when possible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=test_size, random_state=seed, stratify=None
        )
        print("  [warn] stratified split failed; fallback to non-stratified split.")
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test, categorical_x, le, dataset_name


def load_vehicle_dataset(seed: int = 0):
    """기존 Vehicle 실험 호환용 wrapper."""
    return load_openml_dataset_split(data_id=54, seed=seed, test_size=0.25)


def _load_rollout_analyzer_module():
    """rollout 비교 분석 모듈 동적 import."""
    module_path = os.path.join(
        os.path.dirname(__file__), "inspect", "analyze_fixed_query_rollout_mc.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_fixed_query_rollout_mc", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load analyzer module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_fixed_eval_context_indices(
    *,
    n_train: int,
    context_size: int,
    repeats: int,
    seed: int,
) -> tuple[list[np.ndarray], int]:
    """Build fixed train-context index sets for repeated eval."""
    if repeats < 1:
        raise ValueError(f"eval_context_repeats must be >= 1, got {repeats}")
    if n_train < 1:
        raise ValueError("train split is empty.")
    ctx_size = int(np.clip(context_size, 1, n_train))
    idx_all = np.arange(n_train, dtype=int)
    rng = np.random.default_rng(seed)
    idx_sets: list[np.ndarray] = []
    for _ in range(repeats):
        idx = np.asarray(rng.choice(idx_all, size=ctx_size, replace=False), dtype=int)
        idx_sets.append(idx)
    return idx_sets, ctx_size


def _mean_basic_metrics(metrics_list: list[BasicMetrics]) -> BasicMetrics:
    """Average BasicMetrics over multiple runs."""
    from eval import BasicMetrics

    if len(metrics_list) == 0:
        raise ValueError("metrics_list must be non-empty")
    return BasicMetrics(
        accuracy=float(np.mean([m.accuracy for m in metrics_list])),
        nll=float(np.mean([m.nll for m in metrics_list])),
        ece=float(np.mean([m.ece for m in metrics_list])),
        n_samples=int(metrics_list[0].n_samples),
        n_bins=int(metrics_list[0].n_bins),
    )


def _evaluate_with_fixed_contexts(
    *,
    pred_rule,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    context_indices: list[np.ndarray],
) -> BasicMetrics:
    """Evaluate on test by averaging metrics across fixed train-context subsets."""
    from eval import evaluate_basic

    per_ctx: list[BasicMetrics] = []
    for idx in context_indices:
        m = evaluate_basic(
            pred_rule,
            x_train[idx], y_train[idx],
            x_test, y_test,
            use_torch=True,
            task_type="classification",
        )
        per_ctx.append(m)
    return _mean_basic_metrics(per_ctx)


def _snapshot_plain_state_from_train_rule(pred_rule_train) -> dict[str, torch.Tensor]:
    """
    Build a plain-model state_dict snapshot from a LoRA-injected training rule.

    This merges each LoRA module on-the-fly into `<module>.weight` and removes
    LoRA-specific keys so the snapshot can be loaded into a fresh plain TabPFN model.
    """
    from lora import LoRALinear, get_tabpfn_model

    model = get_tabpfn_model(pred_rule_train)
    raw_state = model.state_dict()
    lora_modules: dict[str, LoRALinear] = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }

    out: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        if key.endswith(".original.weight"):
            prefix = key[: -len(".original.weight")]
            lora_mod = lora_modules.get(prefix)
            if lora_mod is None:
                out[key] = value.detach().cpu().clone()
                continue
            with torch.no_grad():
                delta = (lora_mod.lora_A @ lora_mod.lora_B).T * float(lora_mod.scaling)
                merged_w = (lora_mod.original.weight + delta).detach().cpu().clone()
            out[f"{prefix}.weight"] = merged_w
            continue
        if key.endswith(".original.bias"):
            prefix = key[: -len(".original.bias")]
            out[f"{prefix}.bias"] = value.detach().cpu().clone()
            continue
        out[key] = value.detach().cpu().clone()
    return out


def run_experiment(
    X_train, X_test, y_train, y_test, categorical_x,
    config: TrainConfig,
    run_config: RunConfig | None = None,
    dataset_name: str = "dataset",
    class_names=None,
    key=None,
):
    """baseline -> train -> post-eval -> comparison."""
    from eval import EvalComparison, print_comparison
    from train import train_and_merge

    set_openml_global_seeds(config.seed)
    if run_config is None:
        run_config = RunConfig()
    dataset_tag = _slugify_dataset_name(dataset_name)
    if key is None:
        key = jax.random.PRNGKey(config.seed)
    rollout_query_indices = None
    if run_config.enable_rollout_compare:
        rng = np.random.default_rng(config.seed)
        replace = len(X_test) < run_config.rollout_n_queries
        rollout_query_indices = np.array(
            rng.choice(
                len(X_test),
                size=run_config.rollout_n_queries,
                replace=replace,
            ),
            dtype=int,
        )
        print(
            "  Fixed rollout queries (shared before/after): "
            f"{rollout_query_indices.tolist()}"
        )
    eval_context_indices, eval_context_size = _build_fixed_eval_context_indices(
        n_train=len(X_train),
        context_size=run_config.eval_context_size,
        repeats=run_config.eval_context_repeats,
        seed=config.seed,
    )
    print(
        "  Fixed eval contexts (shared baseline/train/post): "
        f"{len(eval_context_indices)} x {eval_context_size}"
    )

    # 1) Baseline evaluation
    print("\n" + "=" * 60)
    print("  BASELINE (before SC training)")
    print("=" * 60)

    pred_rule_baseline = ClassifierPredRule(
        categorical_x,
        n_estimators=config.n_estimators,
    )

    baseline_metrics = _evaluate_with_fixed_contexts(
        pred_rule=pred_rule_baseline,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        context_indices=eval_context_indices,
    )
    print(f"  Accuracy: {baseline_metrics.accuracy:.4f}")
    print(f"  NLL:      {baseline_metrics.nll:.4f}")
    print(f"  ECE:      {baseline_metrics.ece:.4f}")

    if run_config.enable_rollout_compare and run_config.rollout_plot_before:
        print("\n  Generating fixed-query rollout plots (before tuning)...")
        analyzer = _load_rollout_analyzer_module()
        before_rollout_res = analyzer.run_fixed_queries_rollout_grid_analysis(
            x_train=X_train,
            y_train=y_train,
            x_query_pool=X_test,
            query_indices=rollout_query_indices.tolist(),
            categorical_x=categorical_x,
            class_names=class_names,
            pred_rule_factory=lambda: ClassifierPredRule(
                categorical_x, n_estimators=config.n_estimators
            ),
            n_estimators=config.n_estimators,
            base_n=run_config.rollout_base_n,
            depth=run_config.rollout_depth,
            n_paths=run_config.rollout_n_paths,
            seed=config.seed,
            n_rows=run_config.rollout_grid_rows,
            n_cols=run_config.rollout_grid_cols,
            save_dir=run_config.rollout_save_dir,
            tag=f"{dataset_tag}_before_tuning",
            show=False,
        )
        before_src = before_rollout_res["paths"]["grid_plot_png"]
        before_dst = os.path.join(
            run_config.rollout_save_dir,
            f"belief_{dataset_tag}_before_tuning.png",
        )
        if os.path.abspath(before_src) != os.path.abspath(before_dst):
            os.replace(before_src, before_dst)


    # 2) Training
    print("\n" + "=" * 60)
    print("  SC TRAINING")
    print("=" * 60)

    # train과 이후 eval이 독립된 난수 시퀀스를 사용하도록 분리
    key, train_key = jax.random.split(key)

    def _step_metrics_callback(step_i: int, pred_rule_train, _pred_rule_sampling) -> None:
        if not config.enable_emd:
            return
        if (step_i % int(config.emd_fill_every)) != 0:
            return
        eval_rule_step = ClassifierPredRule(
            categorical_x,
            n_estimators=config.n_estimators,
        )
        eval_rule_step._locked_state_dict = _snapshot_plain_state_from_train_rule(
            pred_rule_train
        )
        step_metrics = _evaluate_with_fixed_contexts(
            pred_rule=eval_rule_step,
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test,
            context_indices=eval_context_indices,
        )
        print(
            f"  [MET {step_i:4d}] "
            f"acc={step_metrics.accuracy:.4f}, "
            f"nll={step_metrics.nll:.4f}, "
            f"ece={step_metrics.ece:.4f}"
        )

    t0 = time.time()
    pred_rule_trained, train_state = train_and_merge(
        X_train, y_train, categorical_x,
        config=config,
        key=train_key,
        emd_test_query_pool=X_test,
        emd_test_label_pool=y_test,
        step_callback=_step_metrics_callback,
    )
    train_time = time.time() - t0

    # 3) Post-training evaluation
    print("\n" + "=" * 60)
    print("  POST-TRAINING EVAL")
    print("=" * 60)

    trained_metrics = _evaluate_with_fixed_contexts(
        pred_rule=pred_rule_trained,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        context_indices=eval_context_indices,
    )
    print(f"  Accuracy: {trained_metrics.accuracy:.4f}")
    print(f"  NLL:      {trained_metrics.nll:.4f}")
    print(f"  ECE:      {trained_metrics.ece:.4f}")

    if run_config.enable_rollout_compare:
        print("\n  Generating fixed-query rollout plots (after tuning)...")
        analyzer = _load_rollout_analyzer_module()
        after_rollout_res = analyzer.run_fixed_queries_rollout_grid_analysis(
            x_train=X_train,
            y_train=y_train,
            x_query_pool=X_test,
            query_indices=rollout_query_indices.tolist(),
            categorical_x=categorical_x,
            class_names=class_names,
            pred_rule_factory=lambda: pred_rule_trained,
            n_estimators=config.n_estimators,
            base_n=run_config.rollout_base_n,
            depth=run_config.rollout_depth,
            n_paths=run_config.rollout_n_paths,
            seed=config.seed,
            n_rows=run_config.rollout_grid_rows,
            n_cols=run_config.rollout_grid_cols,
            save_dir=run_config.rollout_save_dir,
            tag=f"{dataset_tag}_after_tuning",
            show=False,
        )
        after_src = after_rollout_res["paths"]["grid_plot_png"]
        rollout_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        after_dst = os.path.join(
            run_config.rollout_save_dir,
            f"belief_{dataset_tag}_after_tuning_{rollout_ts}.png",
        )
        if os.path.abspath(after_src) != os.path.abspath(after_dst):
            os.replace(after_src, after_dst)

    # 4) Comparison
    comp = EvalComparison(
        before=baseline_metrics,
        after=trained_metrics,
    )
    print_comparison(comp)

    # Training summary
    print(f"\n── Training Summary ──")
    print(f"  Total time: {train_time:.1f}s")
    print(f"  Final loss: {train_state.losses[-1]:.4f}")
    print(f"  Loss curve (first/mid/last): "
          f"{train_state.losses[0]:.4f} → "
          f"{train_state.losses[len(train_state.losses)//2]:.4f} → "
          f"{train_state.losses[-1]:.4f}")
    if train_state.best_emd_step is not None and train_state.best_emd_value is not None:
        print(
            f"  Best EMD checkpoint: step={train_state.best_emd_step}, "
            f"emd_mean={train_state.best_emd_value:.6f}"
        )

    # Save LoRA adapter weights for reuse/transfer.
    model_save_dir = "vehicle_model"
    os.makedirs(model_save_dir, exist_ok=True)
    model_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_path = os.path.join(
        model_save_dir,
        f"lora_adapter_{dataset_tag}_{model_ts}.pt",
    )
    adapter_state = getattr(train_state, "lora_adapter_state", None)
    print("\n── Model Export ──")
    if adapter_state is None:
        print(
            "  Skip LoRA adapter export: train_state.lora_adapter_state is unavailable "
            "(train.py version mismatch)."
        )
    else:
        torch.save(adapter_state, adapter_path)
        print(f"  Saved LoRA adapter: {adapter_path}")

    # Always save merged model weights so export never disappears even when
    # adapter snapshot is not available in TrainState.
    merged_path = os.path.join(
        model_save_dir,
        f"merged_model_state_{dataset_tag}_{model_ts}.pt",
    )
    merged_state = {
        k: v.detach().cpu() for k, v in get_tabpfn_model(pred_rule_trained).state_dict().items()
    }
    torch.save(
        {
            "format_version": 1,
            "dataset_name": str(dataset_name),
            "dataset_tag": dataset_tag,
            "state_dict": merged_state,
        },
        merged_path,
    )
    print(f"  Saved merged model state_dict: {merged_path}")

    # Optional: save EMD curve plot
    if config.enable_emd:
        emd_csv_path = f"emd_history_{dataset_tag}.csv"
        with open(emd_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "emd_mean", "emd_std", "coverage"])
            for step_i, emd_val, emd_std, cov in zip(
                train_state.emd_steps,
                train_state.emd_values,
                train_state.emd_stds,
                train_state.emd_coverage,
            ):
                writer.writerow([step_i, emd_val, emd_std, cov])
        print("\n── EMD History ──")
        print(f"  Saved: {emd_csv_path}")

        if len(train_state.emd_steps) > 0:
            print(f"\n── EMD Curve ──")
            print(
                f"  Last: step={train_state.emd_steps[-1]}, "
                f"emd_mean={train_state.emd_values[-1]:.6f}, "
                f"emd_std={train_state.emd_stds[-1]:.6f}, "
                f"coverage={train_state.emd_coverage[-1]}"
            )
            emd_curve_dir = "emd_curve"
            os.makedirs(emd_curve_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            emd_png_path = os.path.join(emd_curve_dir, f"emd_curve_{dataset_tag}_{ts}.png")
            xs = train_state.emd_steps
            ys = train_state.emd_values
            plt.figure(figsize=(7, 4))
            plt.plot(xs, ys, marker="o", linewidth=1.8)
            plt.xlabel("Training step")
            plt.ylabel("EMD")
            plt.title("EMD over training")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(emd_png_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Plot : {emd_png_path}")
    else:
        print("\n── EMD Curve ──")
        print("  Disabled (enable_emd=False)")

    return comp, train_state


if __name__ == "__main__":
    from train import TrainConfig

    dataset_config = DatasetConfig(
        openml_data_id=54,  # 다른 데이터셋 실험 시 여기만 변경
        test_size=0.25,
    )

    config = TrainConfig(
        task_type="classification",
        n_estimators=4,
        seed=42,
        continuation_depth=30,
        n_continuations=8,
        k_max=15,
        sc_query_ratio=0.15,
        sc_context_size=100,
        sc_context_refresh_rate=0.2,        #0.7
        sc_context_refresh_every=1,         #4
        sc_anchor_ks=(3, 7, 11, 15),
        sc_anchor_weights=(1.0, 2.0, 3.0, 4.0),
        sc_chain_beta_warmup=0.3,
        sc_chain_beta_warmup_steps=10,
        sc_chain_beta=0.3,
        sc_k1_kl_lambda_initial=30.0,
        sc_k1_kl_lambda_max=30.0,
        sc_k1_kl_ramp_start_step=1,
        sc_k1_kl_ramp_end_step=40,
        sc_episodes_per_step=1,
        sc_queries_per_episode=4,
        enable_emd=True,
        emd_fill_every=5,
        emd_fixed_rollout_paths=True, # EMD 계산용 rollout 경로를 anchor/query/depth별로 고정할지 여부
        emd_rng_fold_in=202,                #203, 204
        emd_anchor_count=3,
        emd_context_size=100,
        emd_queries_per_anchor=4,
        emd_k_values=(3, 7, 11, 15),
        num_steps=40,
        lr=1e-4,
        lr_decay_after_step=None,
        lr_decay_factor=0.5,
        grad_clip=20.0,
        lora_include_decoder=False,
        device="cuda",
    )
    run_config = RunConfig(
        enable_rollout_compare=True,    # fixed query comparison
        rollout_plot_before=False,        # before tuning fixed-query rollout plot
        rollout_base_n=100,
        rollout_depth=20,
        rollout_n_paths=8,
        rollout_n_queries=6,
        rollout_grid_rows=2,
        rollout_grid_cols=3,
        rollout_save_dir="rollout_plots",
    )

    set_openml_global_seeds(config.seed)
    X_train, X_test, y_train, y_test, categorical_x, le, dataset_name = load_openml_dataset_split(
        data_id=dataset_config.openml_data_id,
        seed=config.seed,
        test_size=dataset_config.test_size,
    )

    comp, state = run_experiment(
        X_train, X_test, y_train, y_test, categorical_x,
        config=config,
        run_config=run_config,
        dataset_name=dataset_name,
        class_names=le.classes_,
    )
