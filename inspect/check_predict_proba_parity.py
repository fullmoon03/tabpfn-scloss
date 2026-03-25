"""
Check parity between sklearn path (predict_proba) and get_belief_torch path.

Usage examples:
  python3 inspect/check_predict_proba_parity.py
  python3 inspect/check_predict_proba_parity.py --trials 20 --n-estimators 1
  python3 inspect/check_predict_proba_parity.py --generated-synthetic
  python3 inspect/check_predict_proba_parity.py --check-batched --batch-size 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import sys
import ssl

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Make project root importable when running as:
#   python3 inspect/check_predict_proba_parity.py
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_synthetic import MixtureConfig, generate_mixture_tensors  # noqa: E402
from predictive_rule import ClassifierPredRule  # noqa: E402


DEFAULT_CC18_DIDS: tuple[int, ...] = (50, 54, 151)


@dataclass
class TrialResult:
    max_abs: float
    mean_abs: float
    max_row_sum_dev_ref: float
    max_row_sum_dev_torch: float
    argmax_match: float
    batched_max_abs: Optional[float] = None
    batched_mean_abs: Optional[float] = None


@dataclass
class ScenarioResult:
    name: str
    trials: int
    max_abs_worst: float
    max_abs_mean: float
    mean_abs_mean: float
    argmax_match_mean: float
    row_sum_dev_ref_max: float
    row_sum_dev_torch_max: float
    passed: bool
    batched_max_abs_worst: Optional[float] = None
    batched_max_abs_mean: Optional[float] = None
    batched_mean_abs_mean: Optional[float] = None


@dataclass
class ParityScenario:
    name: str
    n_estimators: int
    average_before_softmax: bool
    input_mode: Literal["numpy32", "numpy64", "pandas"]
    n_context: int
    n_query: int
    n_features: int
    n_classes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n-context", type=int, default=120)
    parser.add_argument("--n-query", type=int, default=32)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--average-before-softmax",
        action="store_true",
        help="Pass average_before_softmax=True to ClassifierPredRule.",
    )
    parser.add_argument(
        "--generated-synthetic",
        action="store_true",
        help="Use synthetic tasks generated in-memory from inspect/generate_synthetic.py.",
    )
    parser.add_argument(
        "--generated-tasks",
        type=int,
        default=10,
        help="Number of generated synthetic tasks when --generated-synthetic is used.",
    )
    parser.add_argument(
        "--generated-seed",
        type=int,
        default=None,
        help="Seed for generated synthetic tasks. Defaults to --seed.",
    )
    parser.add_argument(
        "--openml-data-id",
        type=int,
        default=None,
        help="Optional OpenML dataset id for real-dataset parity checks.",
    )
    parser.add_argument(
        "--openml-cc18",
        action="store_true",
        help="Use OpenML-CC18 mode with multiple dataset IDs.",
    )
    parser.add_argument(
        "--cc18-dids",
        type=str,
        default="50,54,151",
        help="Comma-separated OpenML dataset IDs used in --openml-cc18 mode.",
    )
    parser.add_argument(
        "--openml-test-size",
        type=float,
        default=0.2,
        help="Test split ratio for OpenML mode.",
    )
    parser.add_argument(
        "--sklearn-dataset",
        type=str,
        default=None,
        choices=["iris", "wine", "breast_cancer", "digits"],
        help=(
            "Use sklearn built-in real dataset for parity checks "
            "(network-free fallback to OpenML)."
        ),
    )
    parser.add_argument(
        "--check-batched",
        action="store_true",
        help="Also compare get_belief_torch_batched vs looped get_belief_torch.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-fastpath",
        action="store_true",
        help=(
            "Allow no_grad fast-path in get_belief_torch "
            "(delegating to predict_proba when context==fit input). "
            "Default is False for meaningful torch-path parity checks."
        ),
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run multiple parity scenarios (estimators/averaging/input-format) "
            "instead of a single configuration."
        ),
    )
    parser.add_argument(
        "--sweep-scenarios",
        type=str,
        default="all",
        help=(
            "Comma-separated scenario names to run in --sweep mode, or 'all'. "
            "Names: est1_np32, est4_np32, est4_pre_softmax_np32, "
            "est1_np64, est4_pre_softmax_np64, est1_pandas, est4_pre_softmax_pandas"
        ),
    )
    parser.add_argument("--tol-max-abs", type=float, default=2e-5)
    parser.add_argument("--tol-mean-abs", type=float, default=2e-6)
    parser.add_argument("--tol-argmax", type=float, default=0.9999)
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable SSL verification for OpenML downloads.",
    )
    return parser.parse_args()


def _relabel_to_contiguous(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y).astype(int)
    _, y_local = np.unique(y_arr, return_inverse=True)
    return y_local.astype(np.int64)


def _make_random_case(
    rng: np.random.Generator,
    n_context: int,
    n_query: int,
    n_features: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_context < 2:
        raise ValueError(f"n_context must be >=2, got {n_context}")
    if n_query < 1:
        raise ValueError(f"n_query must be >=1, got {n_query}")
    if n_features < 1:
        raise ValueError(f"n_features must be >=1, got {n_features}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >=2, got {n_classes}")

    x_ctx = rng.normal(size=(n_context, n_features)).astype(np.float32)
    y_ctx = rng.integers(0, n_classes, size=(n_context,), endpoint=False).astype(np.int64)
    # Ensure at least two classes exist in context.
    if len(np.unique(y_ctx)) < 2:
        y_ctx[:2] = np.array([0, 1], dtype=np.int64)
    y_ctx = _relabel_to_contiguous(y_ctx)
    x_q = rng.normal(size=(n_query, n_features)).astype(np.float32)
    return x_ctx, y_ctx, x_q


def _load_generated_synthetic_tasks(
    *,
    n_tasks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = MixtureConfig()
    x_tasks, y_tasks, _ = generate_mixture_tensors(
        n_tasks=int(n_tasks),
        cfg=cfg,
        seed=int(seed),
        return_metadata=True,
    )
    if x_tasks.ndim != 3 or y_tasks.ndim != 2:
        raise ValueError(
            f"Expected X:(N_task,N_point,d), y:(N_task,N_point), got X:{x_tasks.shape}, y:{y_tasks.shape}"
        )
    return np.asarray(x_tasks, dtype=np.float32), np.rint(y_tasks).astype(np.int64)


def _encode_openml_features(x_frame: pd.DataFrame) -> np.ndarray:
    cols: list[np.ndarray] = []
    for col in x_frame.columns:
        s = x_frame[col]
        is_cat_dtype = isinstance(s.dtype, pd.CategoricalDtype)
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or is_cat_dtype:
            s2 = s.astype("string").fillna("__nan__")
            cat = pd.Categorical(s2)
            cols.append(cat.codes.astype(np.float32))
        else:
            n = pd.to_numeric(s, errors="coerce")
            if n.isna().all():
                n = pd.Series(np.zeros(len(n), dtype=np.float32))
            else:
                n = n.fillna(float(n.median()))
            cols.append(n.to_numpy(dtype=np.float32))
    return np.column_stack(cols).astype(np.float32)


def _parse_int_csv(text: str) -> list[int]:
    vals: list[int] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    return vals


def _load_openml_dataset(
    *,
    data_id: int,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except Exception as e:
        raise RuntimeError(
            "Failed to fetch OpenML dataset. "
            "Try --sklearn-dataset {iris,wine,breast_cancer,digits} as fallback."
        ) from e
    X_raw = data.data
    y_raw = data.target
    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(X_raw)
    y_series = pd.Series(y_raw)
    valid_mask = y_series.notna().to_numpy()
    if not np.all(valid_mask):
        X_raw = X_raw.loc[valid_mask].reset_index(drop=True)
        y_series = y_series.loc[valid_mask].reset_index(drop=True)
    X = _encode_openml_features(X_raw)
    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str).to_numpy()).astype(np.int64)
    name = str(getattr(data, "name", f"openml_{int(data_id)}"))

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
    return X_train, y_train, X_test, y_test, name


def _load_sklearn_dataset(
    *,
    name: str,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader_map = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    if name not in loader_map:
        raise ValueError(f"Unknown sklearn dataset: {name}")
    data = loader_map[name](return_X_y=True)
    X, y_raw = data
    y = np.asarray(y_raw).astype(np.int64)

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
    return X_train, y_train, X_test, y_test


def _make_openml_case(
    rng: np.random.Generator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_context: int,
    n_query: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X_train) < n_context:
        raise ValueError(
            f"OpenML train size {len(X_train)} < n_context {n_context}"
        )
    if len(X_test) < 1:
        raise ValueError("OpenML test split is empty.")

    # Keep trying until we get at least two classes in context.
    for _ in range(100):
        idx_ctx = rng.choice(len(X_train), size=n_context, replace=False)
        x_ctx = np.asarray(X_train[idx_ctx], dtype=np.float32)
        y_ctx_raw = np.asarray(y_train[idx_ctx], dtype=np.int64)
        if len(np.unique(y_ctx_raw)) < 2:
            continue
        y_ctx = _relabel_to_contiguous(y_ctx_raw)

        if len(X_test) >= n_query:
            idx_q = rng.choice(len(X_test), size=n_query, replace=False)
        else:
            idx_q = rng.choice(len(X_test), size=n_query, replace=True)
        x_q = np.asarray(X_test[idx_q], dtype=np.float32)
        return x_ctx, y_ctx, x_q

    raise RuntimeError("Failed to sample OpenML context with >=2 classes.")


def _make_generated_case(
    rng: np.random.Generator,
    x_tasks: np.ndarray,
    y_tasks: np.ndarray,
    n_context: int,
    n_query: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tasks, n_points, _ = x_tasks.shape
    # Keep trying until we get at least two classes in the sampled context.
    for _ in range(50):
        t_idx = int(rng.integers(0, n_tasks))
        x_t = x_tasks[t_idx]
        y_t = y_tasks[t_idx]
        if n_points < (n_context + n_query):
            raise ValueError(
                f"Task has only {n_points} points but needs n_context+n_query={n_context+n_query}"
            )
        perm = rng.permutation(n_points)
        idx_ctx = perm[:n_context]
        idx_q = perm[n_context : n_context + n_query]
        x_ctx = np.asarray(x_t[idx_ctx], dtype=np.float32)
        y_ctx_raw = np.asarray(y_t[idx_ctx]).astype(np.int64)
        if len(np.unique(y_ctx_raw)) < 2:
            continue
        y_ctx = _relabel_to_contiguous(y_ctx_raw)
        x_q = np.asarray(x_t[idx_q], dtype=np.float32)
        return x_ctx, y_ctx, x_q
    raise RuntimeError("Failed to sample context with >=2 classes from generated tasks.")


def _run_one_trial(
    pred_rule: ClassifierPredRule,
    x_ctx: np.ndarray,
    y_ctx: np.ndarray,
    x_q: np.ndarray,
    *,
    input_mode: Literal["numpy32", "numpy64", "pandas"],
    check_batched: bool,
    batch_size: int,
    rng: np.random.Generator,
) -> TrialResult:
    x_ctx_in = _convert_input_x(x_ctx, input_mode)
    x_q_in = _convert_input_x(x_q, input_mode)

    pred_rule.fit(x_ctx_in, y_ctx)

    probs_ref = np.asarray(pred_rule._clf.predict_proba(x_q_in), dtype=np.float64)  # noqa: SLF001
    with torch.no_grad():
        probs_torch = (
            pred_rule.get_belief_torch(x_q_in, x_ctx_in, y_ctx).detach().cpu().numpy()
        )
    probs_torch = np.asarray(probs_torch, dtype=np.float64)

    abs_diff = np.abs(probs_ref - probs_torch)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    max_row_sum_dev_ref = float(np.max(np.abs(probs_ref.sum(axis=1) - 1.0)))
    max_row_sum_dev_torch = float(np.max(np.abs(probs_torch.sum(axis=1) - 1.0)))
    argmax_match = float((probs_ref.argmax(axis=1) == probs_torch.argmax(axis=1)).mean())

    batched_max_abs: Optional[float] = None
    batched_mean_abs: Optional[float] = None
    if check_batched:
        x_contexts: list[np.ndarray] = []
        y_contexts: list[np.ndarray] = []
        x_contexts_in = []
        for _ in range(batch_size):
            noise = rng.normal(scale=0.1, size=x_ctx.shape).astype(np.float32)
            x_b = np.asarray(x_ctx + noise, dtype=np.float32)
            # Re-sample labels while keeping class set contiguous.
            y_b = np.asarray(y_ctx.copy(), dtype=np.int64)
            rng.shuffle(y_b)
            y_b = _relabel_to_contiguous(y_b)
            x_contexts.append(x_b)
            x_contexts_in.append(_convert_input_x(x_b, input_mode))
            y_contexts.append(y_b)

        with torch.no_grad():
            probs_batched = pred_rule.get_belief_torch_batched(
                x_q_in, x_contexts_in, y_contexts
            ).detach().cpu().numpy()
        probs_loop = []
        with torch.no_grad():
            for x_b_in, y_b in zip(x_contexts_in, y_contexts):
                probs_loop.append(
                    pred_rule.get_belief_torch(x_q_in, x_b_in, y_b).detach().cpu().numpy()
                )
        probs_loop_arr = np.asarray(probs_loop, dtype=np.float64)
        probs_batched = np.asarray(probs_batched, dtype=np.float64)
        bdiff = np.abs(probs_batched - probs_loop_arr)
        batched_max_abs = float(np.max(bdiff))
        batched_mean_abs = float(np.mean(bdiff))

    return TrialResult(
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_row_sum_dev_ref=max_row_sum_dev_ref,
        max_row_sum_dev_torch=max_row_sum_dev_torch,
        argmax_match=argmax_match,
        batched_max_abs=batched_max_abs,
        batched_mean_abs=batched_mean_abs,
    )


def _convert_input_x(
    X: np.ndarray,
    mode: Literal["numpy32", "numpy64", "pandas"],
):
    if mode == "numpy32":
        return np.asarray(X, dtype=np.float32)
    if mode == "numpy64":
        return np.asarray(X, dtype=np.float64)
    if mode == "pandas":
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pandas is required for input_mode='pandas'") from e
        return pd.DataFrame(np.asarray(X, dtype=np.float32))
    raise ValueError(f"Unknown input mode: {mode}")


def _summarize(
    results: list[TrialResult],
    *,
    name: str,
    tol_max_abs: float,
    tol_mean_abs: float,
    tol_argmax: float,
) -> ScenarioResult:
    max_abs_vals = np.asarray([r.max_abs for r in results], dtype=np.float64)
    mean_abs_vals = np.asarray([r.mean_abs for r in results], dtype=np.float64)
    argmax_vals = np.asarray([r.argmax_match for r in results], dtype=np.float64)
    ref_sum_dev_vals = np.asarray(
        [r.max_row_sum_dev_ref for r in results], dtype=np.float64
    )
    torch_sum_dev_vals = np.asarray(
        [r.max_row_sum_dev_torch for r in results], dtype=np.float64
    )

    has_batched = results[0].batched_max_abs is not None
    batched_max_abs_worst = None
    batched_max_abs_mean = None
    batched_mean_abs_mean = None
    if has_batched:
        batched_max = np.asarray([r.batched_max_abs for r in results], dtype=np.float64)
        batched_mean = np.asarray(
            [r.batched_mean_abs for r in results], dtype=np.float64
        )
        batched_max_abs_worst = float(batched_max.max())
        batched_max_abs_mean = float(batched_max.mean())
        batched_mean_abs_mean = float(batched_mean.mean())

    max_abs_worst = float(max_abs_vals.max())
    mean_abs_mean = float(mean_abs_vals.mean())
    argmax_match_mean = float(argmax_vals.mean())
    passed = (
        max_abs_worst <= tol_max_abs
        and mean_abs_mean <= tol_mean_abs
        and argmax_match_mean >= tol_argmax
    )
    return ScenarioResult(
        name=name,
        trials=len(results),
        max_abs_worst=max_abs_worst,
        max_abs_mean=float(max_abs_vals.mean()),
        mean_abs_mean=mean_abs_mean,
        argmax_match_mean=argmax_match_mean,
        row_sum_dev_ref_max=float(ref_sum_dev_vals.max()),
        row_sum_dev_torch_max=float(torch_sum_dev_vals.max()),
        passed=passed,
        batched_max_abs_worst=batched_max_abs_worst,
        batched_max_abs_mean=batched_max_abs_mean,
        batched_mean_abs_mean=batched_mean_abs_mean,
    )


def _print_summary(summary: ScenarioResult, *, include_batched: bool) -> None:
    print(f"\n=== Parity Summary ({summary.name}) ===")
    print(f"trials               : {summary.trials}")
    print(f"max_abs (worst)      : {summary.max_abs_worst:.6e}")
    print(f"max_abs (mean)       : {summary.max_abs_mean:.6e}")
    print(f"mean_abs (mean)      : {summary.mean_abs_mean:.6e}")
    print(f"argmax_match (mean)  : {summary.argmax_match_mean:.6f}")
    print(f"row_sum_dev_ref max  : {summary.row_sum_dev_ref_max:.6e}")
    print(f"row_sum_dev_torch max: {summary.row_sum_dev_torch_max:.6e}")
    print(f"pass                 : {summary.passed}")
    if include_batched and summary.batched_max_abs_worst is not None:
        print("=== Batched Consistency ===")
        print(f"batched_max_abs (worst): {summary.batched_max_abs_worst:.6e}")
        print(f"batched_max_abs (mean) : {summary.batched_max_abs_mean:.6e}")
        print(f"batched_mean_abs (mean): {summary.batched_mean_abs_mean:.6e}")


def _default_scenarios(
    *,
    n_features: int,
    n_classes: int,
    n_context: int,
    n_query: int,
) -> list[ParityScenario]:
    return [
        ParityScenario(
            name="est1_np32",
            n_estimators=1,
            average_before_softmax=False,
            input_mode="numpy32",
            n_context=n_context,
            n_query=n_query,
            n_features=n_features,
            n_classes=n_classes,
        ),
        ParityScenario(
            name="est4_np32",
            n_estimators=4,
            average_before_softmax=False,
            input_mode="numpy32",
            n_context=n_context,
            n_query=n_query,
            n_features=n_features,
            n_classes=n_classes,
        ),
        ParityScenario(
            name="est4_pre_softmax_np32",
            n_estimators=4,
            average_before_softmax=True,
            input_mode="numpy32",
            n_context=n_context,
            n_query=n_query,
            n_features=n_features,
            n_classes=n_classes,
        ),
        ParityScenario(
            name="est1_np64",
            n_estimators=1,
            average_before_softmax=False,
            input_mode="numpy64",
            n_context=n_context,
            n_query=n_query,
            n_features=n_features,
            n_classes=n_classes,
        ),
        ParityScenario(
            name="est4_pre_softmax_np64",
            n_estimators=4,
            average_before_softmax=True,
            input_mode="numpy64",
            n_context=n_context,
            n_query=n_query,
            n_features=n_features,
            n_classes=n_classes,
        ),
        ParityScenario(
            name="est1_pandas",
            n_estimators=1,
            average_before_softmax=False,
            input_mode="pandas",
            n_context=n_context,
            n_query=n_query,
            n_features=n_features,
            n_classes=n_classes,
        ),
        ParityScenario(
            name="est4_pre_softmax_pandas",
            n_estimators=4,
            average_before_softmax=True,
            input_mode="pandas",
            n_context=n_context,
            n_query=n_query,
            n_features=n_features,
            n_classes=n_classes,
        ),
    ]


def main() -> None:
    args = parse_args()
    if bool(args.insecure_ssl):
        ssl._create_default_https_context = ssl._create_unverified_context
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    x_tasks = None
    y_tasks = None
    real_pools: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    n_features = int(args.n_features)
    selected_sources = sum(
        int(v is not None)
        for v in (args.openml_data_id, args.sklearn_dataset)
    )
    selected_sources += int(bool(args.generated_synthetic))
    if bool(args.openml_cc18):
        selected_sources += 1
    if selected_sources > 1:
        raise ValueError(
            "Use only one of --generated-synthetic, --openml-data-id, --openml-cc18, --sklearn-dataset."
        )
    if bool(args.generated_synthetic):
        generated_seed = int(args.seed if args.generated_seed is None else args.generated_seed)
        x_tasks, y_tasks = _load_generated_synthetic_tasks(
            n_tasks=int(args.generated_tasks),
            seed=generated_seed,
        )
        n_features = int(x_tasks.shape[-1])
        print(
            f"Loaded generated synthetic tasks: shape={x_tasks.shape}, "
            f"seed={generated_seed}"
        )
    elif args.openml_data_id is not None:
        x_real_train, y_real_train, x_real_test, _, ds_name = _load_openml_dataset(
            data_id=int(args.openml_data_id),
            seed=int(args.seed),
            test_size=float(args.openml_test_size),
        )
        real_pools.append((f"did={int(args.openml_data_id)}:{ds_name}", x_real_train, y_real_train, x_real_test))
        n_features = int(x_real_train.shape[-1])
        print(
            f"Loaded OpenML dataset: did={int(args.openml_data_id)} name={ds_name}, "
            f"train={x_real_train.shape}, test={x_real_test.shape}"
        )
    elif bool(args.openml_cc18):
        dids = _parse_int_csv(args.cc18_dids)
        if len(dids) == 0:
            dids = list(DEFAULT_CC18_DIDS)
        print(f"OpenML-CC18 mode: dids={dids}")
        for i, did in enumerate(dids):
            x_real_train, y_real_train, x_real_test, _, ds_name = _load_openml_dataset(
                data_id=int(did),
                seed=int(args.seed) + i * 1009,
                test_size=float(args.openml_test_size),
            )
            real_pools.append((f"did={int(did)}:{ds_name}", x_real_train, y_real_train, x_real_test))
            print(
                f"  loaded did={int(did)} name={ds_name}, "
                f"train={x_real_train.shape}, test={x_real_test.shape}"
            )
        if len(real_pools) == 0:
            raise RuntimeError("No OpenML-CC18 datasets loaded.")
        n_features = int(real_pools[0][1].shape[-1])
    elif args.sklearn_dataset is not None:
        x_real_train, y_real_train, x_real_test, _ = _load_sklearn_dataset(
            name=str(args.sklearn_dataset),
            seed=int(args.seed),
            test_size=float(args.openml_test_size),
        )
        real_pools.append((str(args.sklearn_dataset), x_real_train, y_real_train, x_real_test))
        n_features = int(x_real_train.shape[-1])
        print(
            f"Loaded sklearn dataset: {args.sklearn_dataset}, "
            f"train={x_real_train.shape}, test={x_real_test.shape}"
        )

    if args.sweep:
        scenario_map = {
            s.name: s
            for s in _default_scenarios(
                n_features=n_features,
                n_classes=int(args.n_classes),
                n_context=int(args.n_context),
                n_query=int(args.n_query),
            )
        }
        requested = str(args.sweep_scenarios).strip()
        if requested.lower() == "all":
            scenarios = list(scenario_map.values())
        else:
            names = [x.strip() for x in requested.split(",") if x.strip()]
            unknown = [n for n in names if n not in scenario_map]
            if unknown:
                raise ValueError(f"Unknown sweep scenarios: {unknown}")
            scenarios = [scenario_map[n] for n in names]

        all_summaries: list[ScenarioResult] = []
        print(f"Running sweep with {len(scenarios)} scenarios")
        for s_idx, scenario in enumerate(scenarios, start=1):
            scenario_seed = int(args.seed) + s_idx * 1000
            scenario_rng = np.random.default_rng(scenario_seed)
            np.random.seed(scenario_seed)
            torch.manual_seed(scenario_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(scenario_seed)

            print(
                f"\n--- Scenario {s_idx}/{len(scenarios)}: {scenario.name} "
                f"(E={scenario.n_estimators}, avg_before_softmax={scenario.average_before_softmax}, "
                f"input={scenario.input_mode}) ---"
            )
            results: list[TrialResult] = []
            for i in range(args.trials):
                trial_source = "random"
                pred_n_features = int(scenario.n_features)
                if x_tasks is not None and y_tasks is not None:
                    x_ctx, y_ctx, x_q = _make_generated_case(
                        rng=scenario_rng,
                        x_tasks=x_tasks,
                        y_tasks=y_tasks,
                        n_context=int(scenario.n_context),
                        n_query=int(scenario.n_query),
                    )
                    pred_n_features = int(x_ctx.shape[1])
                    trial_source = "generated"
                elif len(real_pools) > 0:
                    pool_name, x_real_train, y_real_train, x_real_test = real_pools[i % len(real_pools)]
                    x_ctx, y_ctx, x_q = _make_openml_case(
                        rng=scenario_rng,
                        X_train=x_real_train,
                        y_train=y_real_train,
                        X_test=x_real_test,
                        n_context=int(scenario.n_context),
                        n_query=int(scenario.n_query),
                    )
                    pred_n_features = int(x_ctx.shape[1])
                    trial_source = pool_name
                else:
                    x_ctx, y_ctx, x_q = _make_random_case(
                        rng=scenario_rng,
                        n_context=int(scenario.n_context),
                        n_query=int(scenario.n_query),
                        n_features=int(scenario.n_features),
                        n_classes=int(scenario.n_classes),
                    )
                pred_rule = ClassifierPredRule(
                    categorical_x=[False] * pred_n_features,
                    n_estimators=int(scenario.n_estimators),
                    average_before_softmax=bool(scenario.average_before_softmax),
                )
                pred_rule._allow_predict_proba_fastpath = bool(args.allow_fastpath)  # noqa: SLF001
                trial = _run_one_trial(
                    pred_rule,
                    x_ctx,
                    y_ctx,
                    x_q,
                    input_mode=scenario.input_mode,
                    check_batched=bool(args.check_batched),
                    batch_size=int(args.batch_size),
                    rng=scenario_rng,
                )
                results.append(trial)
                msg = (
                    f"[{i+1:02d}/{args.trials}] max_abs={trial.max_abs:.3e}, "
                    f"mean_abs={trial.mean_abs:.3e}, argmax_match={trial.argmax_match:.3f}, "
                    f"src={trial_source}"
                )
                if args.check_batched:
                    msg += (
                        f", batched_max_abs={trial.batched_max_abs:.3e}, "
                        f"batched_mean_abs={trial.batched_mean_abs:.3e}"
                    )
                print(msg)

            summary = _summarize(
                results,
                name=scenario.name,
                tol_max_abs=float(args.tol_max_abs),
                tol_mean_abs=float(args.tol_mean_abs),
                tol_argmax=float(args.tol_argmax),
            )
            _print_summary(summary, include_batched=bool(args.check_batched))
            all_summaries.append(summary)

        print("\n=== Sweep Overview ===")
        for s in all_summaries:
            status = "PASS" if s.passed else "FAIL"
            print(
                f"{status:4s}  {s.name:28s}  "
                f"max_abs_worst={s.max_abs_worst:.3e}  "
                f"mean_abs_mean={s.mean_abs_mean:.3e}  "
                f"argmax_mean={s.argmax_match_mean:.4f}"
            )
        failed = [s.name for s in all_summaries if not s.passed]
        if failed:
            print(f"Failed scenarios: {failed}")
            raise SystemExit(1)
        return

    results: list[TrialResult] = []
    for i in range(args.trials):
        trial_source = "random"
        pred_n_features = int(n_features)
        if x_tasks is not None and y_tasks is not None:
            x_ctx, y_ctx, x_q = _make_generated_case(
                rng=rng,
                x_tasks=x_tasks,
                y_tasks=y_tasks,
                n_context=int(args.n_context),
                n_query=int(args.n_query),
            )
            pred_n_features = int(x_ctx.shape[1])
            trial_source = "generated"
        elif len(real_pools) > 0:
            pool_name, x_real_train, y_real_train, x_real_test = real_pools[i % len(real_pools)]
            x_ctx, y_ctx, x_q = _make_openml_case(
                rng=rng,
                X_train=x_real_train,
                y_train=y_real_train,
                X_test=x_real_test,
                n_context=int(args.n_context),
                n_query=int(args.n_query),
            )
            pred_n_features = int(x_ctx.shape[1])
            trial_source = pool_name
        else:
            x_ctx, y_ctx, x_q = _make_random_case(
                rng=rng,
                n_context=int(args.n_context),
                n_query=int(args.n_query),
                n_features=n_features,
                n_classes=int(args.n_classes),
            )
        pred_rule = ClassifierPredRule(
            categorical_x=[False] * pred_n_features,
            n_estimators=int(args.n_estimators),
            average_before_softmax=bool(args.average_before_softmax),
        )
        pred_rule._allow_predict_proba_fastpath = bool(args.allow_fastpath)  # noqa: SLF001
        trial = _run_one_trial(
            pred_rule,
            x_ctx,
            y_ctx,
            x_q,
            input_mode="numpy32",
            check_batched=bool(args.check_batched),
            batch_size=int(args.batch_size),
            rng=rng,
        )
        results.append(trial)
        msg = (
            f"[{i+1:02d}/{args.trials}] max_abs={trial.max_abs:.3e}, "
            f"mean_abs={trial.mean_abs:.3e}, argmax_match={trial.argmax_match:.3f}, "
            f"src={trial_source}"
        )
        if args.check_batched:
            msg += (
                f", batched_max_abs={trial.batched_max_abs:.3e}, "
                f"batched_mean_abs={trial.batched_mean_abs:.3e}"
            )
        print(msg)

    summary = _summarize(
        results,
        name="single_run",
        tol_max_abs=float(args.tol_max_abs),
        tol_mean_abs=float(args.tol_mean_abs),
        tol_argmax=float(args.tol_argmax),
    )
    _print_summary(summary, include_batched=bool(args.check_batched))


if __name__ == "__main__":
    main()
