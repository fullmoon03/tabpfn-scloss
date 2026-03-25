"""
evaluate_openml_cc18_metrics.py
------------------------------------------------------------
Evaluate baseline TabPFN on OpenML-CC18 datasets.

For each dataset:
  - fetch OpenML dataset
  - encode labels/features for TabPFN
  - sample context/query split (default: context=100, query=50)
  - compute Accuracy / NLL / ECE

Outputs:
  - per-dataset metrics CSV
  - failed datasets CSV
  - summary figure (curves + distributions + scatter)
"""

import argparse
import csv
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

try:
    import openml as pyopenml
except Exception:
    pyopenml = None

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval import compute_basic_metrics
from predictive_rule import ClassifierPredRule


@dataclass
class DatasetEvalResult:
    data_id: int
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    context_classes: int
    unseen_query_labels: int
    accuracy: float
    nll: float
    ece: float


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """Load plain tensor state_dict from common checkpoint formats."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        raw = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        raw = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    state: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            state[str(k)] = v.detach().cpu().clone()
    if len(state) == 0:
        raise ValueError("No tensor state_dict entries found in checkpoint.")
    return state


def _open_json_url(url: str) -> dict:
    """Open JSON URL with verified SSL first, then fallback to unverified context."""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=30, context=ctx) as r:
            return json.loads(r.read().decode("utf-8"))


def _fetch_cc18_dataset_ids(tag: str = "OpenML-CC18") -> list[int]:
    """
    Fetch OpenML dataset IDs for the given tag.

    Priority:
      1) python-openml suite API
      2) REST tag endpoint fallback
    """
    # 1) python-openml path
    if pyopenml is not None:
        suite_candidates: list[Any] = [tag]
        if str(tag).lower() in {"openml-cc18", "cc18"}:
            suite_candidates.append(99)
        for suite_key in suite_candidates:
            try:
                suite = pyopenml.study.get_suite(suite_key)
                ids = [int(v) for v in list(getattr(suite, "data", []))]
                if len(ids) > 0:
                    return sorted(set(ids))
            except Exception:
                continue

    # 2) REST fallback
    tag_enc = urllib.parse.quote(tag, safe="")
    url = f"https://www.openml.org/api/v1/json/data/list/tag/{tag_enc}/limit/10000"
    obj = _open_json_url(url)
    ds_list = obj.get("data", {}).get("dataset", [])
    data_ids: list[int] = []
    for item in ds_list:
        did = item.get("did")
        try:
            data_ids.append(int(did))
        except Exception:
            continue
    return sorted(set(data_ids))


def _encode_openml_features(X_frame: pd.DataFrame) -> tuple[np.ndarray, list[bool]]:
    """
    Encode mixed-type OpenML features into numeric matrix for TabPFN.

    - categorical/object/bool: category codes
    - numeric: median-imputed float
    """
    cols = []
    categorical_x: list[bool] = []
    for col in X_frame.columns:
        s = X_frame[col]
        is_cat_dtype = isinstance(s.dtype, pd.CategoricalDtype)
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or is_cat_dtype:
            s2 = s.astype("string").fillna("__nan__")
            cat = pd.Categorical(s2)
            vals = cat.codes.astype(np.float32)  # -1 should not happen after fillna, but still numeric
            cols.append(vals)
            categorical_x.append(True)
        else:
            n = pd.to_numeric(s, errors="coerce")
            if n.isna().all():
                n = pd.Series(np.zeros(len(n), dtype=np.float32))
            else:
                n = n.fillna(float(n.median()))
            cols.append(n.to_numpy(dtype=np.float32))
            categorical_x.append(False)

    X = np.column_stack(cols).astype(np.float32)
    return X, categorical_x


def _load_openml_dataset(data_id: int) -> tuple[np.ndarray, np.ndarray, list[bool], str]:
    data = fetch_openml(data_id=int(data_id), as_frame=True, parser="auto")
    X_raw = data.data
    y_raw = data.target
    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(X_raw)

    y_series = pd.Series(y_raw)
    valid_mask = y_series.notna().to_numpy()
    if not np.all(valid_mask):
        X_raw = X_raw.loc[valid_mask].reset_index(drop=True)
        y_series = y_series.loc[valid_mask].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str).to_numpy()).astype(np.int64)
    X, categorical_x = _encode_openml_features(X_raw)
    name = str(getattr(data, "name", f"openml_{int(data_id)}"))
    return X, y, categorical_x, name


def _sample_context_query_indices(
    *,
    y: np.ndarray,
    context_size: int,
    query_size: int,
    seed: int,
    max_tries: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(y))
    if n < 3:
        raise ValueError(f"Dataset too small: n={n}")
    n_ctx = int(np.clip(context_size, 1, n - 2))
    n_q = int(np.clip(query_size, 1, n - n_ctx - 1))

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        perm = np.asarray(rng.permutation(n), dtype=int)
        idx_ctx = perm[:n_ctx]
        idx_q = perm[n_ctx: n_ctx + n_q]
        if len(np.unique(y[idx_ctx])) >= 2:
            return idx_ctx, idx_q
    raise RuntimeError("Failed to sample context with >=2 classes.")


def _evaluate_dataset(
    *,
    data_id: int,
    context_size: int,
    query_size: int,
    seed: int,
    n_estimators: int,
    locked_state_dict: dict[str, torch.Tensor] | None,
) -> DatasetEvalResult:
    X, y, categorical_x, name = _load_openml_dataset(data_id)
    idx_ctx, idx_q = _sample_context_query_indices(
        y=y,
        context_size=context_size,
        query_size=query_size,
        seed=seed,
    )

    x_ctx = np.asarray(X[idx_ctx], dtype=np.float32)
    y_ctx_global = np.asarray(y[idx_ctx], dtype=np.int64)
    x_q = np.asarray(X[idx_q], dtype=np.float32)
    y_q_global = np.asarray(y[idx_q], dtype=np.int64)

    local_classes, y_ctx_local = np.unique(y_ctx_global, return_inverse=True)
    y_ctx_local = y_ctx_local.astype(np.int64)
    c_global = int(np.max(y)) + 1

    rule = ClassifierPredRule(categorical_x, n_estimators=n_estimators)
    if locked_state_dict is not None:
        rule._locked_state_dict = {k: v.clone().cpu() for k, v in locked_state_dict.items()}

    rule.fit(x_ctx, y_ctx_local)
    with torch.no_grad():
        probs_local = rule.get_belief_torch(x_q, x_ctx, y_ctx_local).cpu().numpy()

    probs_global = np.zeros((len(x_q), c_global), dtype=np.float64)
    probs_global[:, np.asarray(local_classes).astype(int)] = np.asarray(probs_local, dtype=np.float64)
    probs_global = probs_global / np.clip(probs_global.sum(axis=1, keepdims=True), 1e-12, None)
    m = compute_basic_metrics(probs_global, y_q_global)
    unseen = int(np.sum(~np.isin(y_q_global, local_classes)))

    return DatasetEvalResult(
        data_id=int(data_id),
        name=name,
        n_samples=int(len(y)),
        n_features=int(X.shape[1]),
        n_classes=int(c_global),
        context_classes=int(len(local_classes)),
        unseen_query_labels=unseen,
        accuracy=float(m.accuracy),
        nll=float(m.nll),
        ece=float(m.ece),
    )


def _plot_summary(
    *,
    rows: list[DatasetEvalResult],
    context_size: int,
    query_size: int,
    save_path: str,
) -> None:
    idx = np.arange(len(rows))
    acc = np.asarray([r.accuracy for r in rows], dtype=np.float64)
    nll = np.asarray([r.nll for r in rows], dtype=np.float64)
    n_classes = np.asarray([r.n_classes for r in rows], dtype=np.int32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(idx, acc, marker="o", linewidth=1.3, color="tab:blue")
    ax.axhline(float(np.mean(acc)), linestyle="--", color="tab:blue", alpha=0.5, label="mean")
    ax.set_title("Accuracy by Dataset")
    ax.set_xlabel("Dataset rank")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(idx, nll, marker="o", linewidth=1.3, color="tab:red")
    ax.axhline(float(np.mean(nll)), linestyle="--", color="tab:red", alpha=0.5, label="mean")
    ax.set_title("NLL by Dataset")
    ax.set_xlabel("Dataset rank")
    ax.set_ylabel("NLL")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    bins = min(14, max(5, len(rows) // 3 if len(rows) > 0 else 5))
    ax.hist(acc, bins=bins, alpha=0.8, color="tab:blue", label="Accuracy")
    ax.hist(nll, bins=bins, alpha=0.5, color="tab:red", label="NLL")
    ax.set_title("Metric Distributions")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    sc = ax.scatter(acc, nll, c=n_classes, cmap="viridis", s=52, alpha=0.9, edgecolors="none")
    ax.set_title("Accuracy vs NLL (color=#classes)")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("NLL")
    ax.grid(True, alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Number of classes")

    fig.suptitle(
        f"OpenML-CC18 TabPFN diagnostics (context={context_size}, query={query_size}, n={len(rows)})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="OpenML-CC18", help="OpenML dataset tag")
    p.add_argument("--context-size", type=int, default=100, help="Context size")
    p.add_argument("--query-size", type=int, default=50, help="Query size")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--n-estimators", type=int, default=2, help="TabPFN n_estimators")
    p.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help="0 means all datasets under tag; >0 evaluates first K dataset ids",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional merged model .pt path. If provided, evaluate with locked weights.",
    )
    p.add_argument("--save-dir", type=str, default="openml_cc18_eval", help="Output dir")
    p.add_argument("--csv-name", type=str, default="cc18_metrics.csv", help="Metrics CSV filename")
    p.add_argument("--fail-csv-name", type=str, default="cc18_failures.csv", help="Failure CSV filename")
    p.add_argument("--plot-name", type=str, default="cc18_summary.png", help="Summary plot filename")
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable SSL certificate verification for OpenML downloads (use only in controlled env).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.insecure_ssl):
        ssl._create_default_https_context = ssl._create_unverified_context
    os.makedirs(args.save_dir, exist_ok=True)

    data_ids = _fetch_cc18_dataset_ids(args.tag)
    if int(args.max_datasets) > 0:
        data_ids = data_ids[: int(args.max_datasets)]
    if len(data_ids) == 0:
        raise RuntimeError(f"No datasets found for tag={args.tag}")

    print("── OpenML-CC18 Diagnostics ──")
    print(
        f"tag={args.tag}, datasets={len(data_ids)}, context={args.context_size}, "
        f"query={args.query_size}, n_estimators={args.n_estimators}"
    )
    if pyopenml is not None:
        print(f"Suite source: python-openml ({pyopenml.__version__})")
    else:
        print("Suite source: REST fallback (python-openml not available)")

    locked_state = None
    if args.model_path.strip():
        model_path = os.path.abspath(args.model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        locked_state = _extract_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model state: {model_path} ({len(locked_state)} tensors)")
    else:
        print("Model path not provided: evaluating baseline (untrained) TabPFN.")

    rows: list[DatasetEvalResult] = []
    failures: list[dict[str, str]] = []
    for i, did in enumerate(data_ids):
        seed_i = int(args.seed + i * 10007)
        try:
            out = _evaluate_dataset(
                data_id=int(did),
                context_size=int(args.context_size),
                query_size=int(args.query_size),
                seed=seed_i,
                n_estimators=int(args.n_estimators),
                locked_state_dict=locked_state,
            )
            rows.append(out)
            print(
                f"[{i+1:3d}/{len(data_ids)}] did={out.data_id:<6d} "
                f"name={out.name[:22]:<22} "
                f"acc={out.accuracy:.4f} nll={out.nll:.4f} "
                f"(cls={out.n_classes}, ctx_cls={out.context_classes}, unseen_q={out.unseen_query_labels})"
            )
        except Exception as e:
            failures.append({"data_id": str(int(did)), "error": repr(e)})
            print(f"[{i+1:3d}/{len(data_ids)}] did={int(did):<6d} FAILED: {repr(e)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f"{os.path.splitext(args.csv_name)[0]}_{ts}.csv")
    fail_csv_path = os.path.join(args.save_dir, f"{os.path.splitext(args.fail_csv_name)[0]}_{ts}.csv")
    plot_path = os.path.join(args.save_dir, f"{os.path.splitext(args.plot_name)[0]}_{ts}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "data_id",
            "name",
            "n_samples",
            "n_features",
            "n_classes",
            "context_classes",
            "unseen_query_labels",
            "accuracy",
            "nll",
            "ece",
        ])
        for r in rows:
            writer.writerow([
                r.data_id,
                r.name,
                r.n_samples,
                r.n_features,
                r.n_classes,
                r.context_classes,
                r.unseen_query_labels,
                r.accuracy,
                r.nll,
                r.ece,
            ])
    print(f"Saved metrics CSV: {csv_path}")

    with open(fail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data_id", "error"])
        for item in failures:
            writer.writerow([item["data_id"], item["error"]])
    print(f"Saved failures CSV: {fail_csv_path}")

    if len(rows) > 0:
        _plot_summary(
            rows=rows,
            context_size=int(args.context_size),
            query_size=int(args.query_size),
            save_path=plot_path,
        )
        print(f"Saved summary plot: {plot_path}")

        acc = np.asarray([r.accuracy for r in rows], dtype=np.float64)
        nll = np.asarray([r.nll for r in rows], dtype=np.float64)
        print("\n── Summary ──")
        print(f"Success: {len(rows)} / {len(data_ids)}")
        print(f"Accuracy mean±std: {float(acc.mean()):.4f} ± {float(acc.std()):.4f}")
        print(f"NLL mean±std:      {float(nll.mean()):.4f} ± {float(nll.std()):.4f}")
    else:
        print("\nNo successful dataset evaluations. Check failures CSV.")


if __name__ == "__main__":
    main()
