"""
predictive_rule.py  (Step 1)
─────────────────────────────────────────────
TabPFN을 rollout 가능한 형태로 감싸는 모듈.

기존 rollout.py와 달라진 점:
  1. sample() → (y_new, belief) 반환  (SC loss에 belief 필요)
  2. get_belief() 별도 메서드: fit 없이 belief만 조회
  3. underlying model 접근 경로 확보 (→ Step 6 LoRA 주입 대비)

Belief 정의:
  - Classification: π_k(x) ∈ Δ^C  (C-dim probability simplex)
  - Regression:     bar-distribution logits (추후 확장)
"""

import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import jax
import numpy as np
import torch
from jaxtyping import PRNGKeyArray
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.inference import (
    _maybe_run_gpu_preprocessing,
    _prepare_model_inputs,
    get_autocast_context,
)
from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.modality_detection import FeatureModality
from tabpfn.validation import ensure_compatible_predict_input_sklearn

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


# ── Return type ─────────────────────────────────────────────
@dataclass
class RolloutStep:
    """한 rollout step의 결과."""
    x_new: np.ndarray        # (1, d) or (d,) — sampled query point
    y_new: np.ndarray        # scalar or (1,) — sampled label
    belief: np.ndarray       # (C,) for classification, logits for regression


# ── Abstract Base ───────────────────────────────────────────
class PredictiveRule(ABC):
    """TabPFN predictive rule 인터페이스."""

    _locked_state_dict: Optional[dict] = None

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """현재 dataset (D_k)으로 모델 fitting."""
        ...

    @abstractmethod
    def get_belief(self, x_new: np.ndarray) -> np.ndarray:
        """
        fit() 이후 x_new에 대한 predictive belief 반환.
        fit()을 먼저 호출해야 함.
        
        Classification: shape (n_query, C) — probability simplex
        Regression:     shape (n_query, n_bins) — bar-dist logits
        """
        ...

    @abstractmethod
    def sample_y(self, key: PRNGKeyArray, belief: np.ndarray) -> np.ndarray:
        """
        Belief로부터 y를 sampling.
        
        Classification: Categorical(belief)에서 sampling
        Regression:     bar-distribution icdf sampling
        """
        ...

    def sample(
        self,
        key: PRNGKeyArray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> RolloutStep:
        """
        Full rollout step: fit → belief → sample.
        Returns RolloutStep(x_new, y_new, belief).
        """
        self.fit(x_prev, y_prev)
        belief = self.get_belief(x_new)
        y_new = self.sample_y(key, belief)
        return RolloutStep(x_new=x_new, y_new=y_new, belief=belief)

    @property
    @abstractmethod
    def model(self):
        """Underlying TabPFN PyTorch model 접근 (LoRA 주입용)."""
        ...

    def lock_weights(self) -> None:
        """
        현재 모델 weights를 잠금 (LoRA merge 후 호출).

        fit()이 모델 객체를 교체해도 저장된 state_dict를
        새 모델에 복원하여 merged weights를 보존.
        """
        self._locked_state_dict = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
        }

    def unlock_weights(self) -> None:
        """Weight lock 해제."""
        self._locked_state_dict = None

    def _restore_locked_weights(self) -> None:
        """fit() 후 호출: locked weights가 있으면 새 모델에 복원."""
        if self._locked_state_dict is not None:
            device = next(self.model.parameters()).device
            state = {k: v.to(device) for k, v in self._locked_state_dict.items()}
            self.model.load_state_dict(state)

    def get_belief_torch(
        self,
        x_query: np.ndarray,
        x_context: np.ndarray,
        y_context: np.ndarray,
    ) -> torch.Tensor:
        """
        Torch-native belief 계산 (autograd graph 유지).

        fit()과 달리 sklearn wrapper를 우회하고 PyTorch model을 직접 호출.
        LoRA 학습 시 gradient가 흘러야 하므로 이 메서드 사용.

        Args:
            x_query:    (M, d) query features
            x_context:  (N, d) context features
            y_context:  (N,)   context labels

        Returns:
            torch.Tensor (M, C) — predicted probabilities with autograd
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_belief_torch"
        )

    def get_belief_torch_batched(
        self,
        x_query: np.ndarray,
        x_contexts: list[np.ndarray],
        y_contexts: list[np.ndarray],
    ) -> torch.Tensor:
        """
        Batched torch-native belief: B개 context를 한 번에 forward pass.

        Args:
            x_query:     (M, d) shared query points
            x_contexts:  list of B (N, d) context arrays
            y_contexts:  list of B (N,) label arrays

        Returns:
            torch.Tensor (B, M, C) — predicted probabilities with autograd
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_belief_torch_batched"
        )


# ── Classification ──────────────────────────────────────────
class ClassifierPredRule(PredictiveRule):
    """
    TabPFN Classifier를 predictive rule로 사용.
    Belief: π_k(x) = P(y | x, D_k) ∈ Δ^C
    """

    def __init__(
        self,
        categorical_x: list[bool],
        n_estimators: int = 4,
        average_before_softmax: bool = False,
    ):
        cat_idx = [i for i, c in enumerate(categorical_x) if c]
        self._clf = TabPFNClassifier(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            categorical_features_indices=cat_idx,
            fit_mode="low_memory",
            model_path="tabpfn-v2-classifier.ckpt",
        )
        self._classes: Optional[np.ndarray] = None
        self._n_classes: int = 0
        self._locked_state_dict = None

        # Preprocessing cache (populated after fit)
        self._cached_members = []       # list[TabPFNEnsembleMember]
        self._fit_x_ref: Optional[np.ndarray] = None
        self._fit_y_ref: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._clf.fit(x, y)
        self._classes = self._clf.classes_
        self._n_classes = len(self._classes)
        self._fit_x_ref = x
        self._fit_y_ref = y
        expected = np.arange(self._n_classes)
        if not np.array_equal(self._classes, expected):
            raise ValueError(
                f"Classes must be contiguous 0..{self._n_classes-1}, "
                f"got {self._classes}. Apply LabelEncoder first."
            )
        # LoRA merge 후: fit()이 model을 교체해도 weights 복원
        self._restore_locked_weights()
        # Cache preprocessing for get_belief_torch
        self._cache_preprocessing(x, y)

    def _cache_preprocessing(self, x: np.ndarray, y: np.ndarray) -> None:
        """fit() 후 ensemble_preprocessor 결과 캐싱 (seeded, deterministic)."""
        try:
            executor = self._clf.executor_
            ep = executor.ensemble_preprocessor
            seed = int(executor.static_seed)
            rng = np.random.default_rng(seed)

            # Prefer the exact internal tensors/args used by TabPFN inference path.
            try:
                kwargs = {
                    "X_train": executor.X_train,
                    "y_train": executor.y_train,
                    "parallel_mode": "in-order",
                    "override_random_state": rng,
                }
                if hasattr(executor, "feature_schema"):
                    kwargs["feature_schema"] = executor.feature_schema
                elif hasattr(executor, "cat_ix"):
                    kwargs["cat_ix"] = executor.cat_ix
                it = ep.fit_transform_ensemble_members_iterator(**kwargs)
                self._cached_members = list(it)
                return
            except Exception:
                pass

            # Backward-compatible fallback for variants using feature_schema.
            kwargs = {
                "X_train": x,
                "y_train": y,
                "parallel_mode": "in-order",
                "override_random_state": rng,
            }
            if hasattr(self._clf, "inferred_feature_schema_"):
                kwargs["feature_schema"] = self._clf.inferred_feature_schema_
            elif hasattr(self._clf, "inferred_categorical_indices_"):
                kwargs["cat_ix"] = self._clf.inferred_categorical_indices_
            it = ep.fit_transform_ensemble_members_iterator(**kwargs)
            self._cached_members = list(it)
        except Exception:
            # Fallback: preprocessing not available
            self._cached_members = []

    def _is_same_as_fit_context(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
    ) -> bool:
        """Check whether context is exactly the last fit() input."""
        if self._fit_x_ref is None or self._fit_y_ref is None:
            return False

        if x_context is self._fit_x_ref and y_context is self._fit_y_ref:
            return True

        x_arr = np.asarray(x_context)
        y_arr = np.asarray(y_context)
        x_fit = np.asarray(self._fit_x_ref)
        y_fit = np.asarray(self._fit_y_ref)

        # Prefix/suffix slices share memory with fit arrays, but are not the same context.
        # Fast-path is safe only when both arrays are the exact same view layout.
        if x_arr.shape != x_fit.shape or y_arr.shape != y_fit.shape:
            return False
        if x_arr.dtype != x_fit.dtype or y_arr.dtype != y_fit.dtype:
            return False
        if x_arr.strides != x_fit.strides or y_arr.strides != y_fit.strides:
            return False

        x_ptr = int(x_arr.__array_interface__["data"][0])
        y_ptr = int(y_arr.__array_interface__["data"][0])
        x_fit_ptr = int(x_fit.__array_interface__["data"][0])
        y_fit_ptr = int(y_fit.__array_interface__["data"][0])
        return (x_ptr == x_fit_ptr) and (y_ptr == y_fit_ptr)

    def _select_member_model(self, member) -> torch.nn.Module:
        """Use the same per-member model index logic as inference engine."""
        model_index = getattr(member.config, "_model_index", 0)
        models = getattr(self._clf, "models_", None)
        if models is not None and len(models) > model_index:
            return models[model_index]
        return self.model

    def _sanitize_predict_input(self, X) -> np.ndarray:
        """Apply the same predict-time sanitation path as TabPFNClassifier._raw_predict."""
        X = ensure_compatible_predict_input_sklearn(X, self._clf)
        cat_ix = self._clf.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)
        X = fix_dtypes(X, cat_indices=cat_ix)
        X = process_text_na_dataframe(
            X=X,
            ord_encoder=getattr(self._clf, "ordinal_encoder_", None),
        )
        return X

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """Convert torch/array-like to numpy without grad connection."""
        if isinstance(arr, np.ndarray):
            return arr
        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    @staticmethod
    def _member_cat_ix(member) -> list[int]:
        """Resolve categorical indices across TabPFN member variants."""
        feature_schema = getattr(member, "feature_schema", None)
        if feature_schema is not None:
            try:
                return list(feature_schema.indices_for(FeatureModality.CATEGORICAL))
            except Exception:
                pass
        for attr in ("cat_ix", "categorical_features", "categorical_inds", "categorical_indices"):
            value = getattr(member, attr, None)
            if value is not None:
                try:
                    return [int(v) for v in list(value)]
                except Exception:
                    pass
        return []

    def get_belief(self, x_new: np.ndarray) -> np.ndarray:
        """
        Returns: shape (n_query, C) probability array.
        x_new: (1, d) or (n_query, d)
        """
        probs = self._clf.predict_proba(x_new)  # (n_query, C)
        return np.atleast_2d(probs)

    def sample_y(self, key: PRNGKeyArray, belief: np.ndarray) -> np.ndarray:
        """
        Categorical sampling from belief.
        belief: (n_query, C) or (C,)
        Returns: scalar or (n_query,) — integer class indices (not original labels).
        _classes는 inference 시 원래 라벨 복원용으로만 사용.
        """
        belief = np.atleast_2d(belief)
        n_query = belief.shape[0]
        samples = []
        for i in range(n_query):
            subkey = jax.random.fold_in(key, i)
            p = belief[i]
            # numerical stability: renormalize
            p = np.clip(p, 0, None)
            p = p / p.sum()
            idx = jax.random.choice(subkey, a=len(p), p=p)
            samples.append(int(idx))
        return np.array(samples).squeeze()

    @property
    def classes(self) -> Optional[np.ndarray]:
        return self._classes

    @property
    def model(self):
        """TabPFN 내부 PyTorch model 접근."""
        return self._clf.model_ if hasattr(self._clf, 'model_') else self._clf

    def _resolve_class_permutation(self, class_permutation) -> Optional[np.ndarray]:
        """Match TabPFN forward() permutation handling."""
        if class_permutation is None:
            return None

        perm = np.asarray(class_permutation)
        if len(perm) != self._n_classes:
            use_perm = np.arange(self._n_classes)
            use_perm[: len(perm)] = perm
            return use_perm

        return perm

    def _postprocess_probas_torch(self, probas: torch.Tensor) -> torch.Tensor:
        """Match TabPFN _predict_proba() post-processing after logits_to_probabilities()."""
        tuned_thresholds = getattr(self._clf, "tuned_classification_thresholds_", None)
        if tuned_thresholds is not None:
            thresholds_t = torch.as_tensor(
                tuned_thresholds,
                device=probas.device,
                dtype=probas.dtype,
            ).clamp_min(1e-8)
            probas = probas / thresholds_t
            probas = probas / probas.sum(dim=-1, keepdim=True)

        if self._clf.inference_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
            scale = float(10**16)
            probas = torch.round(probas * scale) / scale
            probas = torch.where(
                probas < 1e-3,
                torch.zeros_like(probas),
                probas,
            )

        # Keep the same final normalization semantics as TabPFN _predict_proba().
        return probas / probas.sum(dim=-1, keepdim=True)

    def get_belief_torch(
        self,
        x_query: np.ndarray,
        x_context: np.ndarray,
        y_context: np.ndarray,
    ) -> torch.Tensor:
        """
        Torch-native belief: sklearn 전처리 재현 + autograd 유지.

        Pipeline (sklearn predict_proba와 동일):
          1. cpu_preprocessor.transform()으로 X_context, X_query 전처리
          2. class_permutation으로 y 변환: y_internal = perm[y_original]
          3. model(x_all, y) → raw logits
          4. logits[:, perm] → class 순서 복원
          5. logits_to_probabilities + threshold/precision 후처리

        Args:
            x_query:    (M, d) query features (numpy, raw)
            x_context:  (N, d) context features (numpy, raw — rollout subset)
            y_context:  (N,)   context labels (numpy, 0..C-1)

        Returns:
            torch.Tensor (M, C) — softmax probabilities, autograd intact
        """
        assert self._n_classes > 0, "Call fit() at least once before get_belief_torch"
        assert len(self._cached_members) > 0, (
            "Preprocessing cache not available. Call fit() first."
        )

        pt_model = self.model
        device = next(pt_model.parameters()).device
        forced_dtype = getattr(self._clf, "forced_inference_dtype_", None)
        # Autocast is useful for inference but can destabilize training-time gradients.
        use_autocast = (
            bool(getattr(self._clf, "use_autocast_", False))
            and device.type == "cuda"
            and (not torch.is_grad_enabled())
        )
        was_training = pt_model.training
        pt_model.eval()

        per_member_logits: list[torch.Tensor] = []
        y_ctx_int = np.asarray(y_context).astype(int)
        if y_ctx_int.ndim != 1:
            raise ValueError(f"y_context must be 1D, got shape={y_ctx_int.shape}")
        x_query_sanitized = self._sanitize_predict_input(x_query)
        use_fit_context = self._is_same_as_fit_context(x_context, y_context)
        x_context_sanitized = None
        if not use_fit_context:
            x_context_sanitized = self._sanitize_predict_input(x_context)
        model_training_flags: dict[int, tuple[torch.nn.Module, bool]] = {}
        for member in self._cached_members:
            m = self._select_member_model(member)
            key = id(m)
            if key not in model_training_flags:
                model_training_flags[key] = (m, m.training)
                m.eval()

        try:
            # ── 1. CPU preprocessing + per-member forward ──
            for member in self._cached_members:
                use_perm = self._resolve_class_permutation(
                    member.config.class_permutation
                )
                pt_model = self._select_member_model(member)
                device = next(pt_model.parameters()).device

                # Exact parity path: reuse the preprocessed train tensors produced by fit().
                if use_fit_context and hasattr(member, "X_train") and hasattr(member, "y_train"):
                    X_ctx_proc = self._to_numpy(member.X_train)
                    y_proc = self._to_numpy(member.y_train)
                else:
                    assert x_context_sanitized is not None
                    X_ctx_proc = member.cpu_preprocessor.transform(x_context_sanitized).X
                    if use_perm is None:
                        y_proc = y_ctx_int
                    else:
                        y_proc = use_perm[y_ctx_int]

                    subsample_ix = getattr(member.config, "subsample_ix", None)
                    if subsample_ix is not None and len(subsample_ix) <= len(X_ctx_proc):
                        X_ctx_proc = X_ctx_proc[subsample_ix]
                        y_proc = y_proc[subsample_ix]

                if hasattr(member, "transform_X_test"):
                    X_q_proc = self._to_numpy(member.transform_X_test(x_query_sanitized))
                else:
                    X_q_proc = member.cpu_preprocessor.transform(x_query_sanitized).X

                x_t, y_t = _prepare_model_inputs(
                    device,
                    forced_dtype,
                    X_ctx_proc,
                    X_q_proc,
                    y_proc,
                )
                x_t = _maybe_run_gpu_preprocessing(
                    x_t,
                    gpu_preprocessor=getattr(member, "gpu_preprocessor", None),
                    num_train_rows=X_ctx_proc.shape[0],
                )
                inference_ctx = (
                    nullcontext()
                    if torch.is_grad_enabled()
                    else torch.inference_mode()
                )
                with get_autocast_context(device, enabled=use_autocast), inference_ctx:
                    out = pt_model(
                        x_t,
                        y_t,
                        only_return_standard_out=True,
                        categorical_inds=[self._member_cat_ix(member)],
                    )

                if out.ndim == 2:
                    processed_out = out
                elif out.ndim == 3:
                    processed_out = out[:, 0, :]
                else:
                    raise ValueError(
                        f"Expected model output with ndim 2 or 3, got {out.ndim}"
                    )

                if use_perm is None:
                    logits = processed_out[:, : self._n_classes]
                else:
                    logits = processed_out[:, use_perm]

                per_member_logits.append(logits)
        finally:
            for model, was_training_model in model_training_flags.values():
                model.train(was_training_model)
            pt_model.train(was_training)

        # ── 2. Match TabPFN post-processing path ──
        raw_logits = torch.stack(per_member_logits, dim=0)  # (E, M, C)
        probs = self._clf.logits_to_probabilities(raw_logits)
        return self._postprocess_probas_torch(probs)

    def get_belief_torch_batched(
        self,
        x_query: np.ndarray,
        x_contexts: list[np.ndarray],
        y_contexts: list[np.ndarray],
    ) -> torch.Tensor:
        """
        Batched torch-native belief: B개 context를 batch dim으로 묶어 1회 forward.

        같은 depth의 B개 continuation을 한 번에 처리하여 GPU 활용률을 높임.
        모든 context는 동일한 길이(N)여야 함 (같은 depth에서 추출).

        Args:
            x_query:     (M, d) shared query points (numpy, raw)
            x_contexts:  list of B (N, d) context feature arrays
            y_contexts:  list of B (N,) context label arrays

        Returns:
            torch.Tensor (B, M, C) — softmax probabilities, autograd intact
        """
        B = len(x_contexts)
        assert B == len(y_contexts)
        assert B > 0, "Empty context list"
        assert self._n_classes > 0, "Call fit() at least once before get_belief_torch_batched"
        assert len(self._cached_members) > 0, (
            "Preprocessing cache not available. Call fit() first."
        )

        pt_model = self.model
        device = next(pt_model.parameters()).device
        forced_dtype = getattr(self._clf, "forced_inference_dtype_", None)
        input_dtype = forced_dtype if forced_dtype is not None else torch.float32
        use_autocast = (
            bool(getattr(self._clf, "use_autocast_", False))
            and device.type == "cuda"
            and (not torch.is_grad_enabled())
        )
        was_training = pt_model.training
        pt_model.eval()

        per_member_logits: list[torch.Tensor] = []
        y_contexts_int = [np.asarray(yc).astype(int) for yc in y_contexts]
        for yc in y_contexts_int:
            if yc.ndim != 1:
                raise ValueError(f"Each y_context must be 1D, got shape={yc.shape}")
        x_query_sanitized = self._sanitize_predict_input(x_query)
        fit_match_per_context = [
            self._is_same_as_fit_context(xc, yc)
            for xc, yc in zip(x_contexts, y_contexts)
        ]
        x_contexts_sanitized = [
            None if is_fit else self._sanitize_predict_input(xc)
            for xc, is_fit in zip(x_contexts, fit_match_per_context)
        ]
        model_training_flags: dict[int, tuple[torch.nn.Module, bool]] = {}
        for member in self._cached_members:
            m = self._select_member_model(member)
            key = id(m)
            if key not in model_training_flags:
                model_training_flags[key] = (m, m.training)
                m.eval()

        try:
            for member in self._cached_members:
                use_perm = self._resolve_class_permutation(
                    member.config.class_permutation
                )
                pt_model = self._select_member_model(member)
                device = next(pt_model.parameters()).device

                # Query 전처리는 1번만 (모든 batch에서 공유)
                if hasattr(member, "transform_X_test"):
                    X_q_proc = self._to_numpy(member.transform_X_test(x_query_sanitized))
                else:
                    X_q_proc = member.cpu_preprocessor.transform(x_query_sanitized).X

                # B개 context 전처리 + batch 구성
                x_batch_list = []
                y_batch_list = []
                cat_ix = self._member_cat_ix(member)
                for b in range(B):
                    same_as_fit = fit_match_per_context[b]
                    if same_as_fit and hasattr(member, "X_train") and hasattr(member, "y_train"):
                        X_ctx_proc = self._to_numpy(member.X_train)
                        y_proc = self._to_numpy(member.y_train)
                    else:
                        x_context_sanitized = x_contexts_sanitized[b]
                        assert x_context_sanitized is not None
                        X_ctx_proc = member.cpu_preprocessor.transform(x_context_sanitized).X
                        if use_perm is None:
                            y_proc = y_contexts_int[b]
                        else:
                            y_proc = use_perm[y_contexts_int[b]]

                        subsample_ix = getattr(member.config, "subsample_ix", None)
                        if subsample_ix is not None and len(subsample_ix) <= len(X_ctx_proc):
                            X_ctx_proc = X_ctx_proc[subsample_ix]
                            y_proc = y_proc[subsample_ix]

                    y_proc = np.asarray(y_proc)
                    X_all = np.concatenate([X_ctx_proc, X_q_proc], axis=0)  # (N+M, d_proc)
                    x_batch_list.append(torch.as_tensor(X_all, dtype=input_dtype))
                    y_batch_list.append(torch.as_tensor(y_proc, dtype=input_dtype))

                # Stack: x → (N+M, B, d_proc), y → (N, B)
                x_t = torch.stack(x_batch_list, dim=1).to(device)
                y_t = torch.stack(y_batch_list, dim=1).to(device)

                num_train_rows = x_batch_list[0].shape[0] - X_q_proc.shape[0]
                x_t = _maybe_run_gpu_preprocessing(
                    x_t,
                    gpu_preprocessor=getattr(member, "gpu_preprocessor", None),
                    num_train_rows=num_train_rows,
                )
                inference_ctx = (
                    nullcontext()
                    if torch.is_grad_enabled()
                    else torch.inference_mode()
                )
                with get_autocast_context(device, enabled=use_autocast), inference_ctx:
                    out = pt_model(
                        x_t,
                        y_t,
                        only_return_standard_out=True,
                        categorical_inds=[cat_ix for _ in range(B)],
                    )

                if out.ndim != 3:
                    raise ValueError(f"Expected model output with ndim 3, got {out.ndim}")

                if use_perm is None:
                    logits = out[:, :, : self._n_classes]
                else:
                    logits = out[:, :, use_perm]

                # (M, B, C) -> (B, M, C)
                per_member_logits.append(logits.permute(1, 0, 2))
        finally:
            for model, was_training_model in model_training_flags.values():
                model.train(was_training_model)
            pt_model.train(was_training)

        # Ensemble aggregation + post-processing path identical to TabPFN semantics.
        raw_logits = torch.stack(per_member_logits, dim=0)  # (E, B, M, C)
        probs = self._clf.logits_to_probabilities(raw_logits)
        return self._postprocess_probas_torch(probs)


# ── Regression ──────────────────────────────────────────────
class RegressorPredRule(PredictiveRule):
    """
    TabPFN Regressor를 predictive rule로 사용.

    Belief: bar-distribution over 5000 bins (softmax of raw logits).
    - SC loss: 5000-dim distribution 간 CE → classification과 동일 구조
    - Supervised loss: y를 z-normalize → bin index → NLL

    학습 시:
    - get_belief_torch: autograd 유지, sklearn 우회
    - _borders, _y_mean, _y_std: fit 시 캐싱 (bin discretization용)
    """

    def __init__(
        self,
        categorical_x: list[bool],
        n_estimators: int = 8,
        average_before_softmax: bool = False,
    ):
        cat_idx = [i for i, c in enumerate(categorical_x) if c]
        self._reg = TabPFNRegressor(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=1.0,
            categorical_features_indices=cat_idx,
            fit_mode="low_memory",
            model_path="tabpfn-v2-regressor.ckpt",
        )
        self._last_pred = None
        self._locked_state_dict = None
        # z-norm stats (cached after fit)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._borders: Optional[np.ndarray] = None  # (n_bins+1,) in z-norm space
        self._n_bins: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._reg.fit(x, y)
        # Cache z-norm stats for bin discretization
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y))
        if self._y_std < 1e-8:
            self._y_std = 1.0
        # Cache borders from bar distribution
        bardist = self._reg.znorm_space_bardist_
        self._borders = bardist.borders.cpu().numpy()  # (n_bins+1,)
        self._n_bins = len(self._borders) - 1
        # LoRA merge 후: fit()이 model을 교체해도 weights 복원
        self._restore_locked_weights()

    def get_belief(self, x_new: np.ndarray) -> np.ndarray:
        """
        Returns: bar-distribution logits, shape (n_query, n_bins).
        Side-effect: caches full prediction for icdf sampling.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred = self._reg.predict(x_new, output_type="full")
        self._last_pred = pred
        return pred["logits"]  # (n_query, n_bins)

    def sample_y(self, key: PRNGKeyArray, belief: np.ndarray) -> np.ndarray:
        """
        Bar-distribution sampling.
        belief: logits (n_query, n_bins)
        Returns: continuous y values (sampled from bar-dist)
        """
        # Softmax → probabilities
        belief_t = torch.from_numpy(np.asarray(belief)).float()
        probs = torch.softmax(belief_t, dim=-1)  # (n_query, n_bins)

        n_query = probs.shape[0]
        samples = []
        for i in range(n_query):
            subkey = jax.random.fold_in(key, i)
            p = probs[i].numpy()
            p = np.clip(p, 0, None)
            p = p / p.sum()
            # Sample bin index
            bin_idx = int(jax.random.choice(subkey, a=len(p), p=p))
            # Uniform within bin → continuous y (z-norm space)
            lo = self._borders[bin_idx]
            hi = self._borders[bin_idx + 1]
            subkey2 = jax.random.fold_in(key, i + n_query)
            u = float(jax.random.uniform(subkey2))
            y_znorm = lo + u * (hi - lo)
            # De-normalize
            y_val = y_znorm * self._y_std + self._y_mean
            samples.append(y_val)
        return np.array(samples).squeeze()

    def get_belief_torch(
        self,
        x_query: np.ndarray,
        x_context: np.ndarray,
        y_context: np.ndarray,
    ) -> torch.Tensor:
        """
        Torch-native belief for regression: autograd 유지.

        Forward: model(x_all, y_in) → (M, 1, n_bins) raw logits
        → softmax → (M, n_bins) bar-dist probabilities

        주의: y_context는 내부에서 z-normalize 후 모델에 전달.
        (sklearn wrapper가 하는 것과 동일하게 처리)

        Args:
            x_query:    (M, d)
            x_context:  (N, d)
            y_context:  (N,) — original scale (z-norm은 내부에서 처리)

        Returns:
            torch.Tensor (M, n_bins) — bar-dist probabilities, autograd intact
        """
        assert self._n_bins > 0, "Call fit() at least once before get_belief_torch"

        pt_model = self.model
        device = next(pt_model.parameters()).device

        x_ctx = torch.from_numpy(np.asarray(x_context)).float().to(device)
        y_raw = torch.from_numpy(np.asarray(y_context)).float().to(device)
        x_q = torch.from_numpy(np.asarray(x_query)).float().to(device)

        # z-normalize y (모델은 z-normalized y를 기대)
        y_ctx = (y_raw - self._y_mean) / self._y_std

        x_all = torch.cat([x_ctx, x_q], dim=0).unsqueeze(1)  # (N+M, 1, d)
        y_in = y_ctx.unsqueeze(1)                              # (N, 1)

        logits = pt_model(x_all, y_in)  # (M, 1, n_bins)
        logits = logits[:, 0, :]         # (M, n_bins)

        probs = torch.softmax(logits, dim=-1)  # (M, n_bins)
        return probs

    def get_belief_torch_batched(
        self,
        x_query: np.ndarray,
        x_contexts: list[np.ndarray],
        y_contexts: list[np.ndarray],
    ) -> torch.Tensor:
        """
        Batched torch-native belief for regression: B개 context를 1회 forward.

        Args:
            x_query:     (M, d) shared query points
            x_contexts:  list of B (N, d) context arrays
            y_contexts:  list of B (N,) label arrays

        Returns:
            torch.Tensor (B, M, n_bins) — bar-dist probabilities, autograd intact
        """
        B = len(x_contexts)
        assert B == len(y_contexts)
        assert B > 0, "Empty context list"
        assert self._n_bins > 0, "Call fit() at least once before get_belief_torch_batched"

        pt_model = self.model
        device = next(pt_model.parameters()).device

        x_batch_list = []
        y_batch_list = []
        for b in range(B):
            x_ctx = torch.from_numpy(np.asarray(x_contexts[b])).float()
            y_raw = torch.from_numpy(np.asarray(y_contexts[b])).float()
            x_q = torch.from_numpy(np.asarray(x_query)).float()

            y_ctx = (y_raw - self._y_mean) / self._y_std
            x_all = torch.cat([x_ctx, x_q], dim=0).unsqueeze(1)  # (N+M, 1, d)
            x_batch_list.append(x_all)
            y_batch_list.append(y_ctx)

        # (N+M, B, d)
        x_t = torch.cat(x_batch_list, dim=1).to(device)
        # (N, B)
        y_t = torch.stack(y_batch_list, dim=1).to(device)

        logits = pt_model(x_t, y_t)  # (M, B, n_bins)
        logits = logits.permute(1, 0, 2)  # (B, M, n_bins)

        probs = torch.softmax(logits, dim=-1)
        return probs

    def y_to_bin_index(self, y: np.ndarray) -> np.ndarray:
        """
        Continuous y → bar-dist bin index.
        z-normalize → searchsorted on borders.

        Args:
            y: (N,) continuous target values

        Returns:
            (N,) integer bin indices, clipped to [0, n_bins-1]
        """
        assert self._borders is not None, "Call fit() first"
        y_z = (np.asarray(y) - self._y_mean) / self._y_std
        # searchsorted: find bin where borders[i] <= y_z < borders[i+1]
        bin_idx = np.searchsorted(self._borders, y_z, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, self._n_bins - 1)
        return bin_idx.astype(int)

    @property
    def model(self):
        """TabPFN 내부 PyTorch model 접근."""
        return self._reg.model_ if hasattr(self._reg, 'model_') else self._reg

    @property
    def borders(self) -> Optional[np.ndarray]:
        return self._borders

    @property
    def y_stats(self) -> tuple[float, float]:
        return self._y_mean, self._y_std


# ── Empirical X Sampling ────────────────────────────────────
def sample_x_empirical(
    key: PRNGKeyArray,
    x: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    """
    Empirical feature distribution에서 x 샘플링.
    PDF: x_{k+1} ~ F_k (empirical feature distribution)
    
    Args:
        key: JAX PRNG key
        x: (N, d) current dataset features
        n: number of points to sample
    Returns:
        (n, d) sampled feature vectors
    """
    indices = jax.random.randint(key, shape=(n,), minval=0, maxval=x.shape[0])
    return x[np.array(indices)]


# ── Factory ─────────────────────────────────────────────────
def make_predictive_rule(
    task_type: str,
    categorical_x: list[bool],
    n_estimators: int = 4,
    average_before_softmax: bool = False,
) -> PredictiveRule:
    """
    Factory function for predictive rule 생성.
    
    Args:
        task_type: "classification" or "regression"
        categorical_x: 각 feature의 categorical 여부
        n_estimators: TabPFN ensemble 수
        average_before_softmax: ensemble averaging 방식
    """
    if task_type == "classification":
        return ClassifierPredRule(categorical_x, n_estimators, average_before_softmax)
    elif task_type == "regression":
        return RegressorPredRule(categorical_x, n_estimators, average_before_softmax)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
