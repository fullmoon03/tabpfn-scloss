import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


# =========================================================
# Config
# =========================================================

@dataclass
class MixtureConfig:
    mode_name: str = "mixed_full"
    # task mixture weights
    p_gbdt: float = 0.10
    p_scm: float = 0.25
    p_smooth_mlp: float = 0.30
    p_sparse_linear: float = 0.35
    p_nonlinear_link: float = 0.0

    # task size / shape
    # Defaults match the fixed synthetic tensor shape expected in this repo.
    n_features_min: int = 10
    n_features_max: int = 10
    n_samples_min: int = 250
    n_samples_max: int = 250
    n_classes_min: int = 5
    n_classes_max: int = 5
    allow_variable_shapes: bool = False

    # uncertainty / noise
    temperature_min: float = 0.42
    temperature_max: float = 1.00
    label_noise_min: float = 0.00
    label_noise_max: float = 0.05
    deterministic_label_prob: float = 0.65

    # class imbalance: choose Dirichlet alpha from this set per task
    dirichlet_alpha_choices: Tuple[float, ...] = (6.0, 10.0, 14.0)

    # feature sampler mixture
    p_correlated_gaussian: float = 0.6
    p_independent_hetero: float = 0.2
    p_heavy_tail: float = 0.2
    feature_mode: str = "mixed"  # "mixed" | "iid_standard_normal"
    class_prior_mode: str = "dirichlet"  # "dirichlet" | "uniform"

    # SCM details
    scm_max_parents: int = 2
    scm_noise_scale_min: float = 0.03
    scm_noise_scale_max: float = 0.14

    # smooth MLP details
    mlp_hidden_min: int = 12
    mlp_hidden_max: int = 32

    # sparse linear details
    informative_min: int = 4
    informative_max: int = 7
    pairwise_interaction_prob: float = 0.12

    # nonlinear-link details
    nonlinear_beta_min: float = -2.0
    nonlinear_beta_max: float = 3.0
    nonlinear_gmm_a_choices: Tuple[float, ...] = (0.0, -1.0, -2.0)
    nonlinear_link_kind: str = "mixed"  # "mixed" | "logistic" | "gmm"
    nonlinear_fixed_gmm_a: Optional[float] = None
    nonlinear_fixed_beta_seed: Optional[int] = 1058

    # global difficulty / margin
    logit_scale_min: float = 1.6
    logit_scale_max: float = 3.0

    # GBDT details
    gbdt_estimators_min: int = 12
    gbdt_estimators_max: int = 24
    gbdt_depth_min: int = 1
    gbdt_depth_max: int = 2
    gbdt_lr_min: float = 0.05
    gbdt_lr_max: float = 0.18
    gbdt_subsample_min: float = 0.85
    gbdt_subsample_max: float = 1.0

    def __post_init__(self) -> None:
        if self.n_features_min > self.n_features_max:
            raise ValueError("n_features_min must be <= n_features_max")
        if self.n_samples_min > self.n_samples_max:
            raise ValueError("n_samples_min must be <= n_samples_max")
        if self.n_classes_min > self.n_classes_max:
            raise ValueError("n_classes_min must be <= n_classes_max")
        if not self.allow_variable_shapes:
            if self.n_features_min != self.n_features_max:
                raise ValueError("Fixed-shape mode requires n_features_min == n_features_max")
            if self.n_samples_min != self.n_samples_max:
                raise ValueError("Fixed-shape mode requires n_samples_min == n_samples_max")
            if self.n_classes_min != self.n_classes_max:
                raise ValueError("Fixed-shape mode requires n_classes_min == n_classes_max")
        if self.feature_mode not in {"mixed", "iid_standard_normal"}:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
        if self.class_prior_mode not in {"dirichlet", "uniform"}:
            raise ValueError(f"Unknown class_prior_mode: {self.class_prior_mode}")
        if self.nonlinear_link_kind not in {"mixed", "logistic", "gmm"}:
            raise ValueError(f"Unknown nonlinear_link_kind: {self.nonlinear_link_kind}")


def make_mixture_config(mode: str = "mixed_full") -> MixtureConfig:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "mixed_full":
        return MixtureConfig(mode_name="mixed_full")
    if mode_norm == "linear_mix":
        return MixtureConfig(
            mode_name="linear_mix",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=1.0,
            p_nonlinear_link=0.0,
        )
    if mode_norm == "scm_mix":
        return MixtureConfig(
            mode_name="scm_mix",
            p_gbdt=0.0,
            p_scm=1.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=0.0,
        )
    if mode_norm == "nonlinear_link":
        return MixtureConfig(
            mode_name="nonlinear_link",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=1.0,
            n_classes_min=2,
            n_classes_max=2,
            temperature_min=1.0,
            temperature_max=1.0,
            label_noise_min=0.0,
            label_noise_max=0.0,
            deterministic_label_prob=0.0,
            class_prior_mode="uniform",
        )
    if mode_norm == "nonlinear_link_mix":
        return MixtureConfig(
            mode_name="nonlinear_link_mix",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=1.0,
            n_classes_min=2,
            n_classes_max=2,
            temperature_min=1.0,
            temperature_max=1.0,
            label_noise_min=0.0,
            label_noise_max=0.0,
            deterministic_label_prob=0.0,
            class_prior_mode="uniform",
        )
    if mode_norm == "nonlinear_link_logistic":
        return MixtureConfig(
            mode_name="nonlinear_link_logistic",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=1.0,
            n_classes_min=2,
            n_classes_max=2,
            temperature_min=1.0,
            temperature_max=1.0,
            label_noise_min=0.0,
            label_noise_max=0.0,
            deterministic_label_prob=0.0,
            class_prior_mode="uniform",
            nonlinear_link_kind="logistic",
        )
    if mode_norm == "nonlinear_link_gmm0":
        return MixtureConfig(
            mode_name="nonlinear_link_gmm0",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=1.0,
            n_classes_min=2,
            n_classes_max=2,
            temperature_min=1.0,
            temperature_max=1.0,
            label_noise_min=0.0,
            label_noise_max=0.0,
            deterministic_label_prob=0.0,
            class_prior_mode="uniform",
            nonlinear_link_kind="gmm",
            nonlinear_fixed_gmm_a=0.0,
        )
    if mode_norm == "nonlinear_link_gmm_neg1":
        return MixtureConfig(
            mode_name="nonlinear_link_gmm_neg1",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=1.0,
            n_classes_min=2,
            n_classes_max=2,
            temperature_min=1.0,
            temperature_max=1.0,
            label_noise_min=0.0,
            label_noise_max=0.0,
            deterministic_label_prob=0.0,
            class_prior_mode="uniform",
            nonlinear_link_kind="gmm",
            nonlinear_fixed_gmm_a=-1.0,
        )
    if mode_norm == "nonlinear_link_gmm_neg2":
        return MixtureConfig(
            mode_name="nonlinear_link_gmm_neg2",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=1.0,
            n_classes_min=2,
            n_classes_max=2,
            temperature_min=1.0,
            temperature_max=1.0,
            label_noise_min=0.0,
            label_noise_max=0.0,
            deterministic_label_prob=0.0,
            class_prior_mode="uniform",
            nonlinear_link_kind="gmm",
            nonlinear_fixed_gmm_a=-2.0,
        )
    if mode_norm == "simple_linear":
        return MixtureConfig(
            mode_name="simple_linear",
            p_gbdt=0.0,
            p_scm=0.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=1.0,
            temperature_min=0.75,
            temperature_max=0.75,
            label_noise_min=0.01,
            label_noise_max=0.01,
            deterministic_label_prob=0.98,
            dirichlet_alpha_choices=(1.0,),
            p_correlated_gaussian=0.0,
            p_independent_hetero=1.0,
            p_heavy_tail=0.0,
            feature_mode="iid_standard_normal",
            class_prior_mode="uniform",
            informative_min=4,
            informative_max=4,
            pairwise_interaction_prob=0.0,
            logit_scale_min=2.2,
            logit_scale_max=2.2,
        )
    if mode_norm == "scm":
        return MixtureConfig(
            mode_name="scm",
            p_gbdt=0.0,
            p_scm=1.0,
            p_smooth_mlp=0.0,
            p_sparse_linear=0.0,
            p_nonlinear_link=0.0,
        )
    raise ValueError(
        f"Unknown synthetic generator mode: {mode}. "
        "Expected one of: nonlinear_link, nonlinear_link_logistic, nonlinear_link_gmm0, "
        "nonlinear_link_gmm_neg1, nonlinear_link_gmm_neg2, nonlinear_link_mix, "
        "scm, scm_mix, simple_linear, linear_mix, mixed_full."
    )


def _make_linear_mix_subconfig(setting_name: str) -> MixtureConfig:
    cfg = make_mixture_config("simple_linear")
    cfg.mode_name = str(setting_name)

    if setting_name == "simple_linear":
        return cfg
    if setting_name == "clean_linear":
        cfg.label_noise_min = 0.0
        cfg.label_noise_max = 0.0
        cfg.deterministic_label_prob = 1.0
        cfg.logit_scale_min = 2.0
        cfg.logit_scale_max = 2.0
        return cfg
    if setting_name == "corr_gaussian":
        cfg.feature_mode = "mixed"
        cfg.p_correlated_gaussian = 1.0
        cfg.p_independent_hetero = 0.0
        cfg.p_heavy_tail = 0.0
        return cfg
    if setting_name == "corr_low_margin_clean":
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
    raise ValueError(f"Unknown linear mix sub-setting: {setting_name}")


def _make_scm_mix_subconfig(setting_name: str) -> MixtureConfig:
    cfg = make_mixture_config("scm")
    cfg.mode_name = str(setting_name)

    if setting_name == "scm_parent2":
        cfg.scm_max_parents = 2
        return cfg
    if setting_name == "scm_parent2_alpha2_4":
        cfg.scm_max_parents = 2
        cfg.dirichlet_alpha_choices = (2.0, 4.0)
        return cfg
    if setting_name == "scm_parent3":
        cfg.scm_max_parents = 3
        return cfg
    if setting_name == "scm_parent3_alpha2_4":
        cfg.scm_max_parents = 3
        cfg.dirichlet_alpha_choices = (2.0, 4.0)
        return cfg
    raise ValueError(f"Unknown scm mix sub-setting: {setting_name}")


def _make_nonlinear_link_mix_subconfig(setting_name: str) -> MixtureConfig:
    if setting_name == "nonlinear_link_logistic":
        return make_mixture_config("nonlinear_link_logistic")
    if setting_name == "nonlinear_link_gmm0":
        return make_mixture_config("nonlinear_link_gmm0")
    if setting_name == "nonlinear_link_gmm_neg1":
        return make_mixture_config("nonlinear_link_gmm_neg1")
    raise ValueError(f"Unknown nonlinear-link mix sub-setting: {setting_name}")


def _generate_composite_mode_task(
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    if cfg.mode_name == "linear_mix":
        setting_name = str(rng.choice((
            "simple_linear",
            "clean_linear",
            "corr_gaussian",
            "corr_low_margin_clean",
        )))
        subcfg = _make_linear_mix_subconfig(setting_name)
    elif cfg.mode_name == "scm_mix":
        setting_name = str(rng.choice((
            "scm_parent2",
            "scm_parent2_alpha2_4",
            "scm_parent3",
            "scm_parent3_alpha2_4",
        )))
        subcfg = _make_scm_mix_subconfig(setting_name)
    elif cfg.mode_name == "nonlinear_link_mix":
        setting_name = str(rng.choice((
            "nonlinear_link_logistic",
            "nonlinear_link_gmm0",
            "nonlinear_link_gmm_neg1",
        )))
        subcfg = _make_nonlinear_link_mix_subconfig(setting_name)
    else:
        return None

    x, y, meta = generate_mixture_task(subcfg, rng)
    meta["mode_name"] = cfg.mode_name
    meta["setting_name"] = setting_name
    return x, y, meta


# =========================================================
# Utilities
# =========================================================

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    return probs


def _sample_labels_from_probs(
    probs: np.ndarray,
    rng: np.random.Generator,
    class_prior: Optional[np.ndarray] = None,
    label_noise: float = 0.0,
    deterministic_label_prob: float = 0.0,
) -> np.ndarray:
    """
    probs: (N, C)
    class_prior: optional (C,), used as a weak reweighting prior
    """
    probs = probs.copy()

    if class_prior is not None:
        probs = probs * class_prior[None, :]
        probs = probs / probs.sum(axis=1, keepdims=True)

    n, c = probs.shape
    y = np.argmax(probs, axis=1).astype(np.int64)

    if deterministic_label_prob < 1.0:
        sample_mask = rng.random(n) > deterministic_label_prob
        if np.any(sample_mask):
            for i in np.where(sample_mask)[0]:
                y[i] = rng.choice(c, p=probs[i])

    if label_noise > 0:
        flip_mask = rng.random(n) < label_noise
        if np.any(flip_mask):
            y[flip_mask] = rng.integers(0, c, size=flip_mask.sum())

    return y


def _random_dirichlet_prior(n_classes: int, rng: np.random.Generator, alpha_choices: Tuple[float, ...]) -> np.ndarray:
    alpha = float(rng.choice(alpha_choices))
    return rng.dirichlet(alpha * np.ones(n_classes))


def _sample_seed_labels_cover_all_classes(
    n_samples: int,
    n_classes: int,
    class_prior: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_samples < n_classes:
        raise ValueError(
            f"Need at least one sample per class for GBDT seed labels, got n_samples={n_samples}, "
            f"n_classes={n_classes}"
        )

    y = np.empty(n_samples, dtype=np.int64)
    y[:n_classes] = np.arange(n_classes, dtype=np.int64)
    if n_samples > n_classes:
        y[n_classes:] = rng.choice(n_classes, size=n_samples - n_classes, p=class_prior)
    rng.shuffle(y)
    return y


def _sample_logit_scale(cfg: MixtureConfig, rng: np.random.Generator) -> float:
    return float(rng.uniform(cfg.logit_scale_min, cfg.logit_scale_max))


def _make_random_spd_cov(d: int, rng: np.random.Generator, jitter: float = 1e-3) -> np.ndarray:
    a = rng.normal(size=(d, d))
    cov = a @ a.T
    cov = cov / np.trace(cov) * d
    cov = cov + jitter * np.eye(d)
    return cov


def _sample_multivariate_student_t(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
    df: float,
) -> np.ndarray:
    """
    Sample from a zero-mean multivariate Student-t with random SPD scale.
    Uses the Gaussian / chi-square scale-mixture representation.
    """
    cov = _make_random_spd_cov(n_features, rng)
    z = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov, size=n_samples)
    g = rng.chisquare(df, size=(n_samples, 1))
    scale = np.sqrt(np.clip(g / df, 1e-12, None))
    return z / scale


def _one_hot_argmax(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def _expand_gbdt_logits_to_all_classes(
    logits: np.ndarray,
    classes_: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    classes_ = np.asarray(classes_, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)

    if logits.ndim != 2 or logits.shape[1] != len(classes_):
        raise ValueError(
            f"Unexpected GBDT logits shape {logits.shape} for classes {classes_.tolist()}"
        )

    full_logits = np.full((logits.shape[0], n_classes), -1e9, dtype=np.float64)
    full_logits[:, classes_] = logits
    return full_logits


# =========================================================
# Feature samplers
# =========================================================

def sample_features(
    n_samples: int,
    n_features: int,
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mixture over feature distributions:
      1) correlated Gaussian
      2) independent heteroscedastic Gaussian
      3) heavy-tail / skew transform
    """
    if cfg.feature_mode == "iid_standard_normal":
        x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
        return x.astype(np.float32)

    u = rng.random()

    if u < cfg.p_correlated_gaussian:
        cov = _make_random_spd_cov(n_features, rng)
        mean = rng.normal(scale=0.5, size=n_features)
        x = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)

    elif u < cfg.p_correlated_gaussian + cfg.p_independent_hetero:
        scales = np.exp(rng.normal(loc=0.0, scale=0.5, size=n_features))
        means = rng.normal(scale=0.5, size=n_features)
        x = rng.normal(loc=means, scale=scales, size=(n_samples, n_features))

        # add some irrelevant / near-constant dims occasionally
        if n_features >= 4 and rng.random() < 0.4:
            k = rng.integers(1, max(2, n_features // 4))
            idx = rng.choice(n_features, size=k, replace=False)
            x[:, idx] = x[:, idx] * rng.uniform(0.02, 0.15)

    else:
        # Multivariate Student-t with task-specific degrees of freedom.
        df = float(rng.uniform(3.0, 6.0))
        x = _sample_multivariate_student_t(
            n_samples=n_samples,
            n_features=n_features,
            rng=rng,
            df=df,
        )

        # Add skew only to a subset of features so tails remain heterogeneous.
        if n_features >= 2:
            n_skew = int(rng.integers(1, max(2, n_features // 2) + 1))
            skew_idx = rng.choice(n_features, size=n_skew, replace=False)
            skew_mode = str(rng.choice(["exp", "softplus"]))
            skew_strength = float(rng.uniform(0.12, 0.35))
            x_sub = x[:, skew_idx]
            if skew_mode == "exp":
                x[:, skew_idx] = x_sub + skew_strength * np.expm1(np.clip(x_sub, -3.0, 3.0))
            else:
                x[:, skew_idx] = x_sub + skew_strength * np.log1p(np.exp(x_sub))

    # standardize per feature to keep scales stable across priors
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    return x.astype(np.float32)


# =========================================================
# Prior 1: GBDT
# =========================================================

def generate_gbdt_task(
    n_samples: int,
    n_features: int,
    n_classes: int,
    temperature: float,
    label_noise: float,
    class_prior: np.ndarray,
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x = sample_features(n_samples, n_features, cfg, rng)

    # pseudo-labels to induce a tree-structured mapping
    y_seed = _sample_seed_labels_cover_all_classes(
        n_samples=n_samples,
        n_classes=n_classes,
        class_prior=class_prior,
        rng=rng,
    )

    clf = GradientBoostingClassifier(
        n_estimators=int(rng.integers(cfg.gbdt_estimators_min, cfg.gbdt_estimators_max + 1)),
        learning_rate=float(rng.uniform(cfg.gbdt_lr_min, cfg.gbdt_lr_max)),
        max_depth=int(rng.integers(cfg.gbdt_depth_min, cfg.gbdt_depth_max + 1)),
        subsample=float(rng.uniform(cfg.gbdt_subsample_min, cfg.gbdt_subsample_max)),
        random_state=int(rng.integers(0, 1_000_000_000)),
    )
    clf.fit(x, y_seed)

    logits = _expand_gbdt_logits_to_all_classes(
        logits=clf.decision_function(x),
        classes_=clf.classes_,
        n_classes=n_classes,
    )
    logits = logits * _sample_logit_scale(cfg, rng)

    probs = np.asarray(clf.predict_proba(x), dtype=np.float64)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    probs = _softmax(np.log(np.clip(probs, 1e-12, None)) * _sample_logit_scale(cfg, rng), temperature=temperature)
    y = _sample_labels_from_probs(
        probs,
        rng,
        class_prior=class_prior,
        label_noise=label_noise,
        deterministic_label_prob=cfg.deterministic_label_prob,
    )

    meta = {
        "prior_type": "gbdt",
        "temperature": temperature,
        "label_noise": label_noise,
        "class_prior": class_prior,
    }
    return x, y.astype(np.int64), meta


# =========================================================
# Prior 2: SCM
# =========================================================

def _sample_dag_parents(
    n_features: int,
    max_parents: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Simple DAG by topological order 0..d-1:
    node j can only choose parents from {0, ..., j-1}
    """
    parents: List[List[int]] = []
    for j in range(n_features):
        if j == 0:
            parents.append([])
            continue
        k = int(rng.integers(0, min(max_parents, j) + 1))
        if k == 0:
            parents.append([])
        else:
            pa = rng.choice(j, size=k, replace=False).tolist()
            parents.append(pa)
    return parents


def _apply_random_nonlinearity(v: np.ndarray, kind: str) -> np.ndarray:
    if kind == "affine":
        return v
    if kind == "tanh":
        return np.tanh(v)
    if kind == "sin":
        return np.sin(v)
    if kind == "square":
        return np.sign(v) * (np.abs(v) ** 2)
    raise ValueError(f"Unknown nonlinearity: {kind}")


def _generate_scm_features(
    n_samples: int,
    n_features: int,
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict]:
    parents = _sample_dag_parents(n_features, cfg.scm_max_parents, rng)

    x = np.zeros((n_samples, n_features), dtype=np.float64)

    # root nodes from mixed marginal distributions
    for j in range(n_features):
        noise_scale = float(rng.uniform(cfg.scm_noise_scale_min, cfg.scm_noise_scale_max))
        if len(parents[j]) == 0:
            root_mode = rng.choice(["gauss", "tanhgauss"])
            if root_mode == "gauss":
                x[:, j] = rng.normal(0, 1, size=n_samples)
            else:
                z = rng.normal(0, 1.0, size=n_samples)
                x[:, j] = np.tanh(z) + 0.1 * z
            x[:, j] += rng.normal(0, noise_scale, size=n_samples)
            continue

        pa = parents[j]
        xp = x[:, pa]

        # weighted sum
        w = rng.normal(0, 1, size=len(pa))
        base = xp @ w

        # optional pairwise multiplicative interaction among parents
        if len(pa) >= 2 and rng.random() < 0.2:
            i1, i2 = rng.choice(len(pa), size=2, replace=False)
            base = base + rng.uniform(0.15, 0.55) * xp[:, i1] * xp[:, i2]

        # random nonlinearity
        nonlin = rng.choice(["affine", "tanh"])
        out = _apply_random_nonlinearity(base, nonlin)

        # occasional skip/shortcut from one parent
        if rng.random() < 0.25:
            idx = int(rng.integers(0, len(pa)))
            out = out + rng.uniform(0.15, 0.55) * xp[:, idx]

        x[:, j] = out + rng.normal(0, noise_scale, size=n_samples)

    # standardize
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    meta = {"parents": parents}
    return x.astype(np.float32), meta


def generate_scm_task(
    n_samples: int,
    n_features: int,
    n_classes: int,
    temperature: float,
    label_noise: float,
    class_prior: np.ndarray,
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x, scm_meta = _generate_scm_features(n_samples, n_features, cfg, rng)

    # generate class logits from SCM features
    h = x.astype(np.float64)

    logits = np.zeros((n_samples, n_classes), dtype=np.float64)
    logit_scale = _sample_logit_scale(cfg, rng)
    for c in range(n_classes):
        # sparse informative subset for each class
        k = int(rng.integers(max(2, min(3, n_features)), min(n_features, 6) + 1))
        idx = rng.choice(n_features, size=k, replace=False)
        w = rng.normal(0, 1, size=k)
        z = h[:, idx] @ w

        # random nonlinear transform
        nonlin = rng.choice(["affine", "tanh"])
        z = _apply_random_nonlinearity(z, nonlin)

        # occasional multiplicative interaction
        if k >= 2 and rng.random() < 0.25:
            i1, i2 = rng.choice(k, size=2, replace=False)
            z = z + rng.uniform(0.15, 0.45) * h[:, idx[i1]] * h[:, idx[i2]]

        logits[:, c] = logit_scale * z + rng.normal(0, 0.08, size=n_samples)

    probs = _softmax(logits, temperature=temperature)
    y = _sample_labels_from_probs(
        probs,
        rng,
        class_prior=class_prior,
        label_noise=label_noise,
        deterministic_label_prob=cfg.deterministic_label_prob,
    )

    meta = {
        "prior_type": "scm",
        "temperature": temperature,
        "label_noise": label_noise,
        "class_prior": class_prior,
        **scm_meta,
    }
    return x, y.astype(np.int64), meta


# =========================================================
# Prior 3: Smooth MLP
# =========================================================

def _activation(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(x)
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))
    raise ValueError(f"Unknown activation: {kind}")


def generate_smooth_mlp_task(
    n_samples: int,
    n_features: int,
    n_classes: int,
    temperature: float,
    label_noise: float,
    class_prior: np.ndarray,
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x = sample_features(n_samples, n_features, cfg, rng).astype(np.float64)

    h1 = int(rng.integers(cfg.mlp_hidden_min, cfg.mlp_hidden_max + 1))
    h2 = int(rng.integers(cfg.mlp_hidden_min, cfg.mlp_hidden_max + 1))
    act = "tanh"

    w1 = rng.normal(0, 1 / np.sqrt(n_features), size=(n_features, h1))
    b1 = rng.normal(0, 0.3, size=h1)
    w2 = rng.normal(0, 1 / np.sqrt(h1), size=(h1, h2))
    b2 = rng.normal(0, 0.3, size=h2)
    w3 = rng.normal(0, 1 / np.sqrt(h2), size=(h2, n_classes))
    b3 = rng.normal(0, 0.3, size=n_classes)

    h = _activation(x @ w1 + b1, act)
    h = _activation(h @ w2 + b2, act)
    logits = (h @ w3 + b3) * _sample_logit_scale(cfg, rng)

    # mild smooth stochasticity
    logits += rng.normal(0, 0.05, size=logits.shape)

    probs = _softmax(logits, temperature=temperature)
    y = _sample_labels_from_probs(
        probs,
        rng,
        class_prior=class_prior,
        label_noise=label_noise,
        deterministic_label_prob=cfg.deterministic_label_prob,
    )

    meta = {
        "prior_type": "smooth_mlp",
        "temperature": temperature,
        "label_noise": label_noise,
        "class_prior": class_prior,
        "activation": act,
    }
    return x.astype(np.float32), y.astype(np.int64), meta


# =========================================================
# Prior 4: Sparse Linear / Low-order Interaction
# =========================================================

def generate_sparse_linear_task(
    n_samples: int,
    n_features: int,
    n_classes: int,
    temperature: float,
    label_noise: float,
    class_prior: np.ndarray,
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x = sample_features(n_samples, n_features, cfg, rng).astype(np.float64)

    informative = int(
        rng.integers(
            min(cfg.informative_min, n_features),
            min(cfg.informative_max, n_features) + 1,
        )
    )
    feat_idx = rng.choice(n_features, size=informative, replace=False)

    logits = np.zeros((n_samples, n_classes), dtype=np.float64)
    logit_scale = _sample_logit_scale(cfg, rng)

    for c in range(n_classes):
        w = rng.normal(0, 1, size=informative)
        z = x[:, feat_idx] @ w + rng.normal(0, 0.15)

        # optional small number of pairwise interactions
        if informative >= 2:
            for i in range(informative):
                for j in range(i + 1, informative):
                    if rng.random() < cfg.pairwise_interaction_prob:
                        coef = rng.uniform(-0.35, 0.35)
                        z += coef * x[:, feat_idx[i]] * x[:, feat_idx[j]]

        logits[:, c] = logit_scale * z

    probs = _softmax(logits, temperature=temperature)
    y = _sample_labels_from_probs(
        probs,
        rng,
        class_prior=class_prior,
        label_noise=label_noise,
        deterministic_label_prob=cfg.deterministic_label_prob,
    )

    meta = {
        "prior_type": "sparse_linear",
        "temperature": temperature,
        "label_noise": label_noise,
        "class_prior": class_prior,
        "informative_features": feat_idx,
    }
    return x.astype(np.float32), y.astype(np.int64), meta


# =========================================================
# Prior 5: Nonlinear Link (TabMGP-style)
# =========================================================

def _standard_normal_pdf(u: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-8)
    z = (u - float(mean)) / std
    return np.exp(-0.5 * (z ** 2)) / (std * math.sqrt(2.0 * math.pi))


def _standard_normal_cdf(u: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-8)
    z = (u - float(mean)) / (std * math.sqrt(2.0))
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(z))


def _apply_nonlinear_link(
    logits: np.ndarray,
    link_kind: str,
    gmm_a: Optional[float] = None,
) -> np.ndarray:
    if link_kind == "logistic":
        return 1.0 / (1.0 + np.exp(-logits))
    if link_kind == "gmm":
        if gmm_a is None:
            raise ValueError("gmm_a must be provided for gmm nonlinear link")
        return 0.7 * _standard_normal_cdf(logits, mean=float(gmm_a), std=1.0) + 0.3 * _standard_normal_cdf(
            logits,
            mean=2.0,
            std=1.0,
        )
    raise ValueError(f"Unknown nonlinear link kind: {link_kind}")


def _sample_binary_labels_from_probs(
    probs: np.ndarray,
    rng: np.random.Generator,
    label_noise: float = 0.0,
    deterministic_label_prob: float = 0.0,
) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    n = probs.shape[0]

    y = np.empty(n, dtype=np.int64)
    deterministic_mask = rng.random(n) < deterministic_label_prob
    y[deterministic_mask] = (probs[deterministic_mask] >= 0.5).astype(np.int64)

    sample_mask = ~deterministic_mask
    if np.any(sample_mask):
        y[sample_mask] = rng.binomial(1, probs[sample_mask]).astype(np.int64)

    if label_noise > 0:
        flip_mask = rng.random(n) < label_noise
        if np.any(flip_mask):
            y[flip_mask] = 1 - y[flip_mask]

    return y


def generate_nonlinear_link_task(
    n_samples: int,
    n_features: int,
    n_classes: int,
    temperature: float,
    label_noise: float,
    class_prior: np.ndarray,
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # TabMGP-style binary setup:
    #   x ~ Unif([-1, 1]^d)
    #   beta ~ Unif([-2, 3]^d), fixed if nonlinear_fixed_beta_seed is set
    #   p = L(x^T beta), L in {logistic, GMM-CDF(a)}
    #   y ~ Bernoulli(p)
    if n_classes != 2:
        raise ValueError(
            f"nonlinear_link prior is binary TabMGP-style data; expected n_classes=2, got {n_classes}"
        )

    x = rng.uniform(-1.0, 1.0, size=(n_samples, n_features)).astype(np.float64)
    if cfg.nonlinear_fixed_beta_seed is not None:
        beta_rng = np.random.default_rng(cfg.nonlinear_fixed_beta_seed)
    else:
        beta_rng = rng
    beta = beta_rng.uniform(
        cfg.nonlinear_beta_min,
        cfg.nonlinear_beta_max,
        size=(n_features,),
    ).astype(np.float64)
    raw_logits = x @ beta

    if cfg.nonlinear_link_kind == "mixed":
        link_kind = str(rng.choice(["logistic", "gmm"]))
    else:
        link_kind = str(cfg.nonlinear_link_kind)
    gmm_a: Optional[float] = None
    if link_kind == "gmm":
        if cfg.nonlinear_fixed_gmm_a is not None:
            gmm_a = float(cfg.nonlinear_fixed_gmm_a)
        else:
            gmm_a = float(rng.choice(cfg.nonlinear_gmm_a_choices))

    probs_pos = _apply_nonlinear_link(
        raw_logits,
        link_kind=link_kind,
        gmm_a=gmm_a,
    )
    y = _sample_binary_labels_from_probs(
        probs_pos,
        rng,
        label_noise=label_noise,
        deterministic_label_prob=cfg.deterministic_label_prob,
    )

    meta = {
        "prior_type": "nonlinear_link",
        "temperature": temperature,
        "label_noise": label_noise,
        "class_prior": class_prior,
        "link_kind": link_kind,
        "gmm_a": gmm_a,
        "beta": beta,
    }
    return x.astype(np.float32), y.astype(np.int64), meta


# =========================================================
# Mixture task generator
# =========================================================

def _choose_prior(cfg: MixtureConfig, rng: np.random.Generator) -> str:
    priors = ["gbdt", "scm", "smooth_mlp", "sparse_linear", "nonlinear_link"]
    weights = np.array(
        [
            cfg.p_gbdt,
            cfg.p_scm,
            cfg.p_smooth_mlp,
            cfg.p_sparse_linear,
            cfg.p_nonlinear_link,
        ],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    return str(rng.choice(priors, p=weights))


def generate_mixture_task(
    cfg: MixtureConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    composite = _generate_composite_mode_task(cfg, rng)
    if composite is not None:
        return composite

    n_features = int(rng.integers(cfg.n_features_min, cfg.n_features_max + 1))
    n_samples = int(rng.integers(cfg.n_samples_min, cfg.n_samples_max + 1))
    n_classes = int(rng.integers(cfg.n_classes_min, cfg.n_classes_max + 1))

    temperature = float(rng.uniform(cfg.temperature_min, cfg.temperature_max))
    label_noise = float(rng.uniform(cfg.label_noise_min, cfg.label_noise_max))
    if cfg.class_prior_mode == "uniform":
        class_prior = np.full(n_classes, 1.0 / float(n_classes), dtype=np.float64)
    else:
        class_prior = _random_dirichlet_prior(n_classes, rng, cfg.dirichlet_alpha_choices)

    prior = _choose_prior(cfg, rng)

    if prior == "gbdt":
        x, y, meta = generate_gbdt_task(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            temperature=temperature,
            label_noise=label_noise,
            class_prior=class_prior,
            cfg=cfg,
            rng=rng,
        )
    elif prior == "scm":
        x, y, meta = generate_scm_task(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            temperature=temperature,
            label_noise=label_noise,
            class_prior=class_prior,
            cfg=cfg,
            rng=rng,
        )
    elif prior == "smooth_mlp":
        x, y, meta = generate_smooth_mlp_task(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            temperature=temperature,
            label_noise=label_noise,
            class_prior=class_prior,
            cfg=cfg,
            rng=rng,
        )
    elif prior == "sparse_linear":
        x, y, meta = generate_sparse_linear_task(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            temperature=temperature,
            label_noise=label_noise,
            class_prior=class_prior,
            cfg=cfg,
            rng=rng,
        )
    elif prior == "nonlinear_link":
        x, y, meta = generate_nonlinear_link_task(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            temperature=temperature,
            label_noise=label_noise,
            class_prior=class_prior,
            cfg=cfg,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown prior: {prior}")

    meta.update(
        {
            "mode_name": cfg.mode_name,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
        }
    )
    return x, y, meta


def generate_mixture_dataset(
    n_tasks: int,
    cfg: Optional[MixtureConfig] = None,
    seed: Optional[int] = 0,
    return_metadata: bool = True,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Optional[List[Dict]]]:
    """
    Returns
    -------
    tasks : list of (X, y)
        X: float32 [N, D]
        y: int64   [N]
    metadata : list of dict
        optional task metadata
    """
    if cfg is None:
        cfg = MixtureConfig()

    rng = _rng(seed)
    tasks: List[Tuple[np.ndarray, np.ndarray]] = []
    metas: List[Dict] = []

    for _ in range(n_tasks):
        x, y, meta = generate_mixture_task(cfg, rng)
        tasks.append((x, y))
        metas.append(meta)

    return tasks, metas if return_metadata else None


def generate_mixture_tensors(
    n_tasks: int,
    cfg: Optional[MixtureConfig] = None,
    seed: Optional[int] = 0,
    return_metadata: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """
    Returns fixed-shape synthetic tensors compatible with the repo pipeline.

    X : float32 [T, N, D]
    y : int64   [T, N]
    """
    tasks, metas = generate_mixture_dataset(
        n_tasks=n_tasks,
        cfg=cfg,
        seed=seed,
        return_metadata=True,
    )

    x_shapes = {x.shape for x, _ in tasks}
    y_shapes = {y.shape for _, y in tasks}
    if len(x_shapes) != 1 or len(y_shapes) != 1:
        raise ValueError(
            "Synthetic tasks are not fixed-shape. Set equal min/max values or use "
            "MixtureConfig(..., allow_variable_shapes=False)."
        )

    x_tasks = np.stack([x for x, _ in tasks], axis=0).astype(np.float32)
    y_tasks = np.stack([y for _, y in tasks], axis=0).astype(np.int64)
    return x_tasks, y_tasks, metas if return_metadata else None


# =========================================================
# Train/SC split helper
# =========================================================

def split_task_for_sc(
    x: np.ndarray,
    y: np.ndarray,
    context_size: int = 100,
    query_pool_size: int = 20,
    rng: Optional[np.random.Generator] = None,
    stratified_context: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split task into context / query_pool / rollout_pool.

    Parameters
    ----------
    stratified_context : bool
        If True, tries to distribute context examples across classes.
        Useful when severe class imbalance hurts stability.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y)
    idx = np.arange(n)

    if not stratified_context:
        rng.shuffle(idx)
    else:
        # simple approximate stratified permutation
        classes = np.unique(y)
        per_class_indices = [np.where(y == c)[0] for c in classes]
        for arr in per_class_indices:
            rng.shuffle(arr)

        mixed = []
        ptrs = [0 for _ in per_class_indices]
        while len(mixed) < n:
            progressed = False
            for ci, arr in enumerate(per_class_indices):
                if ptrs[ci] < len(arr):
                    mixed.append(arr[ptrs[ci]])
                    ptrs[ci] += 1
                    progressed = True
            if not progressed:
                break
        idx = np.array(mixed, dtype=np.int64)

    c_end = min(context_size, n)
    q_end = min(c_end + query_pool_size, n)

    context_idx = idx[:c_end]
    query_idx = idx[c_end:q_end]
    rollout_idx = idx[q_end:]

    return {
        "context": (x[context_idx], y[context_idx]),
        "query_pool": (x[query_idx], y[query_idx]),
        "rollout_pool": (x[rollout_idx], y[rollout_idx]),
    }


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    cfg = make_mixture_config("mixed_full")

    x_tasks, y_tasks, metas = generate_mixture_tensors(
        n_tasks=100,
        cfg=cfg,
        seed=42,
        return_metadata=True,
    )

    print(f"Generated tensors: X={x_tasks.shape}, y={y_tasks.shape}")

    for i, (x, y, meta) in enumerate(zip(x_tasks, y_tasks, metas)):
        print(f"[Task {i}] prior={meta['prior_type']}, X={x.shape}, classes={len(np.unique(y))}")
        split = split_task_for_sc(
            x, y,
            context_size=100,
            query_pool_size=20,
            stratified_context=False,
        )
        xc, yc = split["context"]
        xq, yq = split["query_pool"]
        xr, yr = split["rollout_pool"]

        print("  context:", xc.shape, yc.shape)
        print("  query  :", xq.shape, yq.shape)
        print("  rollout:", xr.shape, yr.shape)
