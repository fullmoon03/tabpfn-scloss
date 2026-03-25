"""Loss helpers for SC."""

import torch

import numpy as np


EPS = 1e-8


def soft_cross_entropy(
    target: torch.Tensor,
    pred: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Soft target cross-entropy: CE(t, p) = -Σ_c t_c * log(p_c)

    Args:
        target: (C,) or (1, C) soft target distribution (detached)
        pred:   (C,) or (1, C) predicted distribution (gradient flows here)
        eps:    numerical stability

    Returns:
        scalar loss
    """
    target = target.view(-1)
    pred = pred.view(-1)
    pred = pred.clamp(min=eps)
    return -(target * pred.log()).sum()


def _normalize_prob(p: torch.Tensor) -> torch.Tensor:
    """Normalize a probability vector safely."""
    p = p.view(-1)
    return p / p.sum().clamp_min(EPS)


def sc_loss(
    p_by_k: dict[int, torch.Tensor],
    *,
    sampled_pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """
    Pairwise SC loss:
      L_sc = (1 / |P|) * Σ_(k1,k2 in P) H(stopgrad(p^(k2)), p^(k1))

    Args:
        p_by_k:         mapping k -> belief tensor (C,)
        sampled_pairs:  sampled (k1, k2) pairs. Teacher is p^(k2), student is p^(k1).
    """
    if len(sampled_pairs) == 0:
        raise ValueError("SC loss requires non-empty sampled_pairs.")

    pair_terms: list[torch.Tensor] = []
    for k1, k2 in sampled_pairs:
        if int(k1) not in p_by_k or int(k2) not in p_by_k:
            raise ValueError(f"SC loss missing sampled horizons: ({k1},{k2}) not in p_by_k.")
        p_k1 = _normalize_prob(p_by_k[int(k1)])
        p_k2 = _normalize_prob(p_by_k[int(k2)]).detach()
        pair_terms.append(soft_cross_entropy(p_k2, p_k1))

    return torch.stack(pair_terms).mean()


def beliefs_to_torch(
    p_early_list: list[np.ndarray],
    p_late_mean: np.ndarray,
    requires_grad: bool = True,
    device: torch.device = torch.device("cpu"),
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    numpy belief arrays를 torch tensor로 변환.

    Legacy helper: numpy belief arrays를 torch tensor로 변환한다.

    Args:
        p_early_list: list of (C,) numpy arrays — early beliefs
        p_late_mean:  (C,) numpy array — MC marginal target
        requires_grad: early beliefs에 gradient 필요 여부
        device: torch device

    Returns:
        (p_early_torch_list, p_late_mean_torch)
    """
    p_early_torch = [
        torch.from_numpy(p.copy()).float().to(device).requires_grad_(requires_grad)
        for p in p_early_list
    ]
    p_late_torch = torch.from_numpy(p_late_mean.copy()).float().to(device)
    return p_early_torch, p_late_torch
