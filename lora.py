"""
TabPFN 모델에 LoRA adapter를 주입/병합하는 유틸리티.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA-wrapped Linear layer.

    W' = W + (α/r) * B @ A
    - Original weight W is frozen
    - Only A, B are trainable

    Args:
        original: nn.Linear to wrap
        r:        LoRA rank
        alpha:    LoRA scaling factor
    """

    def __init__(self, original: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original.in_features
        out_features = original.out_features
        device = original.weight.device

        # A: (in_features, r) — Kaiming uniform, same device as original
        self.lora_A = nn.Parameter(torch.empty(in_features, r, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # B: (r, out_features) — zero init → initial LoRA contribution = 0
        self.lora_B = nn.Parameter(torch.zeros(r, out_features, device=device))

        # Freeze original weight
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original: x @ W^T + bias
        out = self.original(x)
        # LoRA: (α/r) * x @ A @ B^T  →  shape: (..., out_features)
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) * self.scaling
        return out + lora_out

    def merge_and_unload(self) -> nn.Linear:
        """
        LoRA weights를 original에 merge하고 plain nn.Linear 반환.
        W_merged = W + (α/r) * A @ B  →  shape (out, in)
        """
        with torch.no_grad():
            # A @ B: (in, r) @ (r, out) = (in, out), transpose → (out, in)
            delta = (self.lora_A @ self.lora_B).T * self.scaling
            self.original.weight.add_(delta)
        return self.original


@dataclass
class LoRAConfig:
    """LoRA 주입 설정."""
    r: int = 8                    # LoRA rank
    alpha: float = 16.0           # LoRA scaling
    target_layers: Optional[tuple] = None  # transformer block indices (None = auto: last 4)
    include_decoder: bool = True  # decoder head에도 LoRA 적용


def auto_target_layers(model: nn.Module, n_last: int = 4) -> tuple[int, ...]:
    """
    모델의 transformer block 수를 감지하고 마지막 n_last 블록 index 반환.

    Classifier: 24 layers → (20, 21, 22, 23)
    Regressor:  18 layers → (14, 15, 16, 17)
    """
    layers = model.transformer_encoder.layers
    n_total = len(layers)
    start = max(0, n_total - n_last)
    return tuple(range(start, n_total))


def get_tabpfn_model(pred_rule) -> nn.Module:
    """
    PredictiveRule에서 underlying PyTorch model 추출.

    TabPFNClassifier/Regressor는 fit() 후 model_ 속성에 모델이 있음.
    """
    model = pred_rule.model
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")
    return model


def freeze_backbone(model: nn.Module) -> None:
    """전체 모델 파라미터를 freeze (requires_grad=False)."""
    for param in model.parameters():
        param.requires_grad_(False)


def inject_lora(
    model: nn.Module,
    config: LoRAConfig = LoRAConfig(),
) -> list[LoRALinear]:
    """
    TabPFN 모델에 LoRA adapter 주입.

    1. 전체 backbone freeze
    2. target layers의 MLP(linear1, linear2)에 LoRA 래핑
    3. decoder head의 Linear에 LoRA 래핑

    Args:
        model:  TabPFN PyTorch model (PerFeatureTransformer)
        config: LoRAConfig

    Returns:
        list of LoRALinear modules (for parameter collection)
    """
    # Step 1: Freeze all
    freeze_backbone(model)

    lora_modules = []

    # Resolve target layers
    target_layers = config.target_layers
    if target_layers is None:
        target_layers = auto_target_layers(model)

    # Step 2: LoRA on target transformer blocks' MLP
    layer_stack = model.transformer_encoder.layers
    for idx in target_layers:
        if idx >= len(layer_stack):
            raise IndexError(
                f"Layer index {idx} out of range (model has {len(layer_stack)} layers)"
            )
        block = layer_stack[idx]
        mlp = block.mlp

        # linear1: (192 → 384)
        lora_l1 = LoRALinear(mlp.linear1, r=config.r, alpha=config.alpha)
        mlp.linear1 = lora_l1
        lora_modules.append(lora_l1)

        # linear2: (384 → 192)
        lora_l2 = LoRALinear(mlp.linear2, r=config.r, alpha=config.alpha)
        mlp.linear2 = lora_l2
        lora_modules.append(lora_l2)

    # Step 3: LoRA on decoder head
    if config.include_decoder:
        decoder = model.decoder_dict.standard
        assert isinstance(decoder, nn.Sequential), \
            f"Expected nn.Sequential for decoder, got {type(decoder)}"
        assert isinstance(decoder[0], nn.Linear), \
            f"Expected nn.Linear at decoder[0], got {type(decoder[0])}"
        assert isinstance(decoder[2], nn.Linear), \
            f"Expected nn.Linear at decoder[2], got {type(decoder[2])}"
        # decoder[0]: Linear(192 → 384, bias=True)
        lora_d0 = LoRALinear(decoder[0], r=config.r, alpha=config.alpha)
        decoder[0] = lora_d0
        lora_modules.append(lora_d0)

        # decoder[2]: Linear(384 → output_dim, bias=True)
        lora_d2 = LoRALinear(decoder[2], r=config.r, alpha=config.alpha)
        decoder[2] = lora_d2
        lora_modules.append(lora_d2)

    return lora_modules


def get_lora_params(lora_modules: list[LoRALinear]) -> list[nn.Parameter]:
    """
    LoRA 모듈들에서 trainable parameters 수집.
    Optimizer에 전달할 파라미터 리스트 반환.
    """
    params = []
    for m in lora_modules:
        params.append(m.lora_A)
        params.append(m.lora_B)
    return params


def merge_lora(model: nn.Module) -> int:
    """
    모델 트리에서 LoRALinear를 자동 탐색해 merge + nn.Linear로 치환.

    Returns:
        merged_count: merge된 LoRALinear 개수
    """
    merged_count = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                merged_linear = child.merge_and_unload()
                setattr(parent, child_name, merged_linear)
                merged_count += 1

    # 안전장치: 남아있는 LoRALinear가 없어야 함
    leftovers = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    if len(leftovers) > 0:
        raise RuntimeError(
            f"merge_lora failed: LoRALinear still present in model: {leftovers[:10]}"
            + (" ..." if len(leftovers) > 10 else "")
        )
    return merged_count


def print_lora_summary(model: nn.Module, lora_modules: list[LoRALinear]) -> dict:
    """
    LoRA 주입 후 파라미터 통계 출력.

    Returns:
        dict with total_params, trainable_params, lora_params, ratio
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_p = sum(m.lora_A.numel() + m.lora_B.numel() for m in lora_modules)

    summary = {
        "total_params": total,
        "trainable_params": trainable,
        "lora_params": lora_p,
        "n_lora_modules": len(lora_modules),
        "ratio_pct": 100.0 * trainable / total if total > 0 else 0,
    }

    print(f"── LoRA Summary ──")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  LoRA params:      {lora_p:,}")
    print(f"  LoRA modules:     {len(lora_modules)}")
    print(f"  Trainable ratio:  {summary['ratio_pct']:.2f}%")
    print(f"  LoRA targets:")
    for m in lora_modules:
        name = "unknown"
        for n, mod in model.named_modules():
            if mod is m:
                name = n
                break
        in_f = m.original.in_features
        out_f = m.original.out_features
        print(f"    {name}: ({in_f}→{out_f}), r={m.r}, α={m.alpha}")

    return summary
