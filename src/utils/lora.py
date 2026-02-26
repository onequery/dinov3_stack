from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA wrapper for linear modules.

    Notes:
    - Keeps the original module object intact as `base_layer` so custom behavior
      (e.g. LinearKMaskedBias) is preserved.
    - LoRA update is always additive: y = base(x) + scale * B(A(x)).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: int | None = None,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if dropout < 0.0:
            raise ValueError(f"LoRA dropout must be >= 0, got {dropout}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = int(alpha if alpha is not None else rank)
        self.scaling = self.alpha / float(self.rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.rank))
        self.reset_parameters()
        self.set_base_trainable(not freeze_base)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_base_trainable(self, trainable: bool):
        for param in self.base_layer.parameters():
            param.requires_grad = bool(trainable)

    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + (self.scaling * lora_out)


def _resolve_target_linears(
    backbone: nn.Module,
    target: str = "attn_qkv_proj",
    trainable_block_indices: Iterable[int] | None = None,
) -> List[Tuple[nn.Module, str, int, nn.Linear]]:
    if target != "attn_qkv_proj":
        raise ValueError(f"Unsupported LoRA target: {target}")

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("Backbone does not expose `blocks`; expected ViT-like model.")

    if trainable_block_indices is None:
        allowed = set(range(len(blocks)))
    else:
        allowed = set(trainable_block_indices)

    targets: List[Tuple[nn.Module, str, int, nn.Linear]] = []
    for block_idx, block in enumerate(blocks):
        if block_idx not in allowed:
            continue

        attn = getattr(block, "attn", None)
        if attn is None:
            continue

        for module_name in ("qkv", "proj"):
            linear = getattr(attn, module_name, None)
            if linear is None:
                continue
            if not isinstance(linear, nn.Linear):
                raise TypeError(
                    f"Expected nn.Linear-like module at blocks[{block_idx}].attn.{module_name}, "
                    f"got {type(linear)}"
                )
            targets.append((attn, module_name, block_idx, linear))
    return targets


def inject_lora_into_vit(
    backbone: nn.Module,
    target: str = "attn_qkv_proj",
    rank: int = 4,
    alpha: int | None = None,
    dropout: float = 0.0,
    trainable_block_indices: Iterable[int] | None = None,
    preserve_base_trainability: bool = True,
) -> Dict[str, object]:
    """
    Replace target attention linears with LoRALinear wrappers.
    """
    targets = _resolve_target_linears(
        backbone=backbone,
        target=target,
        trainable_block_indices=trainable_block_indices,
    )
    if not targets:
        raise ValueError("No target linear layers found for LoRA injection.")

    for parent, module_name, _, original in targets:
        freeze_base = True
        if preserve_base_trainability:
            freeze_base = not bool(getattr(original.weight, "requires_grad", False))

        wrapped = LoRALinear(
            base_layer=original,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            freeze_base=freeze_base,
        )
        setattr(parent, module_name, wrapped)

    num_lora_params = sum(
        module.lora_A.numel() + module.lora_B.numel()
        for module in backbone.modules()
        if isinstance(module, LoRALinear)
    )
    return {
        "target": target,
        "rank": rank,
        "alpha": int(alpha if alpha is not None else rank),
        "dropout": dropout,
        "num_wrapped_layers": len(targets),
        "num_lora_params": int(num_lora_params),
    }


def collect_lora_params(module: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for sub_module in module.modules():
        if isinstance(sub_module, LoRALinear):
            params.extend(sub_module.lora_parameters())
    return params


def count_lora_params(module: nn.Module) -> int:
    return int(sum(param.numel() for param in collect_lora_params(module)))

