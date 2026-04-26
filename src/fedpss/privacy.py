from __future__ import annotations

from typing import Dict

import torch


def state_delta(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: new_state[k].detach().cpu() - old_state[k].detach().cpu() for k in old_state}


def add_delta(old_state: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: old_state[k].detach().cpu() + delta[k].detach().cpu() for k in old_state}


def clip_and_noise_update(delta: Dict[str, torch.Tensor], clip_norm: float = 1.0, noise_multiplier: float = 0.0) -> Dict[str, torch.Tensor]:
    total_sq = sum(float(torch.sum(v.float() ** 2)) for v in delta.values())
    norm = total_sq ** 0.5
    scale = min(1.0, clip_norm / (norm + 1e-12))
    out = {}
    for k, v in delta.items():
        clipped = v.float() * scale
        if noise_multiplier > 0:
            clipped = clipped + torch.randn_like(clipped) * (noise_multiplier * clip_norm)
        out[k] = clipped
    return out
