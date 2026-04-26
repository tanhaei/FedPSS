from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch

from fedpss.privacy import add_delta


class FederatedServer:
    def __init__(self, model):
        self.model = model

    def aggregate(self, updates: List[Tuple[int, Dict[str, torch.Tensor]]]) -> None:
        if not updates:
            return
        total = float(sum(n for n, _ in updates))
        agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in updates[0][1].items()}
        for n, delta in updates:
            weight = n / total
            for k, v in delta.items():
                agg[k] += v.float() * weight
        old_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        new_state = add_delta(old_state, agg)
        self.model.load_state_dict(new_state, strict=True)
