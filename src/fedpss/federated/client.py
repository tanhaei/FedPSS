from __future__ import annotations

import copy
from typing import Dict

import torch
from torch.utils.data import DataLoader

from fedpss.data import PatientDataset, collate_batch
from fedpss.privacy import clip_and_noise_update, state_delta
from fedpss.training import train_one_epoch


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        dataset: PatientDataset,
        batch_size: int = 16,
        lr: float = 1e-3,
        local_epochs: int = 1,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    def fit(self, global_model, privacy_cfg: dict | None = None) -> Dict[str, torch.Tensor]:
        local_model = copy.deepcopy(global_model).to(self.device)
        old_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            train_one_epoch(local_model, loader, optimizer, device=self.device)
        new_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
        delta = state_delta(new_state, old_state)
        if privacy_cfg and privacy_cfg.get("enabled", False):
            delta = clip_and_noise_update(
                delta,
                clip_norm=float(privacy_cfg.get("clip_norm", 1.0)),
                noise_multiplier=float(privacy_cfg.get("noise_multiplier", 0.0)),
            )
        return delta
