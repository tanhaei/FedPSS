from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return h[-1]


class CodeEncoder(nn.Module):
    def __init__(self, code_vocab_size: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(code_vocab_size, hidden_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, modalities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(modalities)
        values = self.value(modalities)
        scores = torch.einsum("bmh,h->bm", keys, self.query) / (modalities.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=1)
        fused = torch.einsum("bm,bmh->bh", weights, values)
        return fused, weights


class PatientSimilarityModel(nn.Module):
    def __init__(
        self,
        static_dim: int,
        temporal_dim: int,
        code_vocab_size: int,
        note_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.static_encoder = StaticEncoder(static_dim, hidden_dim)
        self.temporal_encoder = TemporalEncoder(temporal_dim, hidden_dim)
        self.code_encoder = CodeEncoder(code_vocab_size, hidden_dim)
        self.note_encoder = nn.Sequential(nn.Linear(note_dim, hidden_dim), nn.ReLU())
        self.fusion = CrossModalAttention(hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        z_static = self.static_encoder(batch["static"])
        z_temporal = self.temporal_encoder(batch["temporal"], batch["lengths"])
        z_codes = self.code_encoder(batch["codes"])
        z_notes = self.note_encoder(batch["note_embedding"])
        modalities = torch.stack([z_static, z_temporal, z_codes, z_notes], dim=1)
        fused, attention = self.fusion(modalities)
        embedding = F.normalize(self.projection(fused), p=2, dim=-1)
        return embedding, attention


def build_model(cfg: dict) -> PatientSimilarityModel:
    return PatientSimilarityModel(
        static_dim=int(cfg["static_dim"]),
        temporal_dim=int(cfg["temporal_dim"]),
        code_vocab_size=int(cfg["code_vocab_size"]),
        note_dim=int(cfg["note_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        embedding_dim=int(cfg.get("embedding_dim", 64)),
    )
