from __future__ import annotations

from typing import Sequence

import numpy as np


class GemmaSemanticMapper:
    """Frozen Gemma-family encoder for structured clinical descriptors.

    This class extracts mean-pooled hidden-state embeddings from a Gemma-family
    checkpoint. It keeps the backbone frozen and returns descriptor-level
    representations for the downstream federated retrieval model. The mapper is
    not a diagnostic generator and is not intended for raw free-text note
    interpretation.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-2-2b-it",
        max_length: int = 128,
        device_map: str = "auto",
        dtype: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "GemmaSemanticMapper requires torch and transformers. Install with `pip install -r requirements.txt`."
            ) from exc

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": device_map}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModel.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def encode(self, descriptors: Sequence[str]) -> np.ndarray:  # pragma: no cover - too large for CI
        torch = self.torch
        inputs = self.tokenizer(
            list(descriptors),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        return pooled.cpu().numpy().astype(np.float32)


# Backward-compatible alias for the original repository name.
Gemma4NoteEncoder = GemmaSemanticMapper


def build_semantic_mapper(cfg: dict):
    backend = cfg.get("backend", "gemma_lora")
    if backend in {"mock", "mock_semantic"}:
        from .mock import MockSemanticMapper

        return MockSemanticMapper(dim=int(cfg.get("semantic_dim", cfg.get("note_dim", cfg.get("dim", 64)))))
    if backend in {"gemma_lora", "gemma", "gemma4"}:
        return GemmaSemanticMapper(
            model_id=cfg.get("model_id", "google/gemma-2-2b-it"),
            max_length=int(cfg.get("max_length", 128)),
            device_map=cfg.get("device_map", "auto"),
            dtype=cfg.get("dtype", "auto"),
        )
    raise ValueError(f"Unknown semantic mapper backend: {backend}")


# Backward-compatible function name.
def build_note_encoder(cfg: dict):
    return build_semantic_mapper(cfg)
