from __future__ import annotations

from typing import Sequence

import numpy as np


class Gemma4NoteEncoder:
    """Frozen Gemma 4 text encoder for clinical notes.

    This class extracts mean-pooled hidden-state embeddings from a Gemma 4
    checkpoint. It keeps Gemma frozen and returns note-level representations for
    the downstream federated similarity model.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E2B-it",
        max_length: int = 512,
        device_map: str = "auto",
        dtype: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Gemma4NoteEncoder requires torch and transformers. Install with `pip install -r requirements.txt`."
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

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - too large for CI
        torch = self.torch
        inputs = self.tokenizer(
            list(texts),
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


def build_note_encoder(cfg: dict):
    backend = cfg.get("backend", "gemma4")
    if backend == "mock":
        from .mock import MockNoteEncoder

        return MockNoteEncoder(dim=int(cfg.get("note_dim", cfg.get("dim", 64))))
    if backend == "gemma4":
        return Gemma4NoteEncoder(
            model_id=cfg.get("model_id", "google/gemma-4-E2B-it"),
            max_length=int(cfg.get("max_length", 512)),
            device_map=cfg.get("device_map", "auto"),
            dtype=cfg.get("dtype", "auto"),
        )
    raise ValueError(f"Unknown LLM backend: {backend}")
