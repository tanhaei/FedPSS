from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class MockNoteEncoder:
    """Deterministic offline encoder used for tests.

    It is not a clinical language model. It preserves the same interface as the
    Gemma 4 encoder so the whole pipeline can be tested without downloading
    model weights.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            for keyword, offset in [
                ("glaucoma", 0),
                ("cataract", 1),
                ("diabetes", 2),
                ("hypertension", 3),
                ("retina", 4),
            ]:
                if keyword in text.lower():
                    vec[offset::5] += 0.75
            norm = np.linalg.norm(vec) + 1e-8
            out[i] = vec / norm
        return out
