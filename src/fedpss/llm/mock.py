from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class MockSemanticMapper:
    """Deterministic offline semantic mapper used for tests.

    It mimics descriptor-level semantic mapping without downloading any model
    weights. It is not a clinical language model.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, descriptors: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(descriptors), self.dim), dtype=np.float32)
        for i, text in enumerate(descriptors):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            lowered = text.lower()
            for keyword, offset in [
                ("glaucoma", 0),
                ("iop", 0),
                ("intraocular", 0),
                ("cataract", 1),
                ("diabetes", 2),
                ("hypertension", 3),
                ("retina", 4),
                ("oct", 4),
            ]:
                if keyword in lowered:
                    vec[offset::5] += 0.75
            norm = np.linalg.norm(vec) + 1e-8
            out[i] = vec / norm
        return out


# Backward-compatible alias for older tests/imports.
MockNoteEncoder = MockSemanticMapper
