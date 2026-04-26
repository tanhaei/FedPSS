from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


PHENOTYPES = ["glaucoma", "cataract", "diabetes", "hypertension", "retina"]
NOTE_TEMPLATES = {
    "glaucoma": "glaucoma follow-up, IOP trend, visual field change, latanoprost response",
    "cataract": "cataract evaluation, blurred vision, lens opacity, surgery planning",
    "diabetes": "diabetes control, HbA1c monitoring, retinal screening, metformin history",
    "hypertension": "blood pressure follow-up, headache, cardiovascular risk, medication adjustment",
    "retina": "retina evaluation, OCT finding, macular edema, visual acuity change",
}


@dataclass
class PatientRecord:
    patient_id: str
    hospital_id: str
    phenotype: str
    static: np.ndarray
    temporal: np.ndarray
    codes: List[int]
    note: str
    note_embedding: np.ndarray | None = None

    def to_json(self) -> Dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "hospital_id": self.hospital_id,
            "phenotype": self.phenotype,
            "static": self.static.astype(float).tolist(),
            "temporal": self.temporal.astype(float).tolist(),
            "codes": list(map(int, self.codes)),
            "note": self.note,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_synthetic_bioarc(
    n_patients: int = 240,
    n_hospitals: int = 5,
    static_dim: int = 6,
    temporal_dim: int = 5,
    code_vocab_size: int = 64,
    non_iid_strength: float = 0.7,
    seed: int = 42,
) -> List[PatientRecord]:
    """Generate a BioArc-like de-identified cohort for tests and examples.

    The generator creates phenotype-specific clinical patterns and hospital-specific
    skew. It is not a clinical simulator; it is a reproducibility scaffold.
    """
    rng = np.random.default_rng(seed)
    records: List[PatientRecord] = []
    base_centers = rng.normal(0, 1, size=(len(PHENOTYPES), static_dim))
    code_groups = {
        p: rng.choice(code_vocab_size, size=6, replace=False).tolist()
        for p in PHENOTYPES
    }
    for i in range(n_patients):
        hospital_idx = i % n_hospitals
        if rng.random() < non_iid_strength:
            phenotype_idx = hospital_idx % len(PHENOTYPES)
        else:
            phenotype_idx = int(rng.integers(0, len(PHENOTYPES)))
        phenotype = PHENOTYPES[phenotype_idx]
        static = base_centers[phenotype_idx] + rng.normal(0, 0.35, size=static_dim)
        visits = int(rng.integers(2, 8))
        time = np.linspace(0, 1, visits)
        slope = (phenotype_idx + 1) * 0.08
        temporal_base = rng.normal(0, 0.2, size=(visits, temporal_dim))
        temporal_base[:, 0] += slope * time
        temporal_base[:, 1] += math.sin(phenotype_idx + 1) * time
        codes = rng.choice(code_groups[phenotype], size=3, replace=False).tolist()
        note = (
            f"BioArc note for {phenotype}. {NOTE_TEMPLATES[phenotype]}. "
            f"Hospital node {hospital_idx}. Persian-English code switching may appear."
        )
        records.append(
            PatientRecord(
                patient_id=f"P{i:05d}",
                hospital_id=f"hospital_{hospital_idx}",
                phenotype=phenotype,
                static=static.astype(np.float32),
                temporal=temporal_base.astype(np.float32),
                codes=[int(c) for c in codes],
                note=note,
            )
        )
    return records


def load_jsonl(path: str | Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                PatientRecord(
                    patient_id=str(row["patient_id"]),
                    hospital_id=str(row["hospital_id"]),
                    phenotype=str(row.get("phenotype", "unknown")),
                    static=np.asarray(row["static"], dtype=np.float32),
                    temporal=np.asarray(row["temporal"], dtype=np.float32),
                    codes=[int(x) for x in row.get("codes", [])],
                    note=str(row.get("note", "")),
                )
            )
    return records


def save_jsonl(records: Sequence[PatientRecord], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def attach_note_embeddings(records: Sequence[PatientRecord], encoder, batch_size: int = 16) -> None:
    notes = [r.note for r in records]
    embs = []
    for start in range(0, len(notes), batch_size):
        embs.append(encoder.encode(notes[start : start + batch_size]))
    matrix = np.vstack(embs).astype(np.float32)
    for rec, emb in zip(records, matrix):
        rec.note_embedding = emb


class PatientDataset(Dataset):
    def __init__(self, records: Sequence[PatientRecord], code_vocab_size: int, note_dim: int):
        self.records = list(records)
        self.code_vocab_size = code_vocab_size
        self.note_dim = note_dim
        self.phenotype_to_id = {p: i for i, p in enumerate(sorted({r.phenotype for r in self.records}))}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        r = self.records[idx]
        codes = np.zeros(self.code_vocab_size, dtype=np.float32)
        for c in r.codes:
            if 0 <= c < self.code_vocab_size:
                codes[c] = 1.0
        if r.note_embedding is None:
            note_embedding = np.zeros(self.note_dim, dtype=np.float32)
        else:
            note_embedding = np.asarray(r.note_embedding, dtype=np.float32)
        return {
            "patient_id": r.patient_id,
            "hospital_id": r.hospital_id,
            "phenotype": r.phenotype,
            "label": self.phenotype_to_id[r.phenotype],
            "static": r.static.astype(np.float32),
            "temporal": r.temporal.astype(np.float32),
            "codes": codes,
            "note_embedding": note_embedding,
        }


def collate_batch(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    max_len = max(np.asarray(r["temporal"]).shape[0] for r in rows)
    temporal_dim = np.asarray(rows[0]["temporal"]).shape[1]
    temporal = np.zeros((len(rows), max_len, temporal_dim), dtype=np.float32)
    lengths = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        seq = np.asarray(r["temporal"], dtype=np.float32)
        temporal[i, : seq.shape[0], :] = seq
        lengths[i] = seq.shape[0]
    return {
        "patient_id": [str(r["patient_id"]) for r in rows],
        "hospital_id": [str(r["hospital_id"]) for r in rows],
        "phenotype": [str(r["phenotype"]) for r in rows],
        "label": torch.tensor([int(r["label"]) for r in rows], dtype=torch.long),
        "static": torch.tensor(np.stack([r["static"] for r in rows]), dtype=torch.float32),
        "temporal": torch.tensor(temporal, dtype=torch.float32),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "codes": torch.tensor(np.stack([r["codes"] for r in rows]), dtype=torch.float32),
        "note_embedding": torch.tensor(np.stack([r["note_embedding"] for r in rows]), dtype=torch.float32),
    }


def split_by_hospital(records: Sequence[PatientRecord]) -> Dict[str, List[PatientRecord]]:
    groups: Dict[str, List[PatientRecord]] = {}
    for r in records:
        groups.setdefault(r.hospital_id, []).append(r)
    return groups
