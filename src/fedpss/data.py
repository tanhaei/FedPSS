from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


PHENOTYPES = ["glaucoma", "cataract", "diabetes", "hypertension", "retina"]
DESCRIPTOR_TEMPLATES = {
    "glaucoma": [
        "Persian ophthalmology descriptor: intraocular pressure trend",
        "visual-field progression descriptor",
        "medication class: prostaglandin analogue",
    ],
    "cataract": [
        "Persian ophthalmology descriptor: lens opacity",
        "procedure descriptor: cataract surgery planning",
        "visual acuity descriptor",
    ],
    "diabetes": [
        "Persian endocrine descriptor: HbA1c monitoring",
        "medication class: metformin",
        "retinal screening descriptor",
    ],
    "hypertension": [
        "Persian cardiovascular descriptor: blood pressure follow-up",
        "medication class: antihypertensive therapy",
        "cardiovascular risk descriptor",
    ],
    "retina": [
        "Persian retina descriptor: OCT finding",
        "macular edema descriptor",
        "visual acuity change descriptor",
    ],
}


@dataclass
class PatientRecord:
    patient_id: str
    hospital_id: str
    phenotype: str
    static: np.ndarray
    temporal: np.ndarray
    codes: List[int]
    descriptors: List[str]
    semantic_embedding: np.ndarray | None = None

    def to_json(self) -> Dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "hospital_id": self.hospital_id,
            "phenotype": self.phenotype,
            "static": self.static.astype(float).tolist(),
            "temporal": self.temporal.astype(float).tolist(),
            "codes": list(map(int, self.codes)),
            "descriptors": list(self.descriptors),
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

    The generator creates phenotype-specific structured patterns and hospital
    skew. It is not a clinical simulator and it does not create identifiable
    records or raw clinical notes.
    """
    rng = np.random.default_rng(seed)
    records: List[PatientRecord] = []
    base_centers = rng.normal(0, 1, size=(len(PHENOTYPES), static_dim))
    code_groups = {p: rng.choice(code_vocab_size, size=6, replace=False).tolist() for p in PHENOTYPES}
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
        descriptors = list(DESCRIPTOR_TEMPLATES[phenotype]) + [f"BioArc node {hospital_idx} structured descriptor"]
        records.append(
            PatientRecord(
                patient_id=f"P{i:05d}",
                hospital_id=f"hospital_{hospital_idx}",
                phenotype=phenotype,
                static=static.astype(np.float32),
                temporal=temporal_base.astype(np.float32),
                codes=[int(c) for c in codes],
                descriptors=descriptors,
            )
        )
    return records


def _extract_rich_schema_fields(row: dict) -> tuple[str, str, str, list[str], list[int]]:
    record_id = str(row.get("record_id", row.get("patient_id", "unknown_record")))
    source_node = row.get("source_node", {}) if isinstance(row.get("source_node", {}), dict) else {}
    hospital_id = str(source_node.get("node_id", row.get("hospital_id", "unknown_node")))
    labels = row.get("labels", {}) if isinstance(row.get("labels", {}), dict) else {}
    phenotype = str(row.get("phenotype", labels.get("task_notes", "unknown")))
    descriptors = []
    for d in row.get("descriptors", []):
        if isinstance(d, dict):
            descriptors.append(str(d.get("text", d.get("descriptor_id", ""))))
        else:
            descriptors.append(str(d))
    events = row.get("clinical_events", {}) if isinstance(row.get("clinical_events", {}), dict) else {}
    codes: list[int] = []
    for group in ["diagnoses", "procedures", "medications"]:
        for item in events.get(group, []) or []:
            token = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
            codes.append(abs(hash(token)) % 4096)
    return record_id, hospital_id, phenotype, descriptors, codes


def load_jsonl(path: str | Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "static" in row and "temporal" in row:
                patient_id = str(row.get("patient_id", row.get("record_id")))
                hospital_id = str(row.get("hospital_id", row.get("source_node", {}).get("node_id", "unknown_node")))
                phenotype = str(row.get("phenotype", "unknown"))
                descriptors = row.get("descriptors", [])
                if isinstance(descriptors, str):
                    descriptors = [descriptors]
                # Backward compatibility for older flat examples; do not document this as the preferred schema.
                if not descriptors and row.get("note"):
                    descriptors = [str(row["note"])]
                codes = [int(x) for x in row.get("codes", [])]
            else:
                patient_id, hospital_id, phenotype, descriptors, codes = _extract_rich_schema_fields(row)
                # For rich-schema records, numeric feature extraction should normally be performed upstream.
                # The fallback below creates fixed numeric placeholders so the loader remains testable.
                row.setdefault("static", [0.0] * 6)
                row.setdefault("temporal", [[0.0] * 5])
            records.append(
                PatientRecord(
                    patient_id=patient_id,
                    hospital_id=hospital_id,
                    phenotype=phenotype,
                    static=np.asarray(row["static"], dtype=np.float32),
                    temporal=np.asarray(row["temporal"], dtype=np.float32),
                    codes=codes,
                    descriptors=[str(x) for x in descriptors],
                )
            )
    return records


def save_jsonl(records: Sequence[PatientRecord], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def attach_semantic_embeddings(records: Sequence[PatientRecord], mapper, batch_size: int = 16) -> None:
    """Attach one mean-pooled semantic descriptor vector to each record."""
    all_descriptors: list[str] = []
    spans: list[tuple[int, int]] = []
    for r in records:
        start = len(all_descriptors)
        descriptors = r.descriptors or [""]
        all_descriptors.extend(descriptors)
        spans.append((start, len(all_descriptors)))

    chunks = []
    for start in range(0, len(all_descriptors), batch_size):
        chunks.append(mapper.encode(all_descriptors[start : start + batch_size]))
    matrix = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, 1), dtype=np.float32)
    for rec, (start, end) in zip(records, spans):
        emb = matrix[start:end].mean(axis=0)
        norm = np.linalg.norm(emb) + 1e-8
        rec.semantic_embedding = (emb / norm).astype(np.float32)


# Backward-compatible alias for older scripts.
attach_note_embeddings = attach_semantic_embeddings


class PatientDataset(Dataset):
    def __init__(self, records: Sequence[PatientRecord], code_vocab_size: int, semantic_dim: int | None = None, note_dim: int | None = None):
        self.records = list(records)
        self.code_vocab_size = code_vocab_size
        self.semantic_dim = int(semantic_dim if semantic_dim is not None else note_dim if note_dim is not None else 64)
        self.phenotype_to_id = {p: i for i, p in enumerate(sorted({r.phenotype for r in self.records}))}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        r = self.records[idx]
        codes = np.zeros(self.code_vocab_size, dtype=np.float32)
        for c in r.codes:
            if 0 <= c < self.code_vocab_size:
                codes[c] = 1.0
        if r.semantic_embedding is None:
            semantic_embedding = np.zeros(self.semantic_dim, dtype=np.float32)
        else:
            semantic_embedding = np.asarray(r.semantic_embedding, dtype=np.float32)
        return {
            "patient_id": r.patient_id,
            "hospital_id": r.hospital_id,
            "phenotype": r.phenotype,
            "label": self.phenotype_to_id[r.phenotype],
            "static": r.static.astype(np.float32),
            "temporal": r.temporal.astype(np.float32),
            "codes": codes,
            "semantic_embedding": semantic_embedding,
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
    semantic_key = "semantic_embedding" if "semantic_embedding" in rows[0] else "note_embedding"
    return {
        "patient_id": [str(r["patient_id"]) for r in rows],
        "hospital_id": [str(r["hospital_id"]) for r in rows],
        "phenotype": [str(r["phenotype"]) for r in rows],
        "label": torch.tensor([int(r["label"]) for r in rows], dtype=torch.long),
        "static": torch.tensor(np.stack([r["static"] for r in rows]), dtype=torch.float32),
        "temporal": torch.tensor(temporal, dtype=torch.float32),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "codes": torch.tensor(np.stack([r["codes"] for r in rows]), dtype=torch.float32),
        "semantic_embedding": torch.tensor(np.stack([r[semantic_key] for r in rows]), dtype=torch.float32),
    }


def split_by_hospital(records: Sequence[PatientRecord]) -> Dict[str, List[PatientRecord]]:
    groups: Dict[str, List[PatientRecord]] = {}
    for r in records:
        groups.setdefault(r.hospital_id, []).append(r)
    return groups
