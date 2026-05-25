from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedpss.data import (
    PatientDataset,
    attach_semantic_embeddings,
    collate_batch,
    generate_synthetic_bioarc,
    load_jsonl,
    set_seed,
    split_by_hospital,
)
from fedpss.federated.client import FederatedClient
from fedpss.federated.server import FederatedServer
from fedpss.llm.gemma4 import build_semantic_mapper
from fedpss.metrics import evaluate_topk
from fedpss.models.multimodal import build_model
from fedpss.training import embed_dataset, train_one_epoch


def _build_records(cfg: dict, bioarc_jsonl: str | None = None):
    exp = cfg["experiment"]
    model_cfg = cfg["model"]
    if bioarc_jsonl:
        records = load_jsonl(bioarc_jsonl)
    else:
        records = generate_synthetic_bioarc(
            n_patients=int(exp["n_patients"]),
            n_hospitals=int(exp["n_hospitals"]),
            static_dim=int(model_cfg["static_dim"]),
            temporal_dim=int(model_cfg["temporal_dim"]),
            code_vocab_size=int(model_cfg["code_vocab_size"]),
            non_iid_strength=float(exp.get("non_iid_strength", 0.7)),
            seed=int(cfg.get("seed", 42)),
        )
    return records


def _semantic_cfg(cfg: dict) -> dict:
    model_cfg = cfg["model"]
    mapper_cfg = dict(cfg.get("semantic_mapper", cfg.get("llm", {})))
    mapper_cfg.setdefault("semantic_dim", model_cfg.get("semantic_dim", model_cfg.get("note_dim", 64)))
    return mapper_cfg


def _dataset(records, model_cfg: dict) -> PatientDataset:
    return PatientDataset(
        records,
        int(model_cfg["code_vocab_size"]),
        semantic_dim=int(model_cfg.get("semantic_dim", model_cfg.get("note_dim", 64))),
    )


def run_experiment(cfg: dict, outdir: str | Path, bioarc_jsonl: str | None = None, device: str = "cpu") -> dict:
    set_seed(int(cfg.get("seed", 42)))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_cfg = cfg["model"]
    exp_cfg = cfg["experiment"]
    mapper_cfg = _semantic_cfg(cfg)
    mapper = build_semantic_mapper(mapper_cfg)
    records = _build_records(cfg, bioarc_jsonl=bioarc_jsonl)
    attach_semantic_embeddings(records, mapper, batch_size=int(mapper_cfg.get("batch_size", 16)))

    full_ds = _dataset(records, model_cfg)
    full_loader = DataLoader(full_ds, batch_size=int(exp_cfg["batch_size"]), shuffle=False, collate_fn=collate_batch)

    # Centralized upper-bound style model. This is not a privacy-governed deployment mode.
    centralized = build_model(model_cfg).to(device)
    optimizer = torch.optim.Adam(centralized.parameters(), lr=float(exp_cfg["learning_rate"]))
    train_loader = DataLoader(full_ds, batch_size=int(exp_cfg["batch_size"]), shuffle=True, collate_fn=collate_batch)
    for _ in range(max(1, int(exp_cfg.get("local_epochs", 1)) * int(exp_cfg.get("n_rounds", 1)))):
        train_one_epoch(centralized, train_loader, optimizer, device=device)
    central_emb = embed_dataset(centralized, full_loader, device=device)
    top_k = int(exp_cfg.get("top_k", 10))
    central_metrics = evaluate_topk(central_emb["embedding"], central_emb["patient_id"], central_emb["label"], top_k=top_k)

    # Federated model with local training and weighted FedAvg aggregation.
    global_model = build_model(model_cfg).to(device)
    server = FederatedServer(global_model)
    clients = []
    for hospital, rows in split_by_hospital(records).items():
        ds = _dataset(rows, model_cfg)
        clients.append(
            FederatedClient(
                hospital,
                ds,
                batch_size=int(exp_cfg["batch_size"]),
                lr=float(exp_cfg["learning_rate"]),
                local_epochs=int(exp_cfg["local_epochs"]),
                device=device,
            )
        )
    round_metrics = []
    for _round_idx in range(int(exp_cfg["n_rounds"])):
        updates = []
        for client in clients:
            delta = client.fit(global_model, privacy_cfg=cfg.get("privacy", {}))
            updates.append((client.n_samples, delta))
        server.aggregate(updates)
        fed_emb = embed_dataset(global_model, full_loader, device=device)
        round_metrics.append(evaluate_topk(fed_emb["embedding"], fed_emb["patient_id"], fed_emb["label"], top_k=top_k))

    final_emb = embed_dataset(global_model, full_loader, device=device)
    federated_metrics = evaluate_topk(final_emb["embedding"], final_emb["patient_id"], final_emb["label"], top_k=top_k)

    # Local-only baseline: each site trains and retrieves only on its local records.
    local_scores = []
    for _hospital, rows in split_by_hospital(records).items():
        local_model = build_model(model_cfg).to(device)
        ds = _dataset(rows, model_cfg)
        loader = DataLoader(ds, batch_size=int(exp_cfg["batch_size"]), shuffle=True, collate_fn=collate_batch)
        opt = torch.optim.Adam(local_model.parameters(), lr=float(exp_cfg["learning_rate"]))
        for _ in range(max(1, int(exp_cfg["local_epochs"]))):
            train_one_epoch(local_model, loader, opt, device=device)
        eval_loader = DataLoader(ds, batch_size=int(exp_cfg["batch_size"]), shuffle=False, collate_fn=collate_batch)
        emb = embed_dataset(local_model, eval_loader, device=device)
        local_scores.append(evaluate_topk(emb["embedding"], emb["patient_id"], emb["label"], top_k=min(top_k, max(1, len(rows) - 1))))
    local_metrics = {}
    if local_scores:
        keys = local_scores[0].keys()
        for key in keys:
            local_metrics[key] = float(np.mean([score[key] for score in local_scores]))

    result = {
        "config": cfg,
        "n_patients": len(records),
        "n_hospitals": len(split_by_hospital(records)),
        "top_k": top_k,
        "metrics": {
            "local": local_metrics,
            "centralized": central_metrics,
            "federated": federated_metrics,
            "rounds": round_metrics,
        },
        "attention_mean": final_emb["attention"].mean(axis=0).tolist(),
        "modalities": ["static", "temporal", "codes", "semantic_map"],
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.save(outdir / "embeddings.npy", final_emb["embedding"])
    return result
