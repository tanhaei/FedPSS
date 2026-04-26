from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    sim = embeddings @ embeddings.T / temperature
    logits_mask = torch.ones_like(sim, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    labels = labels.view(-1, 1)
    positives = (labels == labels.T) & logits_mask
    exp_sim = torch.exp(sim) * logits_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8))
    pos_count = positives.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return (embeddings @ embeddings.T).mean() * 0.0
    loss = -(log_prob * positives.float()).sum(dim=1)[valid] / pos_count[valid].float()
    return loss.mean()


def train_one_epoch(model, loader, optimizer, device: str = "cpu") -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        optimizer.zero_grad()
        embeddings, _ = model(batch)
        loss = supervised_contrastive_loss(embeddings, batch["label"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


def embed_dataset(model, loader, device: str = "cpu"):
    model.eval()
    ids, labels, phenotypes, hospitals, embeddings, attentions = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            emb, att = model(tensor_batch)
            ids.extend(batch["patient_id"])
            phenotypes.extend(batch["phenotype"])
            hospitals.extend(batch["hospital_id"])
            labels.extend(batch["label"].cpu().tolist())
            embeddings.append(emb.cpu())
            attentions.append(att.cpu())
    return {
        "patient_id": ids,
        "phenotype": phenotypes,
        "hospital_id": hospitals,
        "label": labels,
        "embedding": torch.cat(embeddings, dim=0).numpy(),
        "attention": torch.cat(attentions, dim=0).numpy(),
    }
