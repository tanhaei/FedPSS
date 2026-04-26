from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np

from fedpss.retrieval.index import SimilarityIndex


def dcg(relevance: Sequence[int]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))


def evaluate_topk(embeddings: np.ndarray, ids: list[str], labels: list[int], top_k: int = 10) -> Dict[str, float]:
    index = SimilarityIndex(embeddings, ids, labels)
    label_to_ids: Dict[int, Set[str]] = {}
    for pid, label in zip(ids, labels):
        label_to_ids.setdefault(int(label), set()).add(pid)
    precisions, recalls, ndcgs, mrrs = [], [], [], []
    for row_idx, pid in enumerate(ids):
        label = int(labels[row_idx])
        relevant = set(label_to_ids[label]) - {pid}
        if not relevant:
            continue
        results = index.search(embeddings[row_idx], top_k=top_k, exclude_id=pid)
        retrieved = [rid for rid, _, _ in results]
        hits = [1 if rid in relevant else 0 for rid in retrieved]
        precisions.append(sum(hits) / max(len(retrieved), 1))
        recalls.append(sum(hits) / len(relevant))
        ideal = [1] * min(len(relevant), top_k)
        ndcgs.append(dcg(hits) / max(dcg(ideal), 1e-8))
        rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    return {
        f"precision@{top_k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{top_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }
