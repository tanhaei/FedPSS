from torch.utils.data import DataLoader

from fedpss.data import PatientDataset, attach_semantic_embeddings, collate_batch, generate_synthetic_bioarc
from fedpss.llm.mock import MockSemanticMapper
from fedpss.metrics import evaluate_topk
from fedpss.models.multimodal import PatientSimilarityModel
from fedpss.training import embed_dataset


def test_model_forward_and_metrics():
    records = generate_synthetic_bioarc(n_patients=24, n_hospitals=3, static_dim=6, temporal_dim=5, code_vocab_size=32)
    attach_semantic_embeddings(records, MockSemanticMapper(dim=64), batch_size=8)
    ds = PatientDataset(records, code_vocab_size=32, semantic_dim=64)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_batch)
    model = PatientSimilarityModel(static_dim=6, temporal_dim=5, code_vocab_size=32, semantic_dim=64, hidden_dim=32, embedding_dim=16)
    emb = embed_dataset(model, loader)
    assert emb["embedding"].shape == (24, 16)
    assert emb["attention"].shape == (24, 4)
    metrics = evaluate_topk(emb["embedding"], emb["patient_id"], emb["label"], top_k=3)
    assert "precision@3" in metrics
    assert 0.0 <= metrics["precision@3"] <= 1.0
