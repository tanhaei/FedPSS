from fedpss.data import generate_synthetic_bioarc, PatientDataset
from fedpss.llm.mock import MockNoteEncoder
from fedpss.data import attach_note_embeddings


def test_synthetic_records_have_expected_shapes():
    records = generate_synthetic_bioarc(n_patients=12, n_hospitals=3, static_dim=6, temporal_dim=5, code_vocab_size=32)
    encoder = MockNoteEncoder(dim=64)
    attach_note_embeddings(records, encoder, batch_size=4)
    ds = PatientDataset(records, code_vocab_size=32, note_dim=64)
    row = ds[0]
    assert row["static"].shape == (6,)
    assert row["temporal"].shape[1] == 5
    assert row["codes"].shape == (32,)
    assert row["note_embedding"].shape == (64,)
