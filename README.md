# FedPSS-Gemma4

**Privacy-Preserving Federated Patient Similarity Search over Temporal Multimodal Clinical Records using LLM-Augmented Representations**

This repository contains a Python implementation aligned with the paper concept. It provides a complete experimental scaffold for federated patient similarity search over BioArc-style clinical records. The code supports:

- temporal structured EHR modeling,
- diagnosis/medication/procedure code encoding,
- clinical-note encoding with **Gemma 4**,
- cross-modal attention fusion,
- federated model training with FedAvg,
- optional differential privacy style update clipping and Gaussian noise,
- Top-K patient similarity retrieval,
- Precision@K, Recall@K, NDCG@K, and MRR evaluation,
- synthetic BioArc-like data for reproducible testing,
- plotting scripts that generate PDF figures for the manuscript.

The code is intentionally designed so that tests can run offline with a deterministic mock LLM encoder, while the production configuration uses Gemma 4 for clinical note embeddings.

## Why Gemma 4?

The paper uses an LLM as a **frozen representation module** for clinical notes, not as a diagnostic decision-maker. The repository implements `Gemma4NoteEncoder`, which extracts text embeddings from a Gemma 4 checkpoint via Hugging Face Transformers. The default Gemma 4 configuration uses:

```yaml
llm:
  backend: gemma4
  model_id: google/gemma-4-E2B-it
```

You may change the model identifier if your local Hugging Face installation exposes a different Gemma 4 checkpoint name.

## Repository structure

```text
fedpss-gemma4/
├── configs/
│   ├── gemma4.yaml              # Production-style config using Gemma 4
│   └── test.yaml                # Fast offline config using MockNoteEncoder
├── data/
│   └── examples/
│       └── bioarc_schema_example.json
├── docs/
│   ├── BIOARC_DATA_FORMAT.md    # Expected real BioArc input schema
│   └── MODEL_CARD.md            # Intended use and limitations
├── scripts/
│   ├── run_synthetic_experiment.py
│   └── plot_article_figures.py
├── src/fedpss/
│   ├── config.py
│   ├── data.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── privacy.py
│   ├── training.py
│   ├── federated/
│   │   ├── client.py
│   │   └── server.py
│   ├── llm/
│   │   ├── gemma4.py
│   │   └── mock.py
│   ├── models/
│   │   └── multimodal.py
│   └── retrieval/
│       └── index.py
├── tests/
│   ├── test_data.py
│   ├── test_model_and_metrics.py
│   └── test_federated.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Recommended environment:

- Python 3.10+
- PyTorch
- Transformers for Gemma 4 execution
- GPU recommended for actual Gemma 4 inference

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

For Gemma 4 through Hugging Face, you may need to authenticate and accept the model license:

```bash
huggingface-cli login
```

## Quick offline smoke test

This uses the mock encoder and does not download any LLM weights:

```bash
pytest -q
python scripts/run_synthetic_experiment.py --config configs/test.yaml --outdir outputs/test_run
```

Expected outputs:

```text
outputs/test_run/metrics.json
outputs/test_run/retrieval_performance.pdf
outputs/test_run/privacy_utility.pdf
outputs/test_run/non_iid_robustness.pdf
outputs/test_run/communication_cost.pdf
```

## Running with Gemma 4

Use this when you have enough memory and access to the Gemma 4 checkpoint:

```bash
python scripts/run_synthetic_experiment.py --config configs/gemma4.yaml --outdir outputs/gemma4_run
```

For real BioArc data, convert your de-identified records into the JSONL schema described in `docs/BIOARC_DATA_FORMAT.md`, then pass:

```bash
python scripts/run_synthetic_experiment.py \
  --config configs/gemma4.yaml \
  --bioarc-jsonl /path/to/deidentified_bioarc.jsonl \
  --outdir outputs/bioarc_gemma4
```

## Input schema for real BioArc data

Each line in the JSONL file should represent one de-identified patient:

```json
{
  "patient_id": "P0001",
  "hospital_id": "hospital_a",
  "phenotype": "glaucoma",
  "static": [0.73, 1.0, 0.0, 0.42, 0.61, 0.15],
  "temporal": [[0.1, 0.2, 0.0, 0.3], [0.2, 0.1, 0.1, 0.4]],
  "codes": [3, 9, 15],
  "note": "Persian or code-switched clinical note text..."
}
```

`phenotype` is used only for evaluation labels. If expert-labeled similar-patient pairs are available, adapt the evaluator in `src/fedpss/metrics.py`.

## Main experimental modes

The repository implements the paper's three modes:

1. **Local-only retrieval**: each BioArc node trains and retrieves within its local index.
2. **Federated candidate retrieval**: hospitals train a shared embedding model and return anonymized Top-K candidates.
3. **Regional embedding index**: privacy-filtered embeddings are indexed centrally under governance.

The default script focuses on federated training and global retrieval evaluation.

## Notes on privacy

The privacy layer includes update clipping and Gaussian perturbation. This is an experimental DP-like mechanism and should be replaced with a formal accountant for production-grade privacy guarantees. Raw clinical records are never sent to the federated server in this implementation.

## Article figure generation

To regenerate manuscript-style PDF figures from a metrics JSON file:

```bash
python scripts/plot_article_figures.py --metrics outputs/test_run/metrics.json --outdir outputs/figures
```

If no metrics file is provided, the script generates placeholder figures with `[TBD]`-style synthetic values for layout validation.

## Test status

The repository was tested in offline mode with:

```bash
pytest -q
python scripts/run_synthetic_experiment.py --config configs/test.yaml --outdir outputs/test_run
```

Gemma 4 itself was not downloaded in the test environment. The Gemma 4 integration code is included and isolated behind `Gemma4NoteEncoder`.
