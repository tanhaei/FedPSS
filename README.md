# FedPSS

**Privacy-Governed Federated Patient-Record Similarity Retrieval over Temporal Multimodal Clinical Records using Multilingual Semantic Mapping**

This repository contains a Python implementation aligned with the revised FedPSS manuscript. It provides a reproducible scaffold for privacy-governed federated patient-record similarity retrieval over BioArc-style temporal, multimodal, structured clinical records.

The code supports:

- temporal structured EHR modeling,
- diagnosis/medication/procedure code encoding,
- structured Persian and multilingual clinical descriptor mapping,
- cross-modal attention fusion,
- federated model training with FedAvg,
- optional update clipping and Gaussian perturbation for differential-privacy-style stress testing,
- Top-K patient-record similarity retrieval,
- Precision@K, Recall@K, NDCG@K, and MRR evaluation,
- synthetic BioArc-like data for reproducible offline testing,
- plotting scripts that regenerate manuscript-style PDF figures.

The repository intentionally excludes raw identifiable patient records, raw clinical notes, and raw images. The public schema/configuration files are non-private reproducibility scaffolding only.

## Role of the Gemma-family mapper

The revised manuscript uses a Gemma-family backbone as a **structured descriptor semantic-mapping layer**, not as a diagnostic generator, free-text note interpreter, or autonomous clinical reasoning system. The mapper is intended to align Persian clinical descriptors, local abbreviations, medication classes, laboratory panel names, and related Arabic/English clinical terminology across BioArc institutions.

The offline test configuration uses a deterministic mock semantic mapper. The production-style configuration uses a Gemma-family mapper interface:

```yaml
semantic_mapper:
  backend: gemma_lora
  model_id: google/gemma-2-2b-it
  cache_descriptors: true
```

The model identifier is configurable because local Hugging Face access, licensing, and hardware availability may vary.

## Repository structure

```text
fedpss/
├── configs/
│   ├── gemma4.yaml              # Backward-compatible production-style config; uses Gemma-family mapper
│   └── test.yaml                # Fast offline config using MockSemanticMapper
├── data/
│   └── examples/
│       └── bioarc_schema_example.json
├── docs/
│   ├── BIOARC_DATA_FORMAT.md    # Expected privacy-transformed BioArc input schema
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
│   ├── llm/
│   │   ├── gemma4.py            # Backward-compatible import path for GemmaSemanticMapper
│   │   └── mock.py
│   ├── models/
│   └── retrieval/
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Installation

Recommended environment:

- Python 3.10+
- PyTorch
- Transformers for Gemma-family execution
- GPU recommended for actual Gemma-family inference

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

For Gemma-family checkpoints through Hugging Face, authenticate and accept the applicable model license:

```bash
huggingface-cli login
```

## Quick offline smoke test

This uses the mock semantic mapper and does not download model weights:

```bash
pytest -q
python scripts/run_synthetic_experiment.py --config configs/test.yaml --outdir outputs/test_run
```

Expected outputs:

```text
outputs/test_run/metrics.json
outputs/test_run/embeddings.npy
outputs/test_run/retrieval_performance.pdf
outputs/test_run/privacy_utility.pdf
outputs/test_run/non_iid_robustness.pdf
outputs/test_run/communication_cost.pdf
```

## Running with a Gemma-family semantic mapper

Use this only when you have sufficient memory and checkpoint access:

```bash
python scripts/run_synthetic_experiment.py --config configs/gemma4.yaml --outdir outputs/gemma_run
```

For real BioArc-style data, convert de-identified records into the JSONL schema described in `docs/BIOARC_DATA_FORMAT.md`, then pass:

```bash
python scripts/run_synthetic_experiment.py \
  --config configs/gemma4.yaml \
  --bioarc-jsonl /path/to/privacy_transformed_bioarc.jsonl \
  --outdir outputs/bioarc_gemma
```

## Input schema for real BioArc-style data

Each line in the JSONL file should represent one privacy-transformed patient-record object. A minimal flat example is:

```json
{
  "patient_id": "P0001",
  "hospital_id": "hospital_a",
  "phenotype": "glaucoma",
  "static": [0.73, 1.0, 0.0, 0.42, 0.61, 0.15],
  "temporal": [[0.1, 0.2, 0.0, 0.3, 0.4], [0.2, 0.1, 0.1, 0.4, 0.5]],
  "codes": [3, 9, 15],
  "descriptors": ["Persian ophthalmology descriptor: IOP trend", "medication class: prostaglandin analogue"]
}
```

`phenotype` is used only for synthetic/offline evaluation labels. If expert-labelled similar-record pairs are available, adapt the evaluator in `src/fedpss/metrics.py`.

## Main experimental modes

The repository implements the manuscript's three governed retrieval modes conceptually:

1. **Local-only retrieval**: each BioArc node trains and retrieves within its local index.
2. **Federated candidate retrieval**: hospitals train a shared embedding model and return anonymized Top-K candidates.
3. **Regional embedding index**: privacy-filtered embeddings are indexed centrally under governance.

The default script focuses on federated training and global retrieval evaluation.

## Notes on privacy

The privacy module includes update clipping and Gaussian perturbation. These functions support offline stress testing and should be paired with a formal privacy accountant and institutional review before prospective deployment. Raw clinical records, local descriptor dictionaries, and direct identifiers are not sent to the federated server in this implementation.

## Article figure generation

To regenerate manuscript-style PDF figures from a metrics JSON file:

```bash
python scripts/plot_article_figures.py --metrics outputs/test_run/metrics.json --outdir outputs/figures
```

If no metrics file is provided, the script generates manuscript-aligned default figures using the revised article values.

## Test status

The repository was tested in offline mode with:

```bash
pytest -q
python scripts/run_synthetic_experiment.py --config configs/test.yaml --outdir outputs/test_run
```

The Gemma-family checkpoint itself was not downloaded in the test environment. The integration code is isolated behind `GemmaSemanticMapper`; tests use `MockSemanticMapper` so the repository can be validated offline.
