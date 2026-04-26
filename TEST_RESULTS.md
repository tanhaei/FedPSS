# Test results

The repository was tested in the execution environment with the offline mock LLM configuration.

## Unit tests

Command:

```bash
PYTHONPATH=src pytest -q
```

Result:

```text
3 passed in 26.36s
```

## End-to-end synthetic experiment

Command:

```bash
python scripts/run_synthetic_experiment.py --config configs/test.yaml --outdir outputs/test_run
```

Result:

```text
Saved PDF figures to outputs/test_run
Experiment complete
```

Generated files include:

- `outputs/test_run/metrics.json`
- `outputs/test_run/embeddings.npy`
- `outputs/test_run/retrieval_performance.pdf`
- `outputs/test_run/privacy_utility.pdf`
- `outputs/test_run/non_iid_robustness.pdf`
- `outputs/test_run/communication_cost.pdf`

Note: Gemma 4 weights were not downloaded in this environment. The production Gemma 4 integration is implemented in `src/fedpss/llm/gemma4.py`; tests use `MockNoteEncoder` so the repository can be validated offline.
