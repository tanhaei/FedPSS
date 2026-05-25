# Test results

The repository was tested in the execution environment with the offline mock semantic-mapper configuration.

## Unit tests

Command:

```bash
pytest -q
```

Result:

```text
3 passed in 12.75s
```

## End-to-end synthetic experiment

Command:

```bash
python scripts/run_synthetic_experiment.py --config configs/test.yaml --outdir outputs/ci_check
```

Result:

```text
Saved PDF figures to outputs/ci_check
Experiment complete
```

Returned metrics from the smoke test:

```json
{
  "local": {
    "precision@5": 0.8152380952380952,
    "recall@5": 0.6231681898348564,
    "ndcg@5": 0.9964320350732527,
    "mrr": 1.0
  },
  "centralized": {
    "precision@5": 0.9666666666666667,
    "recall@5": 0.5056216931216931,
    "ndcg@5": 1.0,
    "mrr": 1.0
  },
  "federated": {
    "precision@5": 0.8874999999999998,
    "recall@5": 0.43283730158730155,
    "ndcg@5": 0.9265624173970263,
    "mrr": 0.9583333333333334
  }
}
```

Generated files include:

- `outputs/ci_check/metrics.json`
- `outputs/ci_check/embeddings.npy`
- `outputs/ci_check/retrieval_performance.pdf`
- `outputs/ci_check/privacy_utility.pdf`
- `outputs/ci_check/non_iid_robustness.pdf`
- `outputs/ci_check/communication_cost.pdf`

## Manuscript-aligned figure defaults

Command:

```bash
python scripts/plot_article_figures.py --outdir outputs/article_default
```

Result:

```text
Saved PDF figures to outputs/article_default
```

This generates default figure values aligned with the revised manuscript, including 1.2 GB vs 45 MB communication burden, epsilon-based privacy-utility, and Dirichlet non-IID robustness.

Note: Gemma-family weights were not downloaded in this environment. The production semantic-mapper integration is implemented in `src/fedpss/llm/gemma4.py`; tests use `MockSemanticMapper` so the repository can be validated offline.
