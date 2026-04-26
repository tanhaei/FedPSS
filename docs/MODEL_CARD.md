# Model card

## Intended use

This repository implements research code for privacy-preserving federated patient similarity search. It is intended for retrospective retrieval experiments, cohort discovery, and method development.

## Not intended for

- autonomous diagnosis,
- clinical decision automation without physician review,
- direct deployment on identifiable patient records without institutional approval,
- replacing formal privacy accounting or security review.

## LLM role

Gemma 4 is used as a frozen clinical-note representation module. The model does not generate diagnoses and is not used as an oracle. The downstream embedding model learns cross-modal patient representations from structured, temporal, code, and note embeddings.
