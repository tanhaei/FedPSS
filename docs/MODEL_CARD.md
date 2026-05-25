# Model card

## Intended use

This repository implements research code for privacy-governed federated patient-record similarity retrieval. It is intended for retrospective retrieval experiments, cohort discovery, method development, and reproducibility scaffolding for the revised FedPSS manuscript.

## Not intended for

- autonomous diagnosis,
- autonomous treatment recommendation,
- clinical decision automation without physician review,
- direct deployment on identifiable patient records without institutional approval,
- replacing formal privacy accounting, security review, or hospital governance procedures.

## Semantic mapper role

The Gemma-family component is used as a structured descriptor semantic-mapping layer. It maps Persian and multilingual clinical descriptors into vectors for downstream retrieval. It does not interpret raw free-text notes, generate diagnoses, or act as a clinical reasoning module.

## Inputs

The downstream retrieval model learns cross-modal record embeddings from:

- static structured variables,
- temporal visit trajectories,
- diagnosis/medication/procedure code indicators,
- structured descriptor embeddings.

Raw images and unstructured clinical notes are excluded from the main implementation and are future-work modalities.

## Privacy and governance

The code keeps raw records local in the federated workflow. The privacy utilities support update clipping and Gaussian perturbation for offline stress testing. Production deployment requires secure aggregation, a formal privacy accountant, access control, audit logging, minimum cohort-size rules, and repeated-query monitoring.
