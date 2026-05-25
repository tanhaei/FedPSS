# BioArc JSONL data format

Use one privacy-transformed patient-record object per line. Do **not** include national ID, phone number, raw address, exact date of birth, raw free-text notes, physician free-text identifiers, or any direct identifier.

The repository supports two input styles:

1. a compact flat JSONL format for experiments, and
2. the richer non-private schema in `schema/bioarc_record.schema.json` for governance documentation.

## Minimal flat experimental format

Required fields:

- `patient_id`: site-local pseudonymized ID.
- `hospital_id`: source hospital or BioArc node.
- `phenotype`: evaluation label or clinical group. Use only for offline evaluation; replace the evaluator if expert-pair labels are available.
- `static`: normalized static structured features, such as bucketed demographics and baseline measurements.
- `temporal`: list of visits; each visit is a fixed-size numeric vector based on relative offsets.
- `codes`: integer codes for diagnoses, medications, or procedures.
- `descriptors`: structured clinical descriptors to be semantically mapped, such as Persian lab-panel names, medication classes, specialty measurements, or normalized local synonyms.

Example:

```json
{
  "patient_id": "P0001",
  "hospital_id": "hospital_a",
  "phenotype": "glaucoma",
  "static": [0.73, 1.0, 0.0, 0.42, 0.61, 0.15],
  "temporal": [[0.10, 0.20, 0.00, 0.30, 0.40], [0.20, 0.10, 0.10, 0.40, 0.50]],
  "codes": [3, 9, 15],
  "descriptors": [
    "Persian ophthalmology descriptor: intraocular pressure trend",
    "medication class: prostaglandin analogue",
    "visual-field progression descriptor"
  ]
}
```

## Rich governance schema

The richer schema in `schema/bioarc_record.schema.json` records privacy status, source-node information, visit descriptors, clinical events, and structured descriptors. It is meant for non-private schema validation and documentation. It is not a public release of real BioArc records.

## Dimension assumptions

The code assumes fixed numeric dimensions set in the YAML config. If a real BioArc preprocessing pipeline has additional variables, update `model.static_dim`, `model.temporal_dim`, and `model.code_vocab_size`.

## Data governance reminder

The repository is for de-identified, privacy-transformed analytical records. Real BioArc source records are not publicly releasable and require institutional and BioArc governance authorization.
