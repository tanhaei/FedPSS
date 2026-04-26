# BioArc JSONL data format

Use one de-identified patient per line. Do not include national ID, phone number, raw address, physician free-text identifiers, or any direct identifier.

Required fields:

- `patient_id`: local pseudonymized ID.
- `hospital_id`: source hospital or BioArc node.
- `phenotype`: evaluation label or clinical group. Use `[TBD]` if unavailable and replace the evaluator with expert-pair labels.
- `static`: normalized static structured features, for example demographics and baseline measurements.
- `temporal`: list of visits; each visit is a fixed-size numeric vector.
- `codes`: integer codes for diagnoses, medications, or procedures.
- `note`: clinical note text to be encoded by Gemma 4.

The code assumes fixed dimensions set in the YAML config. If your real BioArc schema has additional variables, update `model.static_dim`, `model.temporal_dim`, and `model.code_vocab_size`.
