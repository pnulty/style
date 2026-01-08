# Creative Writing Corpus (Processed)

## Schema
Each row in `creative_writing_corpus.parquet` follows the normalized schema:

- `text`: The creative-writing passage.
- `label`: `human` or `llm`.
- `source`: Dataset name (`detectrl`, `ghostbuster`).
- `domain`: Fixed value `creative_writing`.
- `metadata`: JSON object with dataset-specific fields (including `split` and the raw row).

## Preprocessing
The data are loaded from `data/raw/{detectrl,ghostbuster}` where splits are expected as
`train/validation/test` JSONL or CSV files. Rows are normalized by:

1. Selecting the first available text field (`text`, `story`, `content`, etc.).
2. Mapping labels to `human` or `llm` using common aliases (`0`/`1`, `real`/`generated`, etc.).
3. Recording the split and raw row inside `metadata`.

Use `python -m src.datasets.build_corpus` to regenerate the Parquet file once
`pyarrow` is available; otherwise the script emits JSONL content with the `.parquet`
suffix for portability.
