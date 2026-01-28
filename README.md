# Style Tests

Small experiment scripts for stylometry and syntax feature analysis.

## Gutenberg syntax features

Compute syntax features from local text files (no downloads).

```bash
python -m src.experiments.gutenberg_syntax_features --glob "*.txt" --list
python -m src.experiments.gutenberg_syntax_features --glob "*Stevenson*.txt"
```

Key flags:
- `--input-dir`: directory containing the local Gutenberg texts (default set in script)
- `--files`: explicit list of filenames/paths to process
- `--glob`: glob pattern within `--input-dir`
- `--list`: show resolved inputs and exit (no parsing)
- `--chunk-mode sentences`: compute features per sentence

### Parse-once sentence cache

Create a reusable sentence-level parse cache:

```bash
python -m src.experiments.gutenberg_syntax_features \
  --glob "*.txt" \
  --chunk-mode sentences \
  --write-sentence-cache
```

Reuse the cache without reparsing:

```bash
python -m src.experiments.gutenberg_syntax_features \
  --chunk-mode sentences \
  --use-sentence-cache
```

Cache flags:
- `--sentence-cache`: JSONL path to store/load sentence parses
- `--write-sentence-cache`: write sentence parses during feature computation
- `--use-sentence-cache`: compute features from the JSONL cache (no parsing)
- `--overwrite-sentence-cache`: replace an existing cache
- `--cache-summary`: print total sentences/texts and counts by author/title

## Baseline vs syntax evaluation

```bash
python -m src.experiments.baseline_vs_syntax
```

Key flags:
- `--data`: path to syntax features parquet
- `--metadata`: optional metadata CSV path to include in the report
- `--report`: output markdown report path
- `--figures`: output figures directory
