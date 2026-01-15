from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .detectrl import load_detectrl
from .gutenberg_authors import DEFAULT_OUTPUT_PATH as DEFAULT_GUTENBERG_OUTPUT
from .ghostbuster import load_ghostbuster
from .schema import NormalizedExample


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/creative_writing_corpus.parquet")


def build_corpus(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> List[NormalizedExample]:
    datasets: list[NormalizedExample] = []
    if (raw_dir / "detectrl").exists():
        datasets.extend(load_detectrl(raw_dir))
    else:
        print("DetectRL raw data not found; skipping.")
    if (raw_dir / "ghostbuster").exists():
        datasets.extend(load_ghostbuster(raw_dir))
    else:
        print("Ghostbuster raw data not found; skipping.")

    gutenberg_path = DEFAULT_GUTENBERG_OUTPUT
    if gutenberg_path.exists():
        datasets.extend(_load_gutenberg(gutenberg_path))
    else:
        print("Gutenberg author corpus not found; skipping.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(output_path, datasets)
    return datasets


def _load_gutenberg(path: Path) -> List[NormalizedExample]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to load Gutenberg corpus parquet.") from exc

    df = pd.read_parquet(path)
    return [
        NormalizedExample(
            text=row["text"],
            label=row["label"],
            source=row.get("source", "gutenberg"),
            domain=row.get("domain", "prose"),
            metadata=row.get("metadata", {}),
        )
        for _, row in df.iterrows()
    ]


def write_parquet(path: Path, rows: Iterable[NormalizedExample]) -> None:
    data = [row.to_dict() for row in rows]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        _write_jsonl_with_parquet_suffix(path, data)
        return

    table = pa.Table.from_pylist(data)
    pq.write_table(table, path)


def _write_jsonl_with_parquet_suffix(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in data:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_corpus()
