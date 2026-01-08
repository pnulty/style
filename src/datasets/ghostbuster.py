from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import NormalizedExample
from .utils import build_metadata, normalize_label

SUPPORTED_SPLITS = ("train", "validation", "test")
TEXT_FIELDS = ("text", "story", "document", "content")
LABEL_FIELDS = ("label", "source", "generated", "is_ai", "ai_generated")


def available_splits(data_dir: Path) -> list[str]:
    splits = []
    for split in SUPPORTED_SPLITS:
        if (data_dir / f"{split}.jsonl").exists() or (data_dir / f"{split}.csv").exists():
            splits.append(split)
    return splits


def load_ghostbuster(base_dir: Path) -> list[NormalizedExample]:
    data_dir = base_dir / "ghostbuster"
    examples: list[NormalizedExample] = []
    for split in available_splits(data_dir):
        for row in iter_rows(data_dir, split):
            text = _get_first(row, TEXT_FIELDS)
            label = normalize_label(_get_first(row, LABEL_FIELDS))
            metadata = build_metadata(split, {"raw": row})
            examples.append(
                NormalizedExample(
                    text=text,
                    label=label,
                    source="ghostbuster",
                    domain="creative_writing",
                    metadata=metadata,
                )
            )
    return examples


def iter_rows(data_dir: Path, split: str) -> Iterator[dict[str, object]]:
    jsonl_path = data_dir / f"{split}.jsonl"
    csv_path = data_dir / f"{split}.csv"
    if jsonl_path.exists():
        yield from _read_jsonl(jsonl_path)
        return
    if csv_path.exists():
        yield from _read_csv(csv_path)
        return
    raise FileNotFoundError(f"No Ghostbuster data found for split '{split}'.")


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_csv(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def _get_first(row: dict[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    raise KeyError(f"Missing expected keys {list(keys)} in row: {row}")
