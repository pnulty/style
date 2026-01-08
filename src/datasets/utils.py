from __future__ import annotations

from typing import Any, Dict

HUMAN_LABELS = {"human", "gold", "human_written", "real", "0", 0, False}
LLM_LABELS = {"llm", "machine", "generated", "ai", "1", 1, True}


def normalize_label(raw_label: Any) -> str:
    if raw_label in HUMAN_LABELS:
        return "human"
    if raw_label in LLM_LABELS:
        return "llm"
    if isinstance(raw_label, str):
        lowered = raw_label.strip().lower()
        if lowered in HUMAN_LABELS:
            return "human"
        if lowered in LLM_LABELS:
            return "llm"
    raise ValueError(f"Unrecognized label value: {raw_label!r}")


def build_metadata(split: str | None, extra: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if split:
        metadata["split"] = split
    metadata.update(extra)
    return metadata
