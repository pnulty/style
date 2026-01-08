"""Download Project Gutenberg texts and compute syntax features."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import spacy

DATA_DIR = Path("data")
GUTENBERG_DIR = DATA_DIR / "gutenberg"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_PATH = GUTENBERG_DIR / "metadata.csv"
FEATURES_PATH = PROCESSED_DIR / "gutenberg_syntax_features.parquet"

HEADER_RE = re.compile(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*")
FOOTER_RE = re.compile(r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*")

CLAUSE_DEPS = {"ccomp", "advcl", "xcomp", "acl", "relcl"}
MOTIF_DEFINITIONS = {
    "nsubj_VERB_dobj": ("VERB", "nsubj", "dobj"),
    "nsubj_VERB_obj": ("VERB", "nsubj", "obj"),
    "nsubj_VERB_iobj": ("VERB", "nsubj", "iobj"),
}


@dataclasses.dataclass(frozen=True)
class GutenbergText:
    text_id: int
    title: str
    author: str
    raw_text: str


def fetch_text(text_id: int) -> str:
    urls = [
        f"https://www.gutenberg.org/files/{text_id}/{text_id}-0.txt",
        f"https://www.gutenberg.org/files/{text_id}/{text_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{text_id}/pg{text_id}.txt",
    ]
    last_error: Exception | None = None
    for url in urls:
        try:
            with urlopen(url) as response:  # noqa: S310 - explicit download
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError) as exc:
            last_error = exc
    raise RuntimeError(f"Unable to download Gutenberg text {text_id}") from last_error


def parse_metadata(raw_text: str, text_id: int) -> tuple[str, str]:
    title = ""
    author = ""
    for line in raw_text.splitlines():
        if line.startswith("Title:"):
            title = line.replace("Title:", "", 1).strip()
        elif line.startswith("Author:"):
            author = line.replace("Author:", "", 1).strip()
        if title and author:
            break
    if not title:
        title = f"Gutenberg {text_id}"
    if not author:
        author = "Unknown"
    return title, author


def strip_header_footer(raw_text: str) -> str:
    lines = raw_text.splitlines()
    start_index = 0
    end_index = len(lines)
    for idx, line in enumerate(lines):
        if HEADER_RE.search(line):
            start_index = idx + 1
            break
    for idx in range(len(lines) - 1, -1, -1):
        if FOOTER_RE.search(lines[idx]):
            end_index = idx
            break
    return "\n".join(lines[start_index:end_index])


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def chunk_fixed_window(text: str, window_size: int) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), window_size):
        chunk_words = words[i : i + window_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def iter_chunks(text: str, mode: str, window_size: int) -> list[str]:
    if mode == "paragraphs":
        return chunk_paragraphs(text)
    if mode == "window":
        return chunk_fixed_window(text, window_size)
    raise ValueError(f"Unknown chunk mode: {mode}")


def store_text(text_id: int, text: str) -> Path:
    output_path = GUTENBERG_DIR / f"{text_id}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def ensure_metadata_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "author", "path"])
        writer.writeheader()


def download_gutenberg(text_id: int) -> GutenbergText:
    raw_text = fetch_text(text_id)
    title, author = parse_metadata(raw_text, text_id)
    return GutenbergText(text_id=text_id, title=title, author=author, raw_text=raw_text)


def dependency_depths(sent) -> list[int]:
    depths: dict[int, int] = {}

    def depth(token) -> int:
        if token.i in depths:
            return depths[token.i]
        if token.head == token:
            depths[token.i] = 0
        else:
            depths[token.i] = 1 + depth(token.head)
        return depths[token.i]

    return [depth(token) for token in sent]


def subtree_sizes(sent) -> list[int]:
    return [len(list(token.subtree)) for token in sent]


def branching_factors(sent) -> list[int]:
    return [len(list(token.children)) for token in sent]


def motif_counts(sent) -> Counter:
    counts: Counter = Counter()
    for token in sent:
        children_by_dep = {child.dep_: child for child in token.children}
        for motif_name, (head_pos, dep_left, dep_right) in MOTIF_DEFINITIONS.items():
            if token.pos_ == head_pos and dep_left in children_by_dep and dep_right in children_by_dep:
                counts[motif_name] += 1
    return counts


def clause_counts(sent) -> Counter:
    counts: Counter = Counter()
    for token in sent:
        if token.dep_ in CLAUSE_DEPS:
            counts[token.dep_] += 1
    return counts


def summarize_distribution(values: Iterable[int]) -> dict[str, float]:
    values_list = list(values)
    if not values_list:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": sum(values_list) / len(values_list),
        "max": max(values_list),
        "min": min(values_list),
    }


def iter_texts(text_ids: Iterable[int]) -> Iterator[GutenbergText]:
    for text_id in text_ids:
        yield download_gutenberg(text_id)


def compute_features(nlp, text_id: int, chunks: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chunk_id, chunk in enumerate(chunks):
        doc = nlp(chunk)
        sentence_depths: list[int] = []
        sentence_branching: list[int] = []
        sentence_subtree_sizes: list[int] = []
        motif_counter: Counter = Counter()
        clause_counter: Counter = Counter()
        token_count = 0
        sentence_count = 0

        for sent in doc.sents:
            sentence_count += 1
            token_count += len(sent)
            sentence_depths.extend(dependency_depths(sent))
            sentence_branching.extend(branching_factors(sent))
            sentence_subtree_sizes.extend(subtree_sizes(sent))
            motif_counter.update(motif_counts(sent))
            clause_counter.update(clause_counts(sent))

        depth_summary = summarize_distribution(sentence_depths)
        branching_summary = summarize_distribution(sentence_branching)
        subtree_summary = summarize_distribution(sentence_subtree_sizes)

        row: dict[str, object] = {
            "text_id": text_id,
            "chunk_id": chunk_id,
            "token_count": token_count,
            "sentence_count": sentence_count,
            "dependency_depth_mean": depth_summary["mean"],
            "dependency_depth_max": depth_summary["max"],
            "dependency_depth_min": depth_summary["min"],
            "branching_factor_mean": branching_summary["mean"],
            "branching_factor_max": branching_summary["max"],
            "branching_factor_min": branching_summary["min"],
            "subtree_size_mean": subtree_summary["mean"],
            "subtree_size_max": subtree_summary["max"],
            "subtree_size_min": subtree_summary["min"],
        }

        for motif_name in MOTIF_DEFINITIONS:
            row[f"motif_{motif_name}_count"] = motif_counter.get(motif_name, 0)
        for dep in CLAUSE_DEPS:
            row[f"clause_{dep}_count"] = clause_counter.get(dep, 0)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Gutenberg texts and compute syntax features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python -m src.experiments.gutenberg_syntax_features --ids 1342 84
            """
        ),
    )
    parser.add_argument("--ids", nargs="+", type=int, default=[1342, 84])
    parser.add_argument("--chunk-mode", choices=["paragraphs", "window"], default="paragraphs")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    GUTENBERG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load(args.spacy_model)

    metadata_rows = []
    feature_rows: list[dict[str, object]] = []

    for gutenberg_text in iter_texts(args.ids):
        cleaned = normalize_text(strip_header_footer(gutenberg_text.raw_text))
        chunks = iter_chunks(cleaned, args.chunk_mode, args.window_size)
        text_path = store_text(gutenberg_text.text_id, cleaned)
        metadata_rows.append(
            {
                "id": gutenberg_text.text_id,
                "title": gutenberg_text.title,
                "author": gutenberg_text.author,
                "path": text_path.as_posix(),
            }
        )
        feature_rows.extend(compute_features(nlp, gutenberg_text.text_id, chunks))

    ensure_metadata_header(METADATA_PATH)
    with METADATA_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "author", "path"])
        for row in metadata_rows:
            writer.writerow(row)

    df = pd.DataFrame(feature_rows)
    df.to_parquet(FEATURES_PATH, index=False)


if __name__ == "__main__":
    main()
