"""Compute syntax features for local Project Gutenberg texts."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import textwrap
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import spacy
from spacy.tokens import Doc

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
LOCAL_GUTENBERG_DIR = Path("/home/paul/style-tests/data/human-gutenberg-sample")
METADATA_PATH = PROCESSED_DIR / "gutenberg_metadata.csv"
FEATURES_PATH = PROCESSED_DIR / "gutenberg_syntax_features.parquet"
SENTENCE_CACHE_PATH = PROCESSED_DIR / "gutenberg_sentence_cache.jsonl"

HEADER_RE = re.compile(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*")
FOOTER_RE = re.compile(r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*")

CLAUSE_DEPS = {"ccomp", "advcl", "xcomp", "acl", "relcl"}
DEFAULT_LITERAL_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
}


@dataclasses.dataclass(frozen=True)
class GutenbergText:
    text_id: int
    title: str
    author: str
    raw_text: str
    source_path: Path


def parse_filename_metadata(path: Path) -> tuple[str, str]:
    stem = path.stem
    parts = stem.split("-")
    title_part = parts[0] if parts else stem
    author_part = parts[1] if len(parts) > 1 else "Unknown"
    title = title_part.replace("_", " ").strip() or "Unknown title"
    author = author_part.replace("_", " ").strip() or "Unknown"
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


def limit_text_to_sentences(nlp, text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return ""
    sentences: list[str] = []
    for doc in nlp.pipe(chunk_paragraphs(text), batch_size=10):
        for sent in doc.sents:
            sentences.append(sent.text)
            if len(sentences) >= max_sentences:
                return " ".join(sentences)
    return " ".join(sentences)


def iter_sentence_docs(nlp, text: str) -> Iterator[Doc]:
    for doc in nlp.pipe(chunk_paragraphs(text), batch_size=10):
        for sent in doc.sents:
            yield sent.as_doc()


def ensure_metadata_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "author", "path"])
        writer.writeheader()

def load_local_text(path: Path, text_id: int) -> GutenbergText:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    title, author = parse_filename_metadata(path)
    return GutenbergText(
        text_id=text_id,
        title=title,
        author=author,
        raw_text=raw_text,
        source_path=path,
    )

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


def load_literal_words(args: argparse.Namespace) -> set[str]:
    words = set(DEFAULT_LITERAL_WORDS)
    if args.literal_words:
        words.update(word.strip().lower() for word in args.literal_words.split(",") if word.strip())
    if args.literal_words_file:
        for line in Path(args.literal_words_file).read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lower()
            if stripped:
                words.add(stripped)
    return words


def child_label(token, mode: str, literal_words: set[str]) -> str:
    if mode == "mixed" and token.lower_ in literal_words:
        return f"tok:{token.lower_}"
    return f"pos:{token.pos_}|dep:{token.dep_}"


def motif_counts(sent, *, mode: str, literal_words: set[str], min_leaves: int) -> Counter:
    counts: Counter = Counter()
    for token in sent:
        child_labels = [child_label(child, mode, literal_words) for child in token.children]
        if len(child_labels) < min_leaves:
            continue
        for size in range(min_leaves, len(child_labels) + 1):
            for combo in combinations(child_labels, size):
                leaves = "+".join(sorted(combo))
                motif_key = f"{token.pos_}:{leaves}"
                counts[motif_key] += 1
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


def iter_texts(paths: Iterable[Path]) -> Iterator[GutenbergText]:
    for text_id, path in enumerate(paths, start=1):
        yield load_local_text(path, text_id)


def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.files:
        for path in args.files:
            path = Path(path)
            if not path.is_absolute():
                path = args.input_dir / path
            paths.append(path)
    if args.glob:
        paths.extend(sorted(args.input_dir.glob(args.glob)))
    if not paths:
        raise ValueError(
            "Provide --files or --glob to select input texts; refusing to scan the full directory by default."
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = "\n".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"Missing input files:\n{missing_list}")
    return paths


def prepare_sentence_cache_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Sentence cache already exists at {path.as_posix()}. Use --overwrite-sentence-cache to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def write_sentence_cache_entry(handle, entry: dict[str, object]) -> None:
    handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def iter_sentence_cache(path: Path, nlp) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            doc_json = entry.get("doc_json", {})
            doc = Doc(nlp.vocab).from_json(doc_json)
            entry["doc"] = doc
            yield entry


def summarize_sentence_cache(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Sentence cache not found at {path.as_posix()}.")
    author_counts: Counter = Counter()
    title_counts: Counter = Counter()
    text_ids: set[int] = set()
    sentence_total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sentence_total += 1
            text_ids.add(int(entry.get("text_id", 0)))
            author_counts[entry.get("author", "Unknown")] += 1
            title_counts[entry.get("title", "Unknown")] += 1
    return {
        "sentence_total": sentence_total,
        "text_total": len(text_ids),
        "authors": author_counts,
        "titles": title_counts,
    }


def compute_features(
    nlp,
    text_id: int,
    chunks: list[str],
    *,
    motif_mode: str,
    literal_words: set[str],
    motif_min_leaves: int,
) -> list[dict[str, object]]:
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
            motif_counter.update(
                motif_counts(
                    sent,
                    mode=motif_mode,
                    literal_words=literal_words,
                    min_leaves=motif_min_leaves,
                )
            )
            clause_counter.update(clause_counts(sent))

        depth_summary = summarize_distribution(sentence_depths)
        branching_summary = summarize_distribution(sentence_branching)
        subtree_summary = summarize_distribution(sentence_subtree_sizes)

        row: dict[str, object] = {
            "text_id": text_id,
            "chunk_id": chunk_id,
            "text": chunk,
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
            "motif_counts": dict(motif_counter),
        }

        for dep in sorted(CLAUSE_DEPS):
            row[f"clause_{dep}_count"] = clause_counter.get(dep, 0)
        rows.append(row)
    return rows


def compute_features_from_sentence_doc(
    text_id: int, sentence_id: int, doc: Doc, *, mode: str, literal_words: set[str], min_leaves: int
) -> dict[str, object]:
    sentence_depths: list[int] = []
    sentence_branching: list[int] = []
    sentence_subtree_sizes: list[int] = []
    motif_counter: Counter = Counter()
    clause_counter: Counter = Counter()

    sentence_count = 0
    token_count = 0

    for sent in doc.sents:
        sentence_count += 1
        token_count += len(sent)
        sentence_depths.extend(dependency_depths(sent))
        sentence_branching.extend(branching_factors(sent))
        sentence_subtree_sizes.extend(subtree_sizes(sent))
        motif_counter.update(
            motif_counts(sent, mode=mode, literal_words=literal_words, min_leaves=min_leaves)
        )
        clause_counter.update(clause_counts(sent))

    depth_summary = summarize_distribution(sentence_depths)
    branching_summary = summarize_distribution(sentence_branching)
    subtree_summary = summarize_distribution(sentence_subtree_sizes)

    row: dict[str, object] = {
        "text_id": text_id,
        "chunk_id": sentence_id,
        "text": doc.text,
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
        "motif_counts": dict(motif_counter),
    }

    for dep in sorted(CLAUSE_DEPS):
        row[f"clause_{dep}_count"] = clause_counter.get(dep, 0)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute syntax features for local Gutenberg texts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python -m src.experiments.gutenberg_syntax_features --glob "*Stevenson*.txt"
            """
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=LOCAL_GUTENBERG_DIR)
    parser.add_argument("--files", nargs="+", type=Path, default=None)
    parser.add_argument("--glob", default=None)
    parser.add_argument("--list", action="store_true", help="List resolved input paths and exit.")
    parser.add_argument(
        "--chunk-mode",
        choices=["paragraphs", "window", "sentences"],
        default="paragraphs",
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--motif-mode", choices=["category", "mixed"], default="category")
    parser.add_argument("--motif-min-leaves", type=int, default=3)
    parser.add_argument("--literal-words", default="")
    parser.add_argument("--literal-words-file", default=None)
    parser.add_argument("--sentence-cache", type=Path, default=SENTENCE_CACHE_PATH)
    parser.add_argument("--write-sentence-cache", action="store_true")
    parser.add_argument("--use-sentence-cache", action="store_true")
    parser.add_argument("--overwrite-sentence-cache", action="store_true")
    parser.add_argument("--cache-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if args.cache_summary:
        try:
            summary = summarize_sentence_cache(args.sentence_cache)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Sentence cache: {args.sentence_cache.as_posix()}")
        print(f"Total sentences: {summary['sentence_total']}")
        print(f"Total texts: {summary['text_total']}")
        print("Sentences by author:")
        for author, count in summary["authors"].most_common():
            print(f"  {author}: {count}")
        print("Sentences by title:")
        for title, count in summary["titles"].most_common():
            print(f"  {title}: {count}")
        return
    try:
        input_paths = [] if args.use_sentence_cache else resolve_input_paths(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.list:
        for path in input_paths:
            print(path.as_posix())
        return
    if args.use_sentence_cache and args.chunk_mode != "sentences":
        raise SystemExit("--use-sentence-cache requires --chunk-mode sentences.")
    if args.use_sentence_cache and args.max_sentences:
        raise SystemExit("Use --max-sentences when creating the cache, not when reading it.")
    if args.write_sentence_cache:
        try:
            prepare_sentence_cache_path(args.sentence_cache, args.overwrite_sentence_cache)
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from exc

    if args.motif_min_leaves < 3:
        raise SystemExit("--motif-min-leaves must be >= 3 to ensure motifs have at least four nodes.")

    nlp = spacy.load(args.spacy_model)
    literal_words = load_literal_words(args)

    metadata_rows = []
    feature_rows: list[dict[str, object]] = []

    if args.use_sentence_cache:
        metadata_by_id: dict[int, dict[str, object]] = {}
        for entry in iter_sentence_cache(args.sentence_cache, nlp):
            text_id = int(entry["text_id"])
            metadata_by_id.setdefault(
                text_id,
                {
                    "id": text_id,
                    "title": entry.get("title", "Unknown"),
                    "author": entry.get("author", "Unknown"),
                    "path": entry.get("source_path", ""),
                },
            )
            feature_rows.append(
                compute_features_from_sentence_doc(
                    text_id,
                    int(entry["sentence_id"]),
                    entry["doc"],
                    mode=args.motif_mode,
                    literal_words=literal_words,
                    min_leaves=args.motif_min_leaves,
                )
            )
        metadata_rows.extend(metadata_by_id.values())
    else:
        cache_handle = None
        if args.write_sentence_cache:
            cache_handle = args.sentence_cache.open("w", encoding="utf-8")
        try:
            for gutenberg_text in iter_texts(input_paths):
                cleaned = normalize_text(strip_header_footer(gutenberg_text.raw_text))
                if args.max_sentences:
                    cleaned = limit_text_to_sentences(nlp, cleaned, args.max_sentences)
                if args.chunk_mode == "sentences":
                    for sentence_id, sent_doc in enumerate(iter_sentence_docs(nlp, cleaned), start=1):
                        if cache_handle:
                            write_sentence_cache_entry(
                                cache_handle,
                                {
                                    "text_id": gutenberg_text.text_id,
                                    "title": gutenberg_text.title,
                                    "author": gutenberg_text.author,
                                    "source_path": gutenberg_text.source_path.as_posix(),
                                    "sentence_id": sentence_id,
                                    "sentence_text": sent_doc.text,
                                    "doc_json": sent_doc.to_json(),
                                },
                            )
                        feature_rows.append(
                            compute_features_from_sentence_doc(
                                gutenberg_text.text_id,
                                sentence_id,
                                sent_doc,
                                mode=args.motif_mode,
                                literal_words=literal_words,
                                min_leaves=args.motif_min_leaves,
                            )
                        )
                else:
                    chunks = iter_chunks(cleaned, args.chunk_mode, args.window_size)
                    feature_rows.extend(
                        compute_features(
                            nlp,
                            gutenberg_text.text_id,
                            chunks,
                            motif_mode=args.motif_mode,
                            literal_words=literal_words,
                            motif_min_leaves=args.motif_min_leaves,
                        )
                    )
                metadata_rows.append(
                    {
                        "id": gutenberg_text.text_id,
                        "title": gutenberg_text.title,
                        "author": gutenberg_text.author,
                        "path": gutenberg_text.source_path.as_posix(),
                    }
                )
        finally:
            if cache_handle:
                cache_handle.close()

    ensure_metadata_header(METADATA_PATH)
    with METADATA_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "author", "path"])
        for row in metadata_rows:
            writer.writerow(row)

    for row in feature_rows:
        motif_counts_dict = row.pop("motif_counts", {})
        for motif_name, count in motif_counts_dict.items():
            row[f"motif_{motif_name}_count"] = count

    df = pd.DataFrame(feature_rows)
    motif_cols = [col for col in df.columns if col.startswith("motif_")]
    if motif_cols:
        df[motif_cols] = df[motif_cols].fillna(0).astype(int)
    df.to_parquet(FEATURES_PATH, index=False)


if __name__ == "__main__":
    main()
