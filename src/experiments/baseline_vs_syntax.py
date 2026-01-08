"""Compare baseline stylometry features with syntactic features."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

FUNCTION_WORDS = [
    "the",
    "and",
    "of",
    "to",
    "a",
    "in",
    "that",
    "is",
    "it",
    "was",
    "i",
    "for",
    "on",
    "you",
    "with",
    "as",
    "have",
    "be",
    "at",
    "or",
    "not",
    "this",
    "but",
    "they",
    "his",
    "her",
    "she",
    "he",
    "we",
    "their",
    "my",
    "me",
    "so",
    "if",
    "there",
    "what",
    "which",
    "who",
    "whom",
    "been",
    "when",
    "were",
    "would",
    "do",
    "did",
    "had",
    "can",
    "could",
    "should",
    "will",
    "just",
    "than",
    "then",
    "them",
]

SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+")
WORD_REGEX = re.compile(r"\b\w+\b")


class FunctionWordFrequency(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary: Iterable[str]):
        self.vocabulary = list(vocabulary)
        self.vectorizer = CountVectorizer(vocabulary=self.vocabulary, lowercase=True)

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X: Iterable[str]):
        counts = self.vectorizer.transform(X)
        total_words = np.array([len(WORD_REGEX.findall(text)) for text in X], dtype=float)
        total_words[total_words == 0] = 1.0
        return counts.multiply(1.0 / total_words[:, None])


class SentenceStats(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None):
        return self

    def transform(self, X: Iterable[str]):
        stats = [self._sentence_stats(text) for text in X]
        return csr_matrix(np.array(stats, dtype=float))

    @staticmethod
    def _sentence_stats(text: str):
        sentences = [s for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
        if not sentences:
            return [0.0, 0.0, 0.0, 0.0]
        lengths = [len(WORD_REGEX.findall(sentence)) for sentence in sentences]
        lengths = np.array(lengths, dtype=float)
        return [lengths.mean(), lengths.std(ddof=0), lengths.min(), lengths.max()]


@dataclass
class FeatureSet:
    name: str
    transformer: Optional[BaseEstimator]
    uses_syntax: bool


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the parquet file.")
    return df


def detect_label_columns(df: pd.DataFrame, requested: Optional[str]) -> List[str]:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Label column '{requested}' not found.")
        return [requested]

    labels: List[str] = []
    if "author" in df.columns:
        labels.append("author")

    candidate_markers = ["machine", "ai", "generated", "synthetic", "human", "label"]
    for col in df.columns:
        if col in labels or col == "text":
            continue
        lower = col.lower()
        if any(marker in lower for marker in candidate_markers):
            if df[col].nunique() <= 2:
                labels.append(col)

    return labels


def syntax_feature_columns(df: pd.DataFrame, label_columns: List[str]) -> List[str]:
    exclude = set(label_columns + ["text"])
    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols


def build_baseline_transformer() -> BaseEstimator:
    return FeatureUnionWrapper(
        [
            ("function_words", FunctionWordFrequency(FUNCTION_WORDS)),
            (
                "char_ngrams",
                TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2),
            ),
            ("sentence_stats", SentenceStats()),
        ]
    )


class FeatureUnionWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformers: List[tuple[str, BaseEstimator]]):
        self.transformers = transformers

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None):
        for _, transformer in self.transformers:
            transformer.fit(X, y)
        return self

    def transform(self, X: Iterable[str]):
        matrices = [transformer.transform(X) for _, transformer in self.transformers]
        return hstack(matrices)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    syntax_cols: List[str],
    feature_set: FeatureSet,
):
    baseline_train = baseline_test = None
    if feature_set.transformer is not None:
        baseline_train = feature_set.transformer.fit_transform(train_df[text_column])
        baseline_test = feature_set.transformer.transform(test_df[text_column])

    syntax_train = syntax_test = None
    if feature_set.uses_syntax:
        scaler = StandardScaler()
        syntax_train = scaler.fit_transform(train_df[syntax_cols].fillna(0.0))
        syntax_test = scaler.transform(test_df[syntax_cols].fillna(0.0))

    if feature_set.name == "baseline":
        return baseline_train, baseline_test
    if feature_set.name == "syntax":
        return csr_matrix(syntax_train), csr_matrix(syntax_test)
    if feature_set.name == "combined":
        return hstack([baseline_train, csr_matrix(syntax_train)]), hstack(
            [baseline_test, csr_matrix(syntax_test)]
        )

    raise ValueError(f"Unknown feature set {feature_set.name}")


def build_classifiers(random_state: int = 13):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "linear_svm": LinearSVC(),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
    }


def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name: str,
    classifier,
):
    if model_name == "random_forest":
        svd = TruncatedSVD(n_components=min(200, X_train.shape[1] - 1))
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        classifier.fit(X_train, y_train)
    else:
        classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
    }


def plot_results(results: pd.DataFrame, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for label in results["label"].unique():
        label_df = results[results["label"] == label]
        for metric in ["accuracy", "f1_macro"]:
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot = label_df.pivot_table(
                index="model", columns="feature_set", values=metric
            )
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{label}: {metric}")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.legend(title="Feature set", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            path = output_dir / f"{label}_{metric}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


def write_report(
    results: pd.DataFrame,
    output_path: Path,
    plots: List[Path],
    metadata: dict,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = ["# Baseline vs. Syntax Feature Evaluation", ""]
    report_lines.append("## Overview")
    report_lines.append(
        "This report compares baseline stylometry features (function words, character "
        "n-grams, and sentence length stats) against precomputed syntactic features and "
        "their combination using simple classifiers."
    )
    report_lines.append("")
    report_lines.append("## Dataset")
    report_lines.append("```json")
    report_lines.append(json.dumps(metadata, indent=2))
    report_lines.append("```")
    report_lines.append("")

    for label in results["label"].unique():
        report_lines.append(f"## Target: {label}")
        label_df = results[results["label"] == label]
        report_lines.append(
            label_df.sort_values(["model", "feature_set"])
            .to_markdown(index=False, floatfmt=".3f")
        )
        report_lines.append("")

    if plots:
        report_lines.append("## Plots")
        for plot in plots:
            relative = plot.as_posix()
            report_lines.append(f"![{plot.stem}]({relative})")
        report_lines.append("")

    output_path.write_text("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/gutenberg_syntax_features.parquet"),
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/baseline_vs_syntax.md"),
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("reports/figures"),
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    label_columns = detect_label_columns(df, args.label)
    if not label_columns:
        raise ValueError("No label columns detected. Use --label to specify one.")

    syntax_cols = syntax_feature_columns(df, label_columns)
    if not syntax_cols:
        raise ValueError("No numeric syntax feature columns found.")

    feature_sets = [
        FeatureSet("baseline", build_baseline_transformer(), uses_syntax=False),
        FeatureSet("syntax", None, uses_syntax=True),
        FeatureSet("combined", build_baseline_transformer(), uses_syntax=True),
    ]

    classifiers = build_classifiers(random_state=args.random_state)

    results = []
    for label in label_columns:
        y = df[label]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        train_df, test_df, y_train, y_test = train_test_split(
            df,
            y_encoded,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_encoded,
        )

        for feature_set in feature_sets:
            X_train, X_test = prepare_features(
                train_df, test_df, "text", syntax_cols, feature_set
            )
            for model_name, classifier in classifiers.items():
                metrics = train_and_evaluate(
                    X_train, X_test, y_train, y_test, model_name, classifier
                )
                results.append(
                    {
                        "label": label,
                        "feature_set": feature_set.name,
                        "model": model_name,
                        **metrics,
                    }
                )

    results_df = pd.DataFrame(results)
    plots = plot_results(results_df, args.figures)
    metadata = {
        "data_path": str(args.data),
        "samples": len(df),
        "labels": label_columns,
        "syntax_features": len(syntax_cols),
    }
    write_report(results_df, args.report, plots, metadata)


if __name__ == "__main__":
    main()
