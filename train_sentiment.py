#!/usr/bin/env python3
"""Train an ``ml_core`` sentiment model from a labelled CSV.

CSV format:
    text,sentiment
    "I loved the film, the story was wonderful",positive
    "I hated this boring and awful film",negative

Usage:
    python train_sentiment.py labelled_reviews.csv
    python train_sentiment.py labelled_reviews.csv --features 8000
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ML_PROJECT_ROOT = PROJECT_ROOT.parent / "machineLearning"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.bin"
VOCABULARY_PATH = MODEL_DIR / "sentiment_vocabulary.json"
TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)
LABELS = {
    "0": 0,
    "negative": 0,
    "negativo": 0,
    "1": 1,
    "positive": 1,
    "positivo": 1,
}


def load_ml_core() -> ModuleType:
    """Load ml_core from the environment or the sibling machineLearning build."""
    try:
        return importlib.import_module("ml_core")
    except ModuleNotFoundError as error:
        if error.name != "ml_core":
            raise

    for build_dir in (
        ML_PROJECT_ROOT / "build" / "pybinding",
        ML_PROJECT_ROOT / "build" / "lib",
    ):
        if build_dir.is_dir():
            sys.path.insert(0, str(build_dir))
    return importlib.import_module("ml_core")


def extract_terms(text: str) -> list[str]:
    """Extract lowercase words and adjacent bigrams."""
    words = TOKEN_PATTERN.findall(text.lower())
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]


def read_dataset(path: Path) -> tuple[list[str], np.ndarray]:
    """Read and validate a CSV containing text and sentiment columns."""
    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"text", "sentiment"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have CSV headers 'text,sentiment'")

        for line, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            sentiment = (row.get("sentiment") or "").strip().lower()
            if not text:
                raise ValueError(f"empty review at CSV line {line}")
            if sentiment not in LABELS:
                raise ValueError(
                    f"invalid sentiment {sentiment!r} at line {line}; "
                    "use positive/negative or 1/0"
                )
            texts.append(text)
            labels.append(LABELS[sentiment])

    y = np.asarray(labels, dtype=np.float64)
    class_counts = np.bincount(y.astype(np.int64), minlength=2)
    if class_counts.min() < 2:
        raise ValueError("provide at least two positive and two negative reviews")
    return texts, y


def make_vocabulary(texts: list[str], max_features: int) -> dict[str, int]:
    terms: Counter[str] = Counter()
    for text in texts:
        terms.update(extract_terms(text))
    if not terms:
        raise ValueError("reviews contain no usable words")
    return {
        term: index
        for index, (term, _) in enumerate(terms.most_common(max_features))
    }


def vectorize(texts: list[str], vocabulary: dict[str, int], width: int) -> np.ndarray:
    features = np.zeros((len(texts), width), dtype=np.float64, order="F")
    for row, text in enumerate(texts):
        for term in extract_terms(text):
            index = vocabulary.get(term)
            if index is not None:
                features[row, index] += 1.0
    return features


def stratified_holdout(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test indices while keeping examples from both classes."""
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in (0.0, 1.0):
        indices = np.flatnonzero(labels == label)
        if indices.size < 2:
            raise ValueError("need at least two examples of each sentiment")
        rng.shuffle(indices)
        test_count = max(1, round(indices.size * 0.2))
        test_count = min(test_count, indices.size - 1)
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.asarray(train_indices), np.asarray(test_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", type=Path, help="CSV con colonne text,sentiment")
    parser.add_argument("--features", type=int, default=5000, help="dimensione massima del vocabolario")
    parser.add_argument("--seed", type=int, default=42, help="seed per la valutazione holdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.features < 1:
        raise ValueError("--features must be greater than zero")
    if not args.training_csv.is_file():
        raise FileNotFoundError(f"training CSV not found: {args.training_csv}")

    texts, labels = read_dataset(args.training_csv)
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    print(f"Read {len(texts)} reviews: {counts[1]} positive, {counts[0]} negative.")

    train_indices, test_indices = stratified_holdout(labels, args.seed)
    evaluation_vocabulary = make_vocabulary(
        [texts[index] for index in train_indices],
        args.features,
    )
    evaluation_model = load_ml_core().LogisticRegression(
        learning_rate=0.1,
        max_iter=500,
        verbose=False,
    )
    evaluation_model.fit(
        vectorize([texts[index] for index in train_indices], evaluation_vocabulary, args.features),
        labels[train_indices],
    )
    accuracy = float(
        evaluation_model.score(
            vectorize([texts[index] for index in test_indices], evaluation_vocabulary, args.features),
            labels[test_indices],
        )
    )
    print(f"Holdout accuracy (20%): {accuracy:.1%}")

    vocabulary = make_vocabulary(texts, args.features)
    final_model = load_ml_core().LogisticRegression(
        learning_rate=0.1,
        max_iter=500,
        verbose=False,
    )
    final_model.fit(vectorize(texts, vocabulary, args.features), labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save(str(MODEL_PATH))
    metadata = {
        "format_version": 1,
        "max_features": args.features,
        "vocabulary": vocabulary,
    }
    temporary_path = VOCABULARY_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(VOCABULARY_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vocabulary: {VOCABULARY_PATH}")
    print("Training complete. You can now run predict_sentiment.py.")


if __name__ == "__main__":
    main()
