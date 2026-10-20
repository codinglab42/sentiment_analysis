#!/usr/bin/env python3
"""Load a trained ml_core model and summarize people's opinions.

Usage:
    python predict_sentiment.py

First train the model with ``train_sentiment.py`` using a labelled CSV.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ML_PROJECT_ROOT = PROJECT_ROOT.parent / "machineLearning"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.bin"
VOCABULARY_PATH = PROJECT_ROOT / "models" / "sentiment_vocabulary.json"
TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


def load_ml_core() -> ModuleType:
    """Load ml_core from the environment or the sibling machineLearning build."""
    try:
        return importlib.import_module("ml_core")
    except ModuleNotFoundError as error:
        if error.name != "ml_core":
            raise

    build_dirs = [
        ML_PROJECT_ROOT / "src",
        ML_PROJECT_ROOT / "build" / "pybinding",
        ML_PROJECT_ROOT / "build" / "lib",
    ]
    build_root = ML_PROJECT_ROOT / "build"
    if build_root.is_dir():
        build_dirs.extend(sorted(build_root.glob("lib.*")))

    for build_dir in build_dirs:
        if build_dir.is_dir():
            build_path = str(build_dir)
            if build_path not in sys.path:
                sys.path.insert(0, build_path)
    try:
        return importlib.import_module("ml_core")
    except ModuleNotFoundError as error:
        if error.name != "ml_core":
            raise
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(
                "ml_core requires Python 3.11 (machineLearning supports "
                ">=3.11,<3.12); run this project with Python 3.11."
            ) from error
        raise ModuleNotFoundError(
            "ml_core is not installed and no compatible sibling build was found; "
            "build or install the machineLearning project first."
        ) from error


def extract_terms(text: str) -> list[str]:
    words = TOKEN_PATTERN.findall(text.lower())
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]


def load_classifier() -> tuple[object, dict[str, int], int]:
    if not MODEL_PATH.is_file() or not VOCABULARY_PATH.is_file():
        raise FileNotFoundError(
            "trained model not found; run train_sentiment.py with a labelled CSV first"
        )

    metadata = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError(f"unsupported vocabulary format: {metadata.get('format_version')!r}")
    width = metadata.get("max_features")
    vocabulary = metadata.get("vocabulary")
    if (
        not isinstance(width, int)
        or width < 1
        or not isinstance(vocabulary, dict)
        or not vocabulary
        or any(not isinstance(index, int) or not 0 <= index < width for index in vocabulary.values())
    ):
        raise ValueError("saved vocabulary is malformed; retrain the model")

    model = load_ml_core().LogisticRegression()
    model.load(str(MODEL_PATH))
    return model, vocabulary, width


def predict(
    model: object,
    vocabulary: dict[str, int],
    width: int,
    text: str,
) -> tuple[float, float]:
    terms = extract_terms(text)
    if not terms:
        raise ValueError("enter a review containing words")

    features = np.zeros((1, width), dtype=np.float64, order="F")
    known_terms = 0
    for term in terms:
        index = vocabulary.get(term)
        if index is not None:
            features[0, index] += 1.0
            known_terms += 1
    if not known_terms:
        raise ValueError(
            "the review has no words in the training vocabulary; "
            "collect more representative training examples"
        )

    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64).reshape(-1)
    if probabilities.size != 1 or not np.isfinite(probabilities[0]):
        raise RuntimeError("ml_core returned an invalid prediction")
    return float(probabilities[0]), known_terms / len(terms)


def main() -> None:
    model, vocabulary, width = load_classifier()
    topic = input("Quale film o argomento vuoi valutare? ").strip()
    if not topic:
        raise ValueError("enter a film title or topic")

    positive = 0
    negative = 0
    skipped = 0
    probability_sum = 0.0
    print(f"\nOpinions about: {topic}")
    print("Enter one person's opinion at a time; submit an empty line to finish.")

    while True:
        try:
            text = input("\nOpinion> ").strip()
        except EOFError:
            print()
            break
        if not text:
            break
        try:
            probability, coverage = predict(model, vocabulary, width, text)
        except ValueError as error:
            print(f"Not classified: {error}")
            skipped += 1
            continue

        label = "LIKES IT" if probability >= 0.5 else "DOES NOT LIKE IT"
        confidence = max(probability, 1.0 - probability)
        print(
            f"{label} — positive probability {probability:.1%}, "
            f"confidence {confidence:.1%}, known terms {coverage:.1%}"
        )
        if probability >= 0.5:
            positive += 1
        else:
            negative += 1
        probability_sum += probability

    total = positive + negative
    print("\n--- Summary ---")
    print(f"Classified opinions: {total}")
    if total:
        print(f"Likes it:         {positive} ({positive / total:.1%})")
        print(f"Does not like it: {negative} ({negative / total:.1%})")
        print(f"Estimated average approval: {probability_sum / total:.1%}")
    if skipped:
        print(f"Not classified: {skipped}")
    if not total:
        print("No classifiable opinions; approval cannot be estimated.")
    print("Percentages describe only opinions entered, not the whole population.")


if __name__ == "__main__":
    main()
