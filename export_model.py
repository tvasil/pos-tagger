"""Export a native CRFsuite model to the static browser model schema."""

import argparse
import json
from pathlib import Path

import pycrfsuite

from helpers import sent2features, tokenize


FORMAT_VERSION = 1
VALIDATION_SENTENCES = [
    "I can fish.",
    "A can of fish.",
    "Time flies like an arrow.",
    "Fruit flies like a banana.",
    "The quick brown fox jumps over the lazy dog.",
    "Book that flight.",
    "Can you book a flight?",
    "NASA launched Apollo 11.",
    "State-of-the-art tools work.",
]


def export_model(
    model_path: Path,
    output_path: Path,
    validation_path: Path,
    corpus_name=None,
    corpus_revision=None,
    corpus_license=None,
    test_token_accuracy=None,
):
    tagger = pycrfsuite.Tagger()
    tagger.open(str(model_path))
    info = tagger.info()
    labels = tagger.labels()
    label_indexes = {label: index for index, label in enumerate(labels)}

    state_features = {}
    for (attribute, label), weight in info.state_features.items():
        state_features.setdefault(attribute, []).extend(
            (label_indexes[label], weight)
        )

    transition_count = len(labels) * len(labels)
    transitions = [0.0] * transition_count
    for (from_label, to_label), weight in info.transitions.items():
        from_index = label_indexes[from_label]
        to_index = label_indexes[to_label]
        transitions[from_index * len(labels) + to_index] = weight

    metadata = {
        "stateFeatureCount": len(info.state_features),
        "attributeCount": len(info.attributes),
        "transitionCount": len(info.transitions),
        "sourceModelBytes": model_path.stat().st_size,
        "trainingAlgorithm": "l2sgd",
        "confidenceScores": False,
    }
    if corpus_name:
        metadata["corpus"] = corpus_name
    if corpus_revision:
        metadata["corpusRevision"] = corpus_revision
    if corpus_license:
        metadata["corpusLicense"] = corpus_license
    if test_token_accuracy is not None:
        metadata["testTokenAccuracy"] = test_token_accuracy

    artifact = {
        "formatVersion": FORMAT_VERSION,
        "modelType": "linear-chain-crf",
        "featureEncoding": "python-crfsuite-item-sequence-v1",
        "tokenizer": "simple-unicode-v1",
        "labels": labels,
        "stateFeatures": state_features,
        "transitions": transitions,
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(artifact, output_file, ensure_ascii=False, separators=(",", ":"))

    cases = []
    for sentence in VALIDATION_SENTENCES:
        tokens = tokenize(sentence)
        tags = tagger.tag(sent2features(tokens))
        cases.append({"sentence": sentence, "tokens": tokens, "tags": tags})

    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("w", encoding="utf-8") as validation_file:
        json.dump(cases, validation_file, ensure_ascii=False, indent=2)
        validation_file.write("\n")

    print(
        f"Exported {len(labels)} labels, {len(info.state_features)} state "
        f"features, and {len(info.transitions)} transitions to {output_path}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a native CRFsuite model for browser inference"
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("tests/browser_validation_cases.json"),
    )
    parser.add_argument("--corpus-name")
    parser.add_argument("--corpus-revision")
    parser.add_argument("--corpus-license")
    parser.add_argument("--test-token-accuracy", type=float)
    arguments = parser.parse_args()
    export_model(
        arguments.model_path,
        arguments.output_path,
        arguments.validation_output,
        arguments.corpus_name,
        arguments.corpus_revision,
        arguments.corpus_license,
        arguments.test_token_accuracy,
    )
