"""Create Python CRFsuite predictions for browser parity validation."""

import argparse
import json
from pathlib import Path
import sys

import pycrfsuite
from conllu import parse_incr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import sent2features, tokenize


def build_cases(model_path: Path, conllu_path: Path, output_path: Path):
    tagger = pycrfsuite.Tagger()
    tagger.open(str(model_path))
    cases = []

    with conllu_path.open("r", encoding="utf-8") as source:
        for sentence in parse_incr(source):
            corpus_tokens = [
                token["form"]
                for token in sentence
                if isinstance(token["id"], int)
                and token["upostag"] not in (None, "_")
            ]
            sentence_text = sentence.metadata.get("text", " ".join(corpus_tokens))
            tokens = tokenize(sentence_text)
            cases.append({
                "sentence": sentence_text,
                "tokens": tokens,
                "tags": tagger.tag(sent2features(tokens)),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(cases, output, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {len(cases)} Python prediction cases to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("conllu_path", type=Path)
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()
    build_cases(arguments.model_path, arguments.conllu_path, arguments.output_path)
