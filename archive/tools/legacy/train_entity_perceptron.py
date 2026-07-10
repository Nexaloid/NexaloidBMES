from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from train_entity_hmm import (
    ALLOWED_NEXT,
    FINAL_STATES,
    START_STATES,
    STATES,
    iter_cases,
    spans_to_tags,
    tags_to_spans,
)


START = "<START>"


def char_at(text: str, index: int) -> str:
    if index < 0:
        return f"<BOS{-index}>"
    if index >= len(text):
        return f"<EOS{index - len(text) + 1}>"
    return text[index]


def char_class(char: str) -> str:
    if char.startswith("<"):
        return char
    codepoint = ord(char)
    if 0x3400 <= codepoint <= 0x9FFF or 0x20000 <= codepoint <= 0x2EBEF:
        return "HAN"
    if char.isdigit():
        return "DIGIT"
    if "A" <= char <= "Z" or "a" <= char <= "z":
        return "LATIN"
    if char.isspace():
        return "SPACE"
    if not char.isalnum():
        return "PUNCT"
    return "OTHER"


def features_at(text: str, index: int) -> tuple[str, ...]:
    c2l = char_at(text, index - 2)
    c1l = char_at(text, index - 1)
    c0 = char_at(text, index)
    c1r = char_at(text, index + 1)
    c2r = char_at(text, index + 2)
    k1l = char_class(c1l)
    k0 = char_class(c0)
    k1r = char_class(c1r)
    return (
        "bias",
        f"c0={c0}",
        f"c-1={c1l}",
        f"c+1={c1r}",
        f"c-2={c2l}",
        f"c+2={c2r}",
        f"c-1c0={c1l}{c0}",
        f"c0c+1={c0}{c1r}",
        f"c-2c-1c0={c2l}{c1l}{c0}",
        f"c0c+1c+2={c0}{c1r}{c2r}",
        f"k0={k0}",
        f"k-1={k1l}",
        f"k+1={k1r}",
        f"k-1k0={k1l}:{k0}",
        f"k0k+1={k0}:{k1r}",
    )


def emission_scores(weights: dict[str, dict[str, float]], features: tuple[str, ...]) -> dict[str, float]:
    scores = {state: 0.0 for state in STATES}
    for feature in features:
        for state, value in weights.get(feature, {}).items():
            scores[state] += value
    return scores


def decode_tags_with_features(
    weights: dict[str, dict[str, float]],
    sentence_features: list[tuple[str, ...]],
) -> list[str]:
    if not sentence_features:
        return []

    first_emission = emission_scores(weights, sentence_features[0])
    transition = weights.get(f"T={START}", {})
    scores = {
        state: first_emission[state] + transition.get(state, 0.0)
        for state in START_STATES
    }
    paths = {state: [state] for state in scores}

    for position in range(1, len(sentence_features)):
        current_emission = emission_scores(weights, sentence_features[position])
        next_scores: dict[str, float] = {}
        next_paths: dict[str, list[str]] = {}
        for previous, base_score in scores.items():
            transition = weights.get(f"T={previous}", {})
            for current in ALLOWED_NEXT[previous]:
                candidate = base_score + transition.get(current, 0.0) + current_emission[current]
                if current not in next_scores or candidate > next_scores[current]:
                    next_scores[current] = candidate
                    next_paths[current] = paths[previous] + [current]
        scores = next_scores
        paths = next_paths

    finals = [state for state in FINAL_STATES if state in scores]
    if not finals:
        return ["O"] * len(sentence_features)
    best = max(finals, key=lambda state: scores[state])
    return paths[best]


def decode_tags(model: dict, text: str) -> list[str]:
    return decode_tags_with_features(model["weights"], [features_at(text, index) for index in range(len(text))])


def decode(model: dict, text: str) -> list[tuple[int, int, str]]:
    return tags_to_spans(decode_tags(model, text))


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class AveragedWeights:
    def __init__(self) -> None:
        self.weights: dict[str, dict[str, float]] = {}
        self.totals: dict[tuple[str, str], float] = {}
        self.timestamps: dict[tuple[str, str], int] = {}
        self.step = 0

    def tick(self) -> None:
        self.step += 1

    def update(self, feature: str, state: str, delta: float) -> None:
        row = self.weights.setdefault(feature, {})
        key = (feature, state)
        current = row.get(state, 0.0)
        self.totals[key] = self.totals.get(key, 0.0) + (self.step - self.timestamps.get(key, 0)) * current
        self.timestamps[key] = self.step
        row[state] = current + delta

    def average(self, min_abs: float) -> dict[str, dict[str, float]]:
        averaged: dict[str, dict[str, float]] = {}
        if self.step == 0:
            return averaged
        for feature, row in self.weights.items():
            out: dict[str, float] = {}
            for state, current in row.items():
                key = (feature, state)
                total = self.totals.get(key, 0.0) + (self.step - self.timestamps.get(key, 0)) * current
                value = total / self.step
                if abs(value) >= min_abs:
                    out[state] = round(value, 6)
            if out:
                averaged[feature] = out
        return dict(sorted(averaged.items()))


def update_sequence(
    model: AveragedWeights,
    sentence_features: list[tuple[str, ...]],
    gold: list[str],
    predicted: list[str],
) -> None:
    for index, features in enumerate(sentence_features):
        gold_previous = START if index == 0 else gold[index - 1]
        predicted_previous = START if index == 0 else predicted[index - 1]
        if gold[index] != predicted[index]:
            for feature in features:
                model.update(feature, gold[index], 1.0)
                model.update(feature, predicted[index], -1.0)
        if gold_previous != predicted_previous or gold[index] != predicted[index]:
            model.update(f"T={gold_previous}", gold[index], 1.0)
            model.update(f"T={predicted_previous}", predicted[index], -1.0)


def train(path: Path, epochs: int, seed: int, min_abs: float) -> dict:
    cases = [
        (text, spans_to_tags(text, spans), [features_at(text, index) for index in range(len(text))])
        for text, spans in iter_cases(path)
    ]
    weights = AveragedWeights()
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(cases)
        mistakes = 0
        for _, gold, sentence_features in cases:
            weights.tick()
            predicted = decode_tags_with_features(weights.weights, sentence_features)
            if predicted != gold:
                mistakes += 1
                update_sequence(weights, sentence_features, gold, predicted)
        print(f"epoch\t{epoch + 1}\tmistakes\t{mistakes}/{len(cases)}")

    averaged = weights.average(min_abs)
    return {
        "schema": "nexaloid.entity_perceptron.v1",
        "states": list(STATES),
        "start_states": list(START_STATES),
        "final_states": list(FINAL_STATES),
        "allowed_next": {state: list(values) for state, values in ALLOWED_NEXT.items()},
        "epochs": epochs,
        "seed": seed,
        "sentence_count": len(cases),
        "feature_count": len(averaged),
        "min_abs": min_abs,
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feature_templates": [
            "bias",
            "characters[-2..+2]",
            "character_bigrams",
            "character_trigrams",
            "character_classes[-1..+1]",
            "previous_tag_transition",
        ],
        "weights": averaged,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/tasks/entity_hmm/train.tsv"))
    parser.add_argument("--out", type=Path, default=Path("data/tasks/entity_hmm/entity_perceptron_wordhub.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-abs", type=float, default=0.05)
    args = parser.parse_args()

    model = train(args.train, args.epochs, args.seed, args.min_abs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(model, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"features\t{model['feature_count']}")
    print(f"wrote\t{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
