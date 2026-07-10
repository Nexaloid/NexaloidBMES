from __future__ import annotations


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
        self.totals[key] = self.totals.get(key, 0.0) + (
            self.step - self.timestamps.get(key, 0)
        ) * current
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
                total = self.totals.get(key, 0.0) + (
                    self.step - self.timestamps.get(key, 0)
                ) * current
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
