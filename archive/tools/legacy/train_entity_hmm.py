from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ENTITY_TYPES = ("PER", "LOC", "ORG")
STATES = ("O",) + tuple(
    f"{prefix}_{kind}"
    for kind in ENTITY_TYPES
    for prefix in ("B", "M", "E", "S")
)
START_STATES = ("O",) + tuple(
    f"{prefix}_{kind}"
    for kind in ENTITY_TYPES
    for prefix in ("B", "S")
)
FINAL_STATES = ("O",) + tuple(
    f"{prefix}_{kind}"
    for kind in ENTITY_TYPES
    for prefix in ("E", "S")
)


def allowed_next() -> dict[str, tuple[str, ...]]:
    entity_starts = tuple(
        f"{prefix}_{kind}"
        for kind in ENTITY_TYPES
        for prefix in ("B", "S")
    )
    after_entity = ("O",) + entity_starts
    out = {"O": after_entity}
    for kind in ENTITY_TYPES:
        out[f"B_{kind}"] = (f"M_{kind}", f"E_{kind}")
        out[f"M_{kind}"] = (f"M_{kind}", f"E_{kind}")
        out[f"E_{kind}"] = after_entity
        out[f"S_{kind}"] = after_entity
    return out


ALLOWED_NEXT = allowed_next()


def parse_spans(text: str, raw: str) -> list[tuple[int, int, str]]:
    if raw == "-":
        return []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for item in raw.split(","):
        start_text, end_text, kind = item.split(":", 2)
        start = int(start_text)
        end = int(end_text)
        if kind not in ENTITY_TYPES or start < cursor or start >= end or end > len(text):
            raise ValueError(f"invalid span {item!r} for {text!r}")
        spans.append((start, end, kind))
        cursor = end
    return spans


def iter_cases(path: Path):
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            text, span_text = line.split("\t", 1)
            spans = parse_spans(text, span_text)
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        yield text, spans


def entity_tags(kind: str, length: int) -> list[str]:
    if length == 1:
        return [f"S_{kind}"]
    if length == 2:
        return [f"B_{kind}", f"E_{kind}"]
    return [f"B_{kind}"] + [f"M_{kind}"] * (length - 2) + [f"E_{kind}"]


def spans_to_tags(text: str, spans: list[tuple[int, int, str]]) -> list[str]:
    tags = ["O"] * len(text)
    for start, end, kind in spans:
        tags[start:end] = entity_tags(kind, end - start)
    return tags


def log_probs(counter: Counter[str], keys: tuple[str, ...], smoothing: float) -> dict[str, float]:
    total = sum(counter.values()) + smoothing * len(keys)
    return {key: math.log((counter[key] + smoothing) / total) for key in keys}


def train(path: Path, smoothing: float) -> dict:
    start_counts: Counter[str] = Counter()
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    emission_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vocab: set[str] = set()
    sentence_count = entity_count = 0

    for text, spans in iter_cases(path):
        if not text:
            continue
        tags = spans_to_tags(text, spans)
        sentence_count += 1
        entity_count += len(spans)
        start_counts[tags[0]] += 1
        for char, tag in zip(text, tags):
            emission_counts[tag][char] += 1
            vocab.add(char)
        for previous, current in zip(tags, tags[1:]):
            if current not in ALLOWED_NEXT[previous]:
                raise ValueError(f"illegal transition {previous}->{current} in {text!r}")
            transition_counts[previous][current] += 1

    emissions: dict[str, dict[str, float]] = {}
    unknown: dict[str, float] = {}
    for state in STATES:
        total = sum(emission_counts[state].values()) + smoothing * (len(vocab) + 1)
        emissions[state] = {
            char: math.log((count + smoothing) / total)
            for char, count in sorted(emission_counts[state].items())
        }
        unknown[state] = math.log(smoothing / total)

    return {
        "schema": "nexaloid.entity_hmm.v1",
        "entity_types": list(ENTITY_TYPES),
        "states": list(STATES),
        "start_states": list(START_STATES),
        "final_states": list(FINAL_STATES),
        "allowed_next": {state: list(values) for state, values in ALLOWED_NEXT.items()},
        "smoothing": smoothing,
        "sentence_count": sentence_count,
        "entity_count": entity_count,
        "vocab_size": len(vocab),
        "start": log_probs(start_counts, START_STATES, smoothing),
        "transition": {
            state: log_probs(transition_counts[state], ALLOWED_NEXT[state], smoothing)
            for state in STATES
        },
        "emission": emissions,
        "unknown_emission": unknown,
    }


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def emission(model: dict, state: str, char: str) -> float:
    return model["emission"][state].get(char, model["unknown_emission"][state])


def previous_states(model: dict, current: str) -> list[str]:
    return [
        previous
        for previous, next_states in model["allowed_next"].items()
        if current in next_states
    ]


def decode_tags(model: dict, text: str) -> list[str]:
    if not text:
        return []
    scores = {
        state: model["start"][state] + emission(model, state, text[0])
        for state in model["start_states"]
    }
    paths = {state: [state] for state in scores}
    for char in text[1:]:
        next_scores: dict[str, float] = {}
        next_paths: dict[str, list[str]] = {}
        for current in model["states"]:
            candidates = [previous for previous in previous_states(model, current) if previous in scores]
            if not candidates:
                continue
            best_previous = max(
                candidates,
                key=lambda previous: scores[previous] + model["transition"][previous][current],
            )
            next_scores[current] = (
                scores[best_previous]
                + model["transition"][best_previous][current]
                + emission(model, current, char)
            )
            next_paths[current] = paths[best_previous] + [current]
        scores = next_scores
        paths = next_paths
    finals = [state for state in model["final_states"] if state in scores]
    if not finals:
        return ["O"] * len(text)
    best_final = max(finals, key=lambda state: scores[state])
    return paths[best_final]


def tags_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if tag == "O":
            index += 1
            continue
        prefix, kind = tag.split("_", 1)
        if prefix == "S":
            spans.append((index, index + 1, kind))
            index += 1
            continue
        if prefix != "B":
            index += 1
            continue
        end = index + 1
        while end < len(tags) and tags[end] == f"M_{kind}":
            end += 1
        if end < len(tags) and tags[end] == f"E_{kind}":
            spans.append((index, end + 1, kind))
            index = end + 1
        else:
            index += 1
    return spans


def decode(model: dict, text: str) -> list[tuple[int, int, str]]:
    return tags_to_spans(decode_tags(model, text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/tasks/entity_hmm/train.tsv"))
    parser.add_argument("--out", type=Path, default=Path("data/tasks/entity_hmm/entity_hmm_wordhub.json"))
    parser.add_argument("--smoothing", type=float, default=0.1)
    args = parser.parse_args()

    model = train(args.train, args.smoothing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(model, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"sentences\t{model['sentence_count']}")
    print(f"entities\t{model['entity_count']}")
    print(f"vocab\t{model['vocab_size']}")
    print(f"states\t{len(model['states'])}")
    print(f"wrote\t{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
