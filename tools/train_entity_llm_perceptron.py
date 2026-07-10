from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from lexicon import load_lexicon
from label_entity_nouns_llm import ENTITY_TYPES
from perceptron_core import (
    START,
    AveragedWeights,
    features_at,
    update_sequence,
)


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


def build_allowed_next() -> dict[str, tuple[str, ...]]:
    starts = START_STATES
    allowed = {"O": starts}
    for kind in ENTITY_TYPES:
        allowed[f"B_{kind}"] = (f"M_{kind}", f"E_{kind}")
        allowed[f"M_{kind}"] = (f"M_{kind}", f"E_{kind}")
        allowed[f"E_{kind}"] = starts
        allowed[f"S_{kind}"] = starts
    return allowed


ALLOWED_NEXT = build_allowed_next()
GENERIC_STATES = ("O", "B", "M", "E", "S")
GENERIC_START_STATES = ("O", "B", "S")
GENERIC_FINAL_STATES = ("O", "E", "S")
GENERIC_ALLOWED_NEXT = {
    "O": GENERIC_START_STATES,
    "B": ("M", "E"),
    "M": ("M", "E"),
    "E": GENERIC_START_STATES,
    "S": GENERIC_START_STATES,
}


def state_system(generic: bool):
    if generic:
        return (
            GENERIC_STATES,
            GENERIC_START_STATES,
            GENERIC_FINAL_STATES,
            GENERIC_ALLOWED_NEXT,
        )
    return STATES, START_STATES, FINAL_STATES, ALLOWED_NEXT


def load_cases(path: Path) -> list[tuple[str, list[tuple[int, int, str]]]]:
    cases = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        location = f"{path}:{line_number}"
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"{location}: text must be a string")
        text = row["text"]
        if not text:
            raise ValueError(f"{location}: text must not be empty")
        raw_spans = row.get("spans")
        if not isinstance(raw_spans, list):
            raise ValueError(f"{location}: spans must be a list")
        spans = []
        cursor = 0
        for index, span in enumerate(raw_spans):
            if not isinstance(span, list) or len(span) != 3:
                raise ValueError(f"{location}: spans[{index}] must be [start,end,type]")
            start, end, kind = span
            if (
                type(start) is not int
                or type(end) is not int
                or kind not in ENTITY_TYPES
                or start < cursor
                or start >= end
                or end > len(text)
            ):
                raise ValueError(f"{location}: invalid span {span!r}")
            spans.append((start, end, kind))
            cursor = end
        cases.append((text, spans))
    return cases


def entity_tags(kind: str, length: int, generic: bool = False) -> list[str]:
    if length == 1:
        prefixes = ["S"]
    elif length == 2:
        prefixes = ["B", "E"]
    else:
        prefixes = ["B"] + ["M"] * (length - 2) + ["E"]
    return prefixes if generic else [f"{prefix}_{kind}" for prefix in prefixes]


def spans_to_tags(
    text: str,
    spans: list[tuple[int, int, str]],
    generic: bool = False,
) -> list[str]:
    tags = ["O"] * len(text)
    for start, end, kind in spans:
        tags[start:end] = entity_tags(kind, end - start, generic)
    return tags


def tags_to_spans(
    tags: list[str], generic: bool = False
) -> list[tuple[int, int, str]]:
    spans = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if tag == "O":
            index += 1
            continue
        if generic:
            prefix, kind = tag, "ENTITY"
        else:
            prefix, kind = tag.split("_", 1)
        if prefix == "S":
            spans.append((index, index + 1, kind))
            index += 1
            continue
        if prefix != "B":
            raise ValueError(f"invalid BMES sequence at {index}: {tag}")
        end = index + 1
        middle = "M" if generic else f"M_{kind}"
        final = "E" if generic else f"E_{kind}"
        while end < len(tags) and tags[end] == middle:
            end += 1
        if end >= len(tags) or tags[end] != final:
            raise ValueError(f"unterminated BMES entity at {index}: {tag}")
        spans.append((index, end + 1, kind))
        index = end + 1
    return spans


def emission_scores(
    weights: dict[str, dict[str, float]],
    features: tuple[str, ...],
    states: tuple[str, ...],
) -> dict[str, float]:
    scores = {state: 0.0 for state in states}
    for feature in features:
        for state, value in weights.get(feature, {}).items():
            scores[state] += value
    return scores


def decode_tags_with_features(
    weights: dict[str, dict[str, float]],
    sentence_features: list[tuple[str, ...]],
    generic: bool = False,
) -> list[str]:
    if not sentence_features:
        return []
    states, start_states, final_states, allowed_next = state_system(generic)
    first_emission = emission_scores(weights, sentence_features[0], states)
    start_transition = weights.get(f"T={START}", {})
    scores = {
        state: first_emission[state] + start_transition.get(state, 0.0)
        for state in start_states
    }
    backpointers: list[dict[str, str]] = []
    for features in sentence_features[1:]:
        emissions = emission_scores(weights, features, states)
        next_scores: dict[str, float] = {}
        previous_for: dict[str, str] = {}
        for previous, base_score in scores.items():
            transition = weights.get(f"T={previous}", {})
            for current in allowed_next[previous]:
                candidate = (
                    base_score
                    + transition.get(current, 0.0)
                    + emissions[current]
                )
                if current not in next_scores or candidate > next_scores[current]:
                    next_scores[current] = candidate
                    previous_for[current] = previous
        scores = next_scores
        backpointers.append(previous_for)

    finals = [state for state in final_states if state in scores]
    if not finals:
        return ["O"] * len(sentence_features)
    best = max(finals, key=lambda state: scores[state])
    tags = [best]
    for previous_for in reversed(backpointers):
        tags.append(previous_for[tags[-1]])
    tags.reverse()
    return tags


def gazetteer_features(
    text: str,
    lexicon: set[str],
    max_word_len: int,
    prefix: str,
) -> list[tuple[str, ...]]:
    features: list[set[str]] = [set() for _ in text]
    for start in range(len(text)):
        for end in range(start + 2, min(len(text), start + max_word_len) + 1):
            if text[start:end] not in lexicon:
                continue
            length = end - start
            bucket = str(length) if length <= 4 else "5+"
            for offset, tag in enumerate(entity_tags("ENTITY", length, generic=True)):
                features[start + offset].add(f"{prefix}={tag}")
                features[start + offset].add(f"{prefix}={tag}:{bucket}")
    return [tuple(sorted(values)) for values in features]


def sentence_features(
    text: str,
    lexicon: set[str] | None = None,
    entity_lexicon: set[str] | None = None,
    max_word_len: int = 12,
) -> list[tuple[str, ...]]:
    base = [features_at(text, index) for index in range(len(text))]
    for prefix, words in (("lx", lexicon), ("ex", entity_lexicon)):
        if words:
            extra = gazetteer_features(text, words, max_word_len, prefix)
            base = [
                features + extra_features
                for features, extra_features in zip(base, extra)
            ]
    return base


def decode(
    weights: dict[str, dict[str, float]],
    text: str,
    generic: bool = False,
    lexicon: set[str] | None = None,
    entity_lexicon: set[str] | None = None,
    max_word_len: int = 12,
) -> list[tuple[int, int, str]]:
    return tags_to_spans(
        decode_tags_with_features(
            weights,
            sentence_features(text, lexicon, entity_lexicon, max_word_len),
            generic,
        ),
        generic,
    )


def score_counts(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def evaluate(
    weights: dict[str, dict[str, float]],
    cases: list[tuple[str, list[tuple[int, int, str]]]],
    generic: bool = False,
    lexicon: set[str] | None = None,
    entity_lexicon: set[str] | None = None,
    max_word_len: int = 12,
) -> dict:
    overall: Counter[str] = Counter()
    kinds = ("ENTITY",) if generic else ENTITY_TYPES
    by_type = {kind: Counter() for kind in kinds}
    for text, gold_spans in cases:
        gold = {
            (start, end, "ENTITY" if generic else kind)
            for start, end, kind in gold_spans
        }
        predicted = set(
            decode(
                weights,
                text,
                generic,
                lexicon,
                entity_lexicon,
                max_word_len,
            )
        )
        for label, values in (("tp", gold & predicted), ("fn", gold - predicted), ("fp", predicted - gold)):
            overall[label] += len(values)
            for _, _, kind in values:
                by_type[kind][label] += 1
    return {
        "overall": score_counts(overall["tp"], overall["fp"], overall["fn"]),
        "by_type": {
            kind: score_counts(counts["tp"], counts["fp"], counts["fn"])
            for kind, counts in by_type.items()
        },
    }


def train(
    cases: list[tuple[str, list[tuple[int, int, str]]]],
    epochs: int,
    seed: int,
    min_abs: float,
    generic: bool = False,
    lexicon: set[str] | None = None,
    entity_lexicon: set[str] | None = None,
    max_word_len: int = 12,
) -> dict[str, dict[str, float]]:
    prepared = [
        (
            spans_to_tags(text, spans, generic),
            sentence_features(text, lexicon, entity_lexicon, max_word_len),
        )
        for text, spans in cases
    ]
    weights = AveragedWeights()
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(prepared)
        mistakes = 0
        for gold, features in prepared:
            weights.tick()
            predicted = decode_tags_with_features(weights.weights, features, generic)
            if predicted != gold:
                mistakes += 1
                update_sequence(weights, features, gold, predicted)
        print(f"epoch\t{epoch + 1}\tmistakes\t{mistakes}/{len(prepared)}")
    return weights.average(min_abs)


def print_metrics(split: str, metrics: dict) -> None:
    overall = metrics["overall"]
    print(
        f"{split}\toverall\tP={overall['precision']:.6f}\t"
        f"R={overall['recall']:.6f}\tF1={overall['f1']:.6f}"
    )
    for kind, score in metrics["by_type"].items():
        if score["tp"] or score["fp"] or score["fn"]:
            print(
                f"{split}\t{kind}\tP={score['precision']:.6f}\t"
                f"R={score['recall']:.6f}\tF1={score['f1']:.6f}"
            )


def self_test() -> None:
    text = "张三访问北京"
    weights = {
        "c0=张": {"B_PER": 5.0},
        "c0=三": {"E_PER": 5.0},
        "c0=北": {"B_LOC": 5.0},
        "c0=京": {"E_LOC": 5.0},
    }
    expected = [(0, 2, "PER"), (4, 6, "LOC")]
    assert decode(weights, text) == expected
    metrics = evaluate(weights, [(text, expected)])
    assert metrics["overall"]["f1"] == 1.0
    generic_weights = {
        "c0=张": {"B": 5.0},
        "c0=三": {"E": 5.0},
        "c0=北": {"B": 5.0},
        "c0=京": {"E": 5.0},
    }
    generic_expected = [(0, 2, "ENTITY"), (4, 6, "ENTITY")]
    assert decode(generic_weights, text, generic=True) == generic_expected
    generic_metrics = evaluate(generic_weights, [(text, expected)], generic=True)
    assert generic_metrics["overall"]["f1"] == 1.0
    lexicon_features = sentence_features("访问北京大学", {"北京", "北京大学"})
    assert "lx=B:2" in lexicon_features[2]
    assert "lx=B:4" in lexicon_features[2]
    assert "lx=E:2" in lexicon_features[3]
    assert "lx=M:4" in lexicon_features[3]
    assert "lx=E:4" in lexicon_features[5]
    entity_features = sentence_features(
        "访问北京大学", entity_lexicon={"北京大学"}
    )
    assert "ex=B:4" in entity_features[2]
    assert "ex=E:4" in entity_features[5]
    assert len(STATES) == 1 + 4 * len(ENTITY_TYPES)
    assert len(GENERIC_STATES) == 5
    print("self_test\tok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/tasks/entity_llm_bmes")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-abs", type=float, default=0.05)
    parser.add_argument("--generic", action="store_true")
    parser.add_argument("--gazetteer", type=Path)
    parser.add_argument("--train-entity-gazetteer-min-count", type=int, default=0)
    parser.add_argument("--gazetteer-max-word-len", type=int, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    paths = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "dev", "test")}
    cases = {split: load_cases(path) for split, path in paths.items()}
    if not cases["train"]:
        raise SystemExit("training data is empty")
    if args.gazetteer_max_word_len < 2:
        raise SystemExit("--gazetteer-max-word-len must be at least 2")
    if args.train_entity_gazetteer_min_count < 0:
        raise SystemExit("--train-entity-gazetteer-min-count must not be negative")
    lexicon = load_lexicon(args.gazetteer) if args.gazetteer else set()
    lexicon = {
        word for word in lexicon if 2 <= len(word) <= args.gazetteer_max_word_len
    }
    entity_counts = Counter(
        text[start:end]
        for text, spans in cases["train"]
        for start, end, _ in spans
        if 2 <= end - start <= args.gazetteer_max_word_len
    )
    entity_lexicon = {
        word
        for word, count in entity_counts.items()
        if args.train_entity_gazetteer_min_count
        and count >= args.train_entity_gazetteer_min_count
    }
    out = args.out or args.data_dir / (
        "entity_llm_perceptron_generic.json"
        if args.generic
        else "entity_llm_perceptron.json"
    )
    states, start_states, final_states, allowed_next = state_system(args.generic)
    weights = train(
        cases["train"],
        args.epochs,
        args.seed,
        args.min_abs,
        args.generic,
        lexicon,
        entity_lexicon,
        args.gazetteer_max_word_len,
    )
    metrics = {
        split: evaluate(
            weights,
            rows,
            args.generic,
            lexicon,
            entity_lexicon,
            args.gazetteer_max_word_len,
        )
        for split, rows in cases.items()
    }
    artifact = {
        "schema": "nexaloid.entity_llm_perceptron.v2",
        "generic": args.generic,
        "evaluation": "exact_boundary" if args.generic else "exact_typed_span",
        "entity_types": ["ENTITY"] if args.generic else list(ENTITY_TYPES),
        "states": list(states),
        "start_states": list(start_states),
        "final_states": list(final_states),
        "allowed_next": {state: list(values) for state, values in allowed_next.items()},
        "epochs": args.epochs,
        "seed": args.seed,
        "min_abs": args.min_abs,
        "feature_count": len(weights),
        "feature_templates": [
            "bias",
            "characters[-2..+2]",
            "character_bigrams",
            "character_trigrams",
            "character_classes[-1..+1]",
            "previous_tag_transition",
        ]
        + (["gazetteer_bmes", "gazetteer_bmes_length"] if lexicon else [])
        + (
            ["training_entity_bmes", "training_entity_bmes_length"]
            if entity_lexicon
            else []
        ),
        "gazetteer": (
            {
                "path": args.gazetteer.as_posix() if args.gazetteer else None,
                "sha256": (
                    hashlib.sha256(args.gazetteer.read_bytes()).hexdigest()
                    if args.gazetteer
                    else None
                ),
                "word_count": len(lexicon),
                "max_word_len": args.gazetteer_max_word_len,
                "training_entity_word_count": len(entity_lexicon),
                "training_entity_min_count": args.train_entity_gazetteer_min_count,
                "training_entity_words": sorted(entity_lexicon),
            }
            if args.gazetteer or entity_lexicon
            else None
        ),
        "data": {
            split: {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sentence_count": len(cases[split]),
            }
            for split, path in paths.items()
        },
        "metrics": metrics,
        "weights": weights,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for split in ("train", "dev", "test"):
        print_metrics(split, metrics[split])
    print(f"features\t{artifact['feature_count']}")
    print(f"wrote\t{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
