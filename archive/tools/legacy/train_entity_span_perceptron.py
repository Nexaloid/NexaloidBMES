from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from build_entity_hmm_training import (
    COMMON_SURNAMES,
    CORE_PLACES,
    HAN,
    ORG_SUFFIXES as DATA_ORG_SUFFIXES,
    PERSON_STOP,
    PLACE_STOP,
    PLACE_SUFFIXES,
)
from check_entity_hmm_quality import print_metrics, score_cases
from train_entity_hmm import ENTITY_TYPES, iter_cases
from train_entity_perceptron import AveragedWeights


LABELS = ("NONE",) + ENTITY_TYPES
GENERIC_ORG_SUFFIXES = (
    "科技有限公司",
    "人民医院",
    "证券交易所",
    "联合实验室",
    "技术委员会",
    "有限公司",
    "研究院",
    "科学院",
    "计算所",
    "研究所",
    "委员会",
    "交易所",
    "实验室",
    "公安局",
    "派出所",
    "人民政府",
    "医院",
    "大学",
    "学院",
    "银行",
    "中心",
    "政府",
)
COMPOUND_SURNAMES = (
    "欧阳",
    "司马",
    "上官",
    "诸葛",
    "东方",
    "皇甫",
    "尉迟",
    "公羊",
    "赫连",
    "澹台",
    "公冶",
    "宗政",
    "濮阳",
    "淳于",
    "单于",
    "太叔",
    "申屠",
    "公孙",
    "仲孙",
    "轩辕",
    "令狐",
    "钟离",
    "宇文",
    "长孙",
    "慕容",
    "鲜于",
    "闾丘",
    "司徒",
    "司空",
)
ENTITY_LEFT_CUES = {
    "PER": (
        "采访了",
        "采访",
        "联系了",
        "联系",
        "会议由",
        "欢迎",
        "提到了",
        "提到",
        "昨日",
        "由",
        "派",
    ),
    "LOC": (
        "长期居住在",
        "居住在",
        "正在驶向",
        "活动将在",
        "驶向",
        "前往",
        "抵达",
        "位于",
        "来自",
        "回到",
        "进入",
        "访问",
        "在",
        "从",
        "向",
        "到",
        "赴",
    ),
    "ORG": (
        "报告由",
        "会议在",
        "访问了",
        "加入了",
        "我们与",
        "成立的",
        "成立",
        "拘捕",
        "这个",
        "代表",
        "访问",
        "加入",
        "与",
        "由",
        "在",
    ),
}
ENTITY_RIGHT_CUES = {
    "PER": (
        "明天参加",
        "今天发表",
        "参加",
        "发表",
        "主持",
        "负责",
        "抵达",
        "到访",
        "到",
        "加入",
        "代表",
        "精神",
        "工作",
    ),
    "LOC": (
        "市长",
        "今天出现",
        "今天降温",
        "开展",
        "发布",
        "返回",
        "附近",
        "出发",
        "举行",
        "调研",
        "迎来",
        "出现",
        "访问",
    ),
    "ORG": (
        "发布",
        "签署",
        "完成",
        "提交",
        "招聘",
        "宣布",
        "召开",
        "正在",
        "访问",
        "派",
    ),
}
DEFAULT_CANDIDATE_CONFIG = {
    "person_min_length": 2,
    "person_max_length": 4,
    "person_preferred_max_length": 3,
    "nickname_length": 2,
    "place_min_length": 2,
    "place_max_length": 8,
    "place_min_stem": 1,
    "org_min_length": 2,
    "org_max_length": 20,
    "org_min_stem": 2,
    "common_surnames": "".join(sorted(COMMON_SURNAMES)),
    "compound_surnames": list(COMPOUND_SURNAMES),
    "nickname_prefixes": "小阿",
    "person_stop": sorted(
        PERSON_STOP
        | {
            "全国",
            "文艺",
            "别可",
            "熊猫",
            "伊军",
            "明天",
            "成本",
            "管理",
            "申请",
            "公告",
            "通用",
            "平台",
            "相传",
        }
    ),
    "place_stop": sorted(
        PLACE_STOP | {"全国", "我国", "城市道路", "道路", "马路", "铁路", "水路", "线路"}
    ),
    "place_bad_fragments": ["部分地区", "偏远地区", "贫困山区", "国旗", "军旗"],
    "place_bad_prefixes": ["盘活"],
    "org_stop": ["基础设施投资银行"],
    "entity_bad_start_chars": "的了",
    "left_cues": {kind: list(values) for kind, values in ENTITY_LEFT_CUES.items()},
    "right_cues": {kind: list(values) for kind, values in ENTITY_RIGHT_CUES.items()},
    "core_places": sorted(CORE_PLACES | {"巴黎", "长春"}),
    "place_suffixes": sorted(set(PLACE_SUFFIXES), key=lambda value: (-len(value), value)),
    "org_suffixes": sorted(
        set(DATA_ORG_SUFFIXES) | set(GENERIC_ORG_SUFFIXES),
        key=lambda value: (-len(value), value),
    ),
}

Candidate = tuple[int, int, tuple[str, ...]]
Span = tuple[int, int, str]


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


def longest_suffix(value: str, suffixes: list[str]) -> str:
    return next((suffix for suffix in suffixes if value.endswith(suffix)), "-")


def generate_candidates(text: str, config: dict | None = None) -> list[Candidate]:
    config = config or DEFAULT_CANDIDATE_CONFIG
    candidates: dict[tuple[int, int], set[str]] = {}

    def add(start: int, end: int, source: str) -> None:
        value = text[start:end]
        if (
            start >= 0
            and end <= len(text)
            and HAN.fullmatch(value)
            and not (source == "LOC" and value in config.get("place_stop", ()))
            and not (
                source == "LOC"
                and any(fragment in value for fragment in config.get("place_bad_fragments", ()))
            )
            and not (
                source == "LOC"
                and any(value.startswith(prefix) for prefix in config.get("place_bad_prefixes", ()))
            )
            and not (
                source in ("LOC", "ORG")
                and value[0] in config.get("entity_bad_start_chars", "")
            )
            and not (source == "ORG" and value in config.get("org_stop", ()))
        ):
            candidates.setdefault((start, end), set()).add(source)

    for start in range(len(text)):
        is_person_start = text[start] in config["common_surnames"]
        is_nickname_start = (
            text[start] in config["nickname_prefixes"]
            and start + 1 < len(text)
            and text[start + 1] in config["common_surnames"]
        )
        is_stopped = any(text.startswith(stop, start) for stop in config["person_stop"])
        if is_nickname_start:
            add(start, start + config["nickname_length"], "PER")
        elif is_person_start and not is_stopped:
            for length in range(
                config["person_min_length"], config["person_preferred_max_length"] + 1
            ):
                if start + length <= len(text):
                    add(start, start + length, "PER")
            if text[start : start + 2] in config["compound_surnames"]:
                add(start, start + config["person_max_length"], "PER")

    for place in config["core_places"]:
        start = text.find(place)
        while start >= 0:
            add(start, start + len(place), "LOC")
            start = text.find(place, start + 1)

    for source, suffixes, min_length, max_length, minimum_stem in (
        (
            "LOC",
            config["place_suffixes"],
            config["place_min_length"],
            config["place_max_length"],
            config["place_min_stem"],
        ),
        (
            "ORG",
            config["org_suffixes"],
            config["org_min_length"],
            config["org_max_length"],
            config["org_min_stem"],
        ),
    ):
        for suffix in suffixes:
            suffix_start = text.find(suffix)
            while suffix_start >= 0:
                end = suffix_start + len(suffix)
                first_length = max(min_length, len(suffix) + minimum_stem)
                for length in range(first_length, max_length + 1):
                    add(end - length, end, source)
                suffix_start = text.find(suffix, suffix_start + 1)

    return [
        (start, end, tuple(kind for kind in ENTITY_TYPES if kind in sources))
        for (start, end), sources in sorted(candidates.items())
    ]


def candidate_spans(model: dict, text: str) -> list[tuple[int, int]]:
    return [
        (start, end)
        for start, end, _ in generate_candidates(
            text, model.get("candidate_config", DEFAULT_CANDIDATE_CONFIG)
        )
    ]


def span_features(text: str, candidate: Candidate, config: dict) -> tuple[str, ...]:
    start, end, sources = candidate
    value = text[start:end]
    left1 = char_at(text, start - 1)
    right1 = char_at(text, end)
    place_suffix = longest_suffix(value, config["place_suffixes"])
    org_suffix = longest_suffix(value, config["org_suffixes"])
    features = [
        "bias",
        f"length={len(value)}",
        f"sources={'+'.join(sources)}",
        f"first={value[0]}",
        f"left1={left1}",
        f"right1={right1}",
        f"left_class={char_class(left1)}",
        f"right_class={char_class(right1)}",
    ]
    if start == 0:
        features.append("start=BOS")
    if end == len(text):
        features.append("end=EOS")

    before = text[:start]
    after = text[end:]
    left_cues = config.get("left_cues", {})
    right_cues = config.get("right_cues", {})
    matched_left_cues: set[str] = set()
    matched_right_cues: set[str] = set()
    for source in sources:
        if any(before.endswith(cue) for cue in left_cues.get(source, ())):
            features.append(f"left_cue={source}")
            matched_left_cues.add(source)
        if any(after.startswith(cue) for cue in right_cues.get(source, ())):
            features.append(f"right_cue={source}")
            matched_right_cues.add(source)
        if any(
            len(cue) > 1 and cue in value
            for cue in left_cues.get(source, ())
        ):
            features.append(f"contains_left_cue={source}")
        if source in matched_left_cues and source in matched_right_cues:
            features.append(f"both_cues={source}")

    if "PER" in sources:
        features.extend((f"per_length={len(value)}", f"surname={value[0]}"))
        if value[0] in config["nickname_prefixes"]:
            features.append("nickname")
    if "LOC" in sources:
        features.append(f"place_suffix={place_suffix}")
        stem_length = len(value) - len(place_suffix) if place_suffix != "-" else len(value)
        features.append(f"place_stem_length={stem_length}")
        internal_suffixes = sum(
            suffix in value[:-1] for suffix in config["place_suffixes"]
        )
        features.append(f"internal_place_suffixes={min(internal_suffixes, 3)}")
        if value in config["core_places"]:
            features.append("core_place")
            if start == 0:
                features.append("core_place_at_bos")
            if "LOC" in matched_left_cues:
                features.append("core_place_left_cue")
            if "LOC" in matched_right_cues:
                features.append("core_place_right_cue")
    if "ORG" in sources:
        features.append(f"org_suffix={org_suffix}")
        stem_length = len(value) - len(org_suffix) if org_suffix != "-" else len(value)
        features.append(f"org_stem_length={stem_length}")
    features.extend(f"source={source}" for source in sources)
    return tuple(features)


def label_scores(
    weights: dict[str, dict[str, float]], features: tuple[str, ...]
) -> dict[str, float]:
    scores = {label: 0.0 for label in LABELS}
    for feature in features:
        for label, value in weights.get(feature, {}).items():
            scores[label] += value
    return scores


def predict_label(
    weights: dict[str, dict[str, float]],
    features: tuple[str, ...],
    sources: tuple[str, ...] = ENTITY_TYPES,
) -> str:
    scores = label_scores(weights, features)
    return max(("NONE",) + sources, key=lambda label: scores[label])


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(model: dict, text: str) -> list[Span]:
    config = model.get("candidate_config", DEFAULT_CANDIDATE_CONFIG)
    weights = model["weights"]
    margin = float(model.get("margin", 0.0))
    scored: list[tuple[float, int, int, str]] = []
    for start, end, sources in generate_candidates(text, config):
        scores = label_scores(weights, span_features(text, (start, end, sources), config))
        entity = max(sources, key=lambda label: scores[label])
        entity_margin = scores[entity] - scores["NONE"]
        if entity_margin > margin:
            scored.append((entity_margin, start, end, entity))

    # ponytail: flat greedy selection is enough for v1; use lattice ranking if
    # nested entities become a runtime requirement.
    spans: list[Span] = []
    for _, start, end, entity in sorted(
        scored,
        key=lambda item: (-item[0], -(item[2] - item[1]), item[1], item[2], item[3]),
    ):
        if all(end <= other_start or start >= other_end for other_start, other_end, _ in spans):
            spans.append((start, end, entity))
    return sorted(spans)


def candidate_metrics(cases, config: dict) -> dict:
    sentences = chars = candidates = gold = covered = 0
    by_type = {kind: Counter(gold=0, covered=0) for kind in ENTITY_TYPES}
    for text, spans in cases:
        sentences += 1
        chars += len(text)
        generated = {
            (start, end): sources
            for start, end, sources in generate_candidates(text, config)
        }
        candidates += len(generated)
        for start, end, kind in spans:
            gold += 1
            by_type[kind]["gold"] += 1
            if kind in generated.get((start, end), ()):
                covered += 1
                by_type[kind]["covered"] += 1
    return {
        "sentences": sentences,
        "chars": chars,
        "candidates": candidates,
        "candidates_per_1k_chars": candidates * 1000 / chars if chars else 0.0,
        "gold": gold,
        "covered": covered,
        "coverage": covered / gold if gold else 1.0,
        "by_type": {
            kind: {
                "gold": counts["gold"],
                "covered": counts["covered"],
                "coverage": counts["covered"] / counts["gold"] if counts["gold"] else 1.0,
            }
            for kind, counts in by_type.items()
        },
    }


def train_cases(
    cases: list[tuple[str, list[Span]]],
    epochs: int,
    seed: int,
    min_abs: float,
    margin: float,
    positive_weight: float,
    train_sha256: str,
    verbose: bool = True,
) -> dict:
    config = json.loads(json.dumps(DEFAULT_CANDIDATE_CONFIG, ensure_ascii=False))
    coverage = candidate_metrics(cases, config)
    if coverage["coverage"] < 0.99:
        raise ValueError(f"candidate coverage below 0.99: {coverage['coverage']:.4f}")
    if verbose and coverage["covered"] != coverage["gold"]:
        print(
            f"candidate_coverage\t{coverage['covered']}/{coverage['gold']}"
            f"\t{coverage['coverage']:.4f}"
        )

    weights = AveragedWeights()
    rng = random.Random(seed)
    label_counts: Counter[str] = Counter()
    for text, spans in cases:
        gold = {(start, end): kind for start, end, kind in spans}
        for start, end, _ in generate_candidates(text, config):
            label_counts[gold.get((start, end), "NONE")] += 1

    for epoch in range(epochs):
        order = list(range(len(cases)))
        rng.shuffle(order)
        mistakes = examples = 0
        for case_index in order:
            text, spans = cases[case_index]
            gold = {(start, end): kind for start, end, kind in spans}
            for candidate in generate_candidates(text, config):
                weights.tick()
                examples += 1
                features = span_features(text, candidate, config)
                expected = gold.get((candidate[0], candidate[1]), "NONE")
                predicted = predict_label(weights.weights, features, candidate[2])
                if predicted == expected:
                    continue
                mistakes += 1
                delta = positive_weight if expected != "NONE" else 1.0
                for feature in features:
                    weights.update(feature, expected, delta)
                    weights.update(feature, predicted, -delta)
        if verbose:
            print(f"epoch\t{epoch + 1}\tmistakes\t{mistakes}/{examples}")

    averaged = weights.average(min_abs)
    return {
        "schema": "nexaloid.entity_span_perceptron.v1",
        "labels": list(LABELS),
        "entity_types": list(ENTITY_TYPES),
        "epochs": epochs,
        "seed": seed,
        "margin": margin,
        "positive_weight": positive_weight,
        "min_abs": min_abs,
        "sentence_count": len(cases),
        "candidate_count": coverage["candidates"],
        "entity_count": coverage["gold"],
        "training_candidate_coverage": coverage,
        "label_counts": dict(sorted(label_counts.items())),
        "feature_count": len(averaged),
        "train_sha256": train_sha256,
        "candidate_config": config,
        "feature_templates": [
            "span_length",
            "candidate_source_and_suffix",
            "stem_length_and_nested_suffix_count",
            "first_and_boundary_characters",
            "character_classes",
            "sentence_boundaries",
            "typed_left_right_cues",
            "surname_nickname_and_core_place_flags",
        ],
        "weights": averaged,
    }


def train(
    path: Path,
    epochs: int,
    seed: int,
    min_abs: float,
    margin: float,
    positive_weight: float,
) -> dict:
    return train_cases(
        list(iter_cases(path)),
        epochs,
        seed,
        min_abs,
        margin,
        positive_weight,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def evaluate(model: dict, paths: dict[str, Path], max_failures: int) -> dict:
    scores = {
        name: score_cases(
            iter_cases(path), lambda text: decode(model, text), max_failures
        )
        for name, path in paths.items()
    }
    coverage = {
        name: candidate_metrics(iter_cases(path), model["candidate_config"])
        for name, path in paths.items()
    }
    return {
        "schema": "nexaloid.entity_span_perceptron_metrics.v1",
        "margin": model["margin"],
        "scores": scores,
        "candidates": coverage,
    }


def self_test() -> None:
    cases = [
        ("王小明在东湖区工作", [(0, 3, "PER"), (4, 7, "LOC")]),
        ("阿明加入星澜智算研究院", [(0, 2, "PER"), (4, 11, "ORG")]),
        ("上海证券交易所发布公告", [(0, 7, "ORG")]),
        ("你还小明天再说", []),
        ("研究生命起源", []),
    ]
    expected = {(0, 3), (4, 7)}
    assert expected <= set(candidate_spans({"candidate_config": DEFAULT_CANDIDATE_CONFIG}, cases[0][0]))
    first = train_cases(cases * 4, 8, 7, 0.0, 0.0, 4.0, "self-test", verbose=False)
    second = train_cases(cases * 4, 8, 7, 0.0, 0.0, 4.0, "self-test", verbose=False)
    assert first == second
    assert set(decode(first, cases[0][0])) == set(cases[0][1])
    assert decode(first, cases[3][0]) == []
    print("self_test\tok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/tasks/entity_hmm/train.tsv"))
    parser.add_argument("--dev", type=Path, default=Path("data/tasks/entity_hmm/dev.tsv"))
    parser.add_argument("--test", type=Path, default=Path("data/tasks/entity_hmm/test.tsv"))
    parser.add_argument(
        "--badcases", type=Path, default=Path("data/tasks/entity_hmm/eval_badcases.tsv")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tasks/entity_hmm/entity_span_perceptron_wordhub.json"),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("data/tasks/entity_hmm/entity_span_perceptron_metrics.json"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--min-abs", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.min_abs < 0:
        raise SystemExit("--min-abs must be non-negative")
    if args.positive_weight < 1:
        raise SystemExit("--positive-weight must be at least 1")

    model = train(
        args.train,
        args.epochs,
        args.seed,
        args.min_abs,
        args.margin,
        args.positive_weight,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(model, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    paths = {"dev": args.dev, "test": args.test, "badcases": args.badcases}
    metrics = evaluate(model, paths, args.max_failures)
    metrics["model_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"features\t{model['feature_count']}")
    print(f"candidates\t{model['candidate_count']}")
    for name, values in metrics["scores"].items():
        print_metrics(name, values)
        candidate_values = metrics["candidates"][name]
        print(
            f"{name}_candidates\tcoverage={candidate_values['coverage']:.4f}"
            f"\tper_1k={candidate_values['candidates_per_1k_chars']:.3f}"
        )
    print(f"wrote\t{args.out}")
    print(f"metrics\t{args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
