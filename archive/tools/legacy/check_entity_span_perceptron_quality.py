from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from check_entity_hmm_quality import format_span, metrics_from_counts, require
from train_entity_hmm import ENTITY_TYPES, iter_cases
from train_entity_span_perceptron import candidate_spans, decode, load_model


def score_cases(cases, decode_text, count_candidates, max_failures: int = 0) -> dict:
    total = exact = chars = candidates = 0
    true_positive = predicted = gold = 0
    type_counts = {
        kind: Counter(tp=0, predicted=0, gold=0)
        for kind in ENTITY_TYPES
    }
    negative_cases = negative_false_positives = 0
    failures: list[dict] = []

    for text, expected in cases:
        total += 1
        chars += len(text)
        candidates += count_candidates(text)
        actual_set = set(decode_text(text))
        expected_set = set(expected)
        overlap = actual_set & expected_set
        true_positive += len(overlap)
        predicted += len(actual_set)
        gold += len(expected_set)
        exact += actual_set == expected_set
        if not expected_set:
            negative_cases += 1
            negative_false_positives += len(actual_set)
        for kind in ENTITY_TYPES:
            actual_type = {span for span in actual_set if span[2] == kind}
            expected_type = {span for span in expected_set if span[2] == kind}
            type_counts[kind]["tp"] += len(actual_type & expected_type)
            type_counts[kind]["predicted"] += len(actual_type)
            type_counts[kind]["gold"] += len(expected_type)
        if actual_set != expected_set and (max_failures < 0 or len(failures) < max_failures):
            failures.append(
                {
                    "text": text,
                    "expected": [format_span(text, span) for span in sorted(expected_set)],
                    "actual": [format_span(text, span) for span in sorted(actual_set)],
                }
            )

    metrics = metrics_from_counts(true_positive, predicted, gold)
    metrics.update(
        {
            "exact": exact,
            "total": total,
            "exact_rate": exact / total if total else 0.0,
            "chars": chars,
            "false_positives_per_1k_chars": (
                (predicted - true_positive) * 1000 / chars if chars else 0.0
            ),
            "candidate_count": candidates,
            "candidates_per_1k_chars": candidates * 1000 / chars if chars else 0.0,
            "negative_cases": negative_cases,
            "negative_false_positives": negative_false_positives,
            "by_type": {
                kind: metrics_from_counts(
                    values["tp"], values["predicted"], values["gold"]
                )
                for kind, values in type_counts.items()
            },
            "failures": failures,
        }
    )
    return metrics


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name}\t{metrics['exact']}/{metrics['total']}"
        f"\tprecision={metrics['precision']:.4f}"
        f"\trecall={metrics['recall']:.4f}"
        f"\tf1={metrics['f1']:.4f}"
        f"\tfp_per_1k={metrics['false_positives_per_1k_chars']:.3f}"
        f"\tcandidates_per_1k={metrics['candidates_per_1k_chars']:.3f}"
    )
    for kind, values in metrics["by_type"].items():
        print(
            f"{name}_{kind}"
            f"\tprecision={values['precision']:.4f}"
            f"\trecall={values['recall']:.4f}"
            f"\tf1={values['f1']:.4f}"
        )
    for failure in metrics["failures"]:
        print(f"FAIL\t{failure['text']}")
        print(f"  expected\t{'; '.join(failure['expected']) or '-'}")
        print(f"  actual\t{'; '.join(failure['actual']) or '-'}")


def self_test() -> None:
    cases = [
        ("甲乙丙", [(0, 2, "ORG")]),
        ("普通文本", []),
    ]
    predictions = {
        "甲乙丙": [(0, 2, "ORG"), (2, 3, "PER")],
        "普通文本": [],
    }
    candidate_counts = {"甲乙丙": 3, "普通文本": 2}
    metrics = score_cases(
        cases,
        lambda text: predictions[text],
        lambda text: candidate_counts[text],
    )
    assert metrics["tp"] == 1
    assert metrics["predicted"] == 2
    assert metrics["gold"] == 1
    assert metrics["candidate_count"] == 5
    assert metrics["negative_cases"] == 1
    assert metrics["negative_false_positives"] == 0
    assert metrics["by_type"]["ORG"]["f1"] == 1.0
    assert metrics["by_type"]["PER"]["precision"] == 0.0
    print("self_test\tok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/tasks/entity_hmm/entity_span_perceptron_wordhub.json"),
    )
    parser.add_argument("--dev", type=Path, default=Path("data/tasks/entity_hmm/dev.tsv"))
    parser.add_argument("--test", type=Path, default=Path("data/tasks/entity_hmm/test.tsv"))
    parser.add_argument(
        "--badcases",
        type=Path,
        default=Path("data/tasks/entity_hmm/eval_badcases.tsv"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--max-fp-per-1k", type=float, default=1.0)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    model = load_model(args.model)
    if model.get("schema") != "nexaloid.entity_span_perceptron.v1":
        raise SystemExit("invalid model schema")
    results = {
        "dev": score_cases(
            iter_cases(args.dev),
            lambda text: decode(model, text),
            lambda text: len(candidate_spans(model, text)),
            args.max_failures,
        ),
        "test": score_cases(
            iter_cases(args.test),
            lambda text: decode(model, text),
            lambda text: len(candidate_spans(model, text)),
            args.max_failures,
        ),
        "badcases": score_cases(
            iter_cases(args.badcases),
            lambda text: decode(model, text),
            lambda text: len(candidate_spans(model, text)),
            args.max_failures,
        ),
    }
    for name, metrics in results.items():
        print_metrics(name, metrics)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if not args.report_only:
        for split in ("dev", "test"):
            require(f"{split} precision", results[split]["precision"] >= args.min_precision)
            require(f"{split} recall", results[split]["recall"] >= args.min_recall)
            require(
                f"{split} false positives",
                results[split]["false_positives_per_1k_chars"] <= args.max_fp_per_1k,
            )
        require(
            "badcase negative false positives",
            results["badcases"]["negative_false_positives"] == 0,
        )
        print("quality_gate\tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
