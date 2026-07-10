from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_entity_hmm_quality import print_metrics, require, score_cases
from train_entity_hmm import iter_cases
from train_entity_perceptron import decode, load_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/tasks/entity_hmm/entity_perceptron_wordhub.json"))
    parser.add_argument("--dev", type=Path, default=Path("data/tasks/entity_hmm/dev.tsv"))
    parser.add_argument("--test", type=Path, default=Path("data/tasks/entity_hmm/test.tsv"))
    parser.add_argument("--badcases", type=Path, default=Path("data/tasks/entity_hmm/eval_badcases.tsv"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--max-fp-per-1k", type=float, default=1.0)
    args = parser.parse_args()

    model = load_model(args.model)
    if model.get("schema") != "nexaloid.entity_perceptron.v1":
        raise SystemExit("invalid model schema")
    decode_text = lambda text: decode(model, text)
    results = {
        "dev": score_cases(iter_cases(args.dev), decode_text, args.max_failures),
        "test": score_cases(iter_cases(args.test), decode_text, args.max_failures),
        "badcases": score_cases(iter_cases(args.badcases), decode_text, args.max_failures),
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
        require("badcase negative false positives", results["badcases"]["negative_false_positives"] == 0)
        print("quality_gate\tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
