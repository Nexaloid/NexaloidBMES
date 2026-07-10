from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lexicon import load_lexicon
from train_entity_llm_perceptron import evaluate, load_cases


def require(name: str, ok: bool) -> None:
    if not ok:
        raise SystemExit(f"quality gate failed: {name}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/tasks/entity_llm_bmes/entity_llm_perceptron_generic.json"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/tasks/entity_llm_bmes")
    )
    parser.add_argument("--min-dev-f1", type=float, default=0.49)
    parser.add_argument("--min-test-f1", type=float, default=0.48)
    args = parser.parse_args()

    model = json.loads(args.model.read_text(encoding="utf-8"))
    require("schema", model.get("schema") == "nexaloid.entity_llm_perceptron.v2")
    require("generic", model.get("generic") is True)
    require("states", model.get("states") == ["O", "B", "M", "E", "S"])

    data_manifest_path = args.data_dir / "manifest.json"
    if data_manifest_path.is_file():
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        for item in data_manifest.get("inputs", []) + data_manifest.get("outputs", []):
            path = Path(item["path"])
            require(f"manifest input {path}", path.is_file() and sha256(path) == item["sha256"])

    gazetteer = model.get("gazetteer") or {}
    lexicon_path = Path(gazetteer.get("path", ""))
    require("gazetteer path", lexicon_path.is_file())
    require("gazetteer hash", sha256(lexicon_path) == gazetteer.get("sha256"))
    max_word_len = int(gazetteer["max_word_len"])
    lexicon = {
        word for word in load_lexicon(lexicon_path) if 2 <= len(word) <= max_word_len
    }
    entity_lexicon = set(gazetteer.get("training_entity_words", []))
    require("gazetteer size", len(lexicon) == gazetteer.get("word_count"))
    require(
        "entity gazetteer size",
        len(entity_lexicon) == gazetteer.get("training_entity_word_count"),
    )

    seen_groups: dict[str, str] = {}
    cases = {}
    for split in ("train", "dev", "test"):
        path = args.data_dir / f"{split}.jsonl"
        require("data hash", sha256(path) == model["data"][split]["sha256"])
        rows = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw]
        for row in rows:
            previous = seen_groups.setdefault(row["group_id"], split)
            require("group leakage", previous == split)
        cases[split] = load_cases(path)

    for split, threshold in (("dev", args.min_dev_f1), ("test", args.min_test_f1)):
        metrics = evaluate(
            model["weights"],
            cases[split],
            generic=True,
            lexicon=lexicon,
            entity_lexicon=entity_lexicon,
            max_word_len=max_word_len,
        )["overall"]
        require("stored metrics", metrics == model["metrics"][split]["overall"])
        require(f"{split} f1", metrics["f1"] >= threshold)
        print(
            f"{split}\tP={metrics['precision']:.6f}\t"
            f"R={metrics['recall']:.6f}\tF1={metrics['f1']:.6f}"
        )
    print(f"groups\t{len(seen_groups)}")
    print("quality_gate\tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
