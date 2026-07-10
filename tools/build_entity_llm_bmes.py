from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from label_entity_nouns_llm import ENTITY_TYPES

SPLITS = ("train", "dev", "test")
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    datasets = {split: [] for split in SPLITS}
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    for source_rows in by_source.values():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in source_rows:
            groups[row["group_id"]].append(row)
        targets = {
            split: max(1.0, len(source_rows) * SPLIT_RATIOS[split])
            for split in SPLITS
        }
        counts: Counter[str] = Counter()
        ordered_groups = sorted(
            groups.items(),
            key=lambda item: (
                -len(item[1]),
                hashlib.sha256(item[0].encode("utf-8")).digest(),
            ),
        )
        for _, group_rows in ordered_groups:
            split = min(
                SPLITS,
                key=lambda name: (
                    (counts[name] + len(group_rows)) / targets[name],
                    SPLITS.index(name),
                ),
            )
            datasets[split].extend(group_rows)
            counts[split] += len(group_rows)
    return datasets


def validate_row(value: object, location: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: row must be a JSON object")
    sentence_id = value.get("id")
    group_id = value.get("group_id", sentence_id)
    text = value.get("text")
    entities = value.get("entities")
    if not isinstance(sentence_id, str) or not sentence_id:
        raise ValueError(f"{location}: id must be a non-empty string")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError(f"{location}: group_id must be a non-empty string")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{location}: text must be a non-empty string")
    if any(char in text for char in "\t\r\n"):
        raise ValueError(f"{location}: text contains a TSV control character")
    if not isinstance(entities, list):
        raise ValueError(f"{location}: entities must be a list")

    spans: list[tuple[int, int, str]] = []
    for index, entity in enumerate(entities):
        entity_location = f"{location}:entities[{index}]"
        if not isinstance(entity, dict):
            raise ValueError(f"{entity_location}: entity must be an object")
        start = entity.get("start")
        end = entity.get("end")
        kind = entity.get("type")
        entity_text = entity.get("text")
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"{entity_location}: start/end must be integers")
        if kind not in ENTITY_TYPES:
            raise ValueError(f"{entity_location}: invalid entity type {kind!r}")
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"{entity_location}: span {start}:{end} is out of bounds")
        if not isinstance(entity_text, str) or text[start:end] != entity_text:
            raise ValueError(f"{entity_location}: entity text does not match its span")
        spans.append((start, end, kind))

    spans.sort(key=lambda span: (span[0], span[1], span[2]))
    cursor = 0
    for start, end, _ in spans:
        if start < cursor:
            raise ValueError(f"{location}: entity spans overlap")
        cursor = end
    return {"id": sentence_id, "group_id": group_id, "text": text, "spans": spans}


def load_rows(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    inputs = []
    for path in paths:
        file_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
                row = validate_row(value, location)
                if row["id"] in seen_ids:
                    raise ValueError(f"{location}: duplicate sentence id {row['id']!r}")
                if row["text"] in seen_texts:
                    raise ValueError(f"{location}: duplicate sentence text")
                seen_ids.add(row["id"])
                seen_texts.add(row["text"])
                row["source"] = path.as_posix()
                rows.append(row)
                file_rows += 1
        inputs.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "sentence_count": file_rows,
            }
        )
    return rows, inputs


def encode_spans(spans: list[tuple[int, int, str]]) -> str:
    return ",".join(f"{start}:{end}:{kind}" for start, end, kind in spans) or "-"


def write_split(out_dir: Path, split: str, rows: list[dict]) -> list[dict]:
    tsv_path = out_dir / f"{split}.tsv"
    jsonl_path = out_dir / f"{split}.jsonl"
    tsv_lines = ["# text\tstart:end:type,..."]
    tsv_lines.extend(
        f"{row['text']}\t{encode_spans(row['spans'])}" for row in rows
    )
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8", newline="\n")
    jsonl_path.write_text(
        "".join(
            json.dumps(
                {
                    "group_id": row["group_id"],
                    "source": row["source"],
                    "text": row["text"],
                    "spans": row["spans"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    return [
        {"path": tsv_path.as_posix(), "sha256": sha256(tsv_path)},
        {"path": jsonl_path.as_posix(), "sha256": sha256(jsonl_path)},
    ]


def self_test() -> None:
    text = "|".join(ENTITY_TYPES)
    entities = []
    cursor = 0
    for kind in ENTITY_TYPES:
        entities.append(
            {"start": cursor, "end": cursor + len(kind), "type": kind, "text": kind}
        )
        cursor += len(kind) + 1
    row = validate_row(
        {
            "id": "all-types",
            "group_id": "all-types-document",
            "text": text,
            "entities": entities,
        },
        "self-test",
    )
    assert len(row["spans"]) == len(ENTITY_TYPES)
    assert row["group_id"] == "all-types-document"
    assert encode_spans([]) == "-"
    demo_rows = [
        {"id": f"row-{index}", "group_id": f"group-{index // 2}", "source": "demo"}
        for index in range(20)
    ]
    demo_splits = split_rows(demo_rows)
    seen_groups = {}
    for split, split_rows_value in demo_splits.items():
        for demo_row in split_rows_value:
            assert seen_groups.setdefault(demo_row["group_id"], split) == split
    try:
        validate_row(
            {
                "id": "bad",
                "text": "实体",
                "entities": [
                    {"start": 0, "end": 2, "type": "UNKNOWN", "text": "实体"}
                ],
            },
            "self-test",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid entity type was accepted")
    print("self_test\tok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/tasks/entity_llm_reconciled")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/tasks/entity_llm_bmes")
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    paths = sorted(args.input_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no JSONL inputs found under {args.input_dir}")
    rows, inputs = load_rows(paths)
    datasets = split_rows(rows)
    for split in SPLITS:
        datasets[split].sort(key=lambda row: row["id"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    split_stats = {}
    total_types: Counter[str] = Counter()
    for split in SPLITS:
        split_rows_value = datasets[split]
        outputs.extend(write_split(args.out_dir, split, split_rows_value))
        type_counts = Counter(
            kind for row in split_rows_value for _, _, kind in row["spans"]
        )
        total_types.update(type_counts)
        split_stats[split] = {
            "sentence_count": len(split_rows_value),
            "group_count": len({row["group_id"] for row in split_rows_value}),
            "character_count": sum(len(row["text"]) for row in split_rows_value),
            "entity_count": sum(len(row["spans"]) for row in split_rows_value),
            "type_counts": dict(sorted(type_counts.items())),
        }

    manifest = {
        "schema": "nexaloid.entity_llm_bmes_data.v1",
        "entity_types": list(ENTITY_TYPES),
        "split_method": "source-stratified deterministic greedy grouping; target train/dev/test=80/10/10 by sentence count",
        "inputs": inputs,
        "outputs": outputs,
        "sentence_count": len(rows),
        "group_count": len({row["group_id"] for row in rows}),
        "entity_count": sum(total_types.values()),
        "type_counts": dict(sorted(total_types.items())),
        "splits": split_stats,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for split in SPLITS:
        print(f"{split}_sentences\t{split_stats[split]['sentence_count']}")
    print(f"entities\t{manifest['entity_count']}")
    print(f"wrote\t{args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
