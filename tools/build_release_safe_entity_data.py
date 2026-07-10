from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ENTITY_SOURCES = (
    ("PER", "THUOCL_lishimingren.txt", 400, ("{word}的生平资料正在整理。", "研究人员重新核对了{word}的相关记录。")),
    ("LOC", "THUOCL_diming.txt", 600, ("团队计划前往{word}开展调研。", "本次路线将经过{word}后继续向前。")),
    ("SPECIES", "THUOCL_animal.txt", 300, ("观测人员记录到了{word}。", "{word}被列入本次物种调查名单。")),
    ("PRODUCT", "THUOCL_car.txt", 200, ("展会上重点介绍了{word}的配置。", "用户正在比较{word}的使用体验。")),
    ("TECH", "THUOCL_IT.txt", 400, ("文档重点解释了{word}。", "项目正在评估{word}的应用。")),
    ("TECH", "THUOCL_medical.txt", 400, ("医学资料中记录了{word}。", "研究人员正在分析{word}。")),
    ("TECH", "THUOCL_caijing.txt", 300, ("财经报告重点说明了{word}。", "分析师正在研究{word}。")),
    ("LAW", "THUOCL_law.txt", 300, ("本案适用{word}。", "报告引用了{word}。")),
    ("PRODUCT", "THUOCL_food.txt", 200, ("菜单重点介绍了{word}。", "用户正在评价{word}的口感。")),
)
GENERAL_SOURCES = (
    ("THUOCL_chengyu.txt", 500),
    ("THUOCL_poem.txt", 500),
)
GAZETTEER_LIMITS = {
    "THUOCL_lishimingren.txt": None,
    "THUOCL_diming.txt": None,
    "THUOCL_animal.txt": 2_000,
    "THUOCL_car.txt": 1_000,
    "THUOCL_IT.txt": 3_000,
    "THUOCL_medical.txt": 3_000,
    "THUOCL_caijing.txt": 1_500,
    "THUOCL_law.txt": 3_000,
    "THUOCL_food.txt": 2_000,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(path: Path, limit: int | None, seen: set[str]) -> list[str]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        word = parts[0].strip()
        if not 2 <= len(word) <= 12 or any(char.isspace() for char in word) or word in seen:
            continue
        seen.add(word)
        out.append(word)
        if limit is not None and len(out) == limit:
            break
    if limit is not None and len(out) != limit:
        raise ValueError(f"{path}: expected {limit} usable words, got {len(out)}")
    return out


def row_id(prefix: str, value: str) -> str:
    return hashlib.sha256(f"{prefix}\0{value}".encode("utf-8")).hexdigest()[:20]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thuocl-root", type=Path, default=Path("G:/WordHub/THUOCL"))
    parser.add_argument(
        "--jd-labels",
        type=Path,
        default=Path("data/tasks/entity_llm_reconciled/jd_labels.jsonl"),
    )
    parser.add_argument(
        "--jd-manifest",
        type=Path,
        default=Path("data/tasks/entity_llm_reconciled/jd_labels.manifest.json"),
    )
    parser.add_argument(
        "--jd-readme",
        type=Path,
        default=Path("data/tasks/entity_release_labels/JD_README.md"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/tasks/entity_release_labels")
    )
    parser.add_argument(
        "--gazetteer",
        type=Path,
        default=Path("data/resources/lexicon_thuocl_mit.txt"),
    )
    args = parser.parse_args()

    thuocl_data = args.thuocl_root / "data"
    license_path = args.thuocl_root / "LICENSE"
    jd_manifest = json.loads(args.jd_manifest.read_text(encoding="utf-8"))
    jd_license = jd_manifest["source_license"]
    jd_readme = args.jd_readme
    if (
        sha256(args.jd_labels) != jd_manifest["output_sha256"]
        or "Apache-2.0" not in jd_license
        or "license: Apache License 2.0" not in jd_readme.read_text(encoding="utf-8")
        or "MIT License" not in license_path.read_text(encoding="utf-8")
    ):
        raise SystemExit("release-safe source license check failed")

    seen: set[str] = set()
    rows = []
    entity_words = []
    for kind, filename, limit, templates in ENTITY_SOURCES:
        for word in words(thuocl_data / filename, limit, seen):
            entity_words.append(word)
            group_id = row_id(f"thuocl-{kind}", word)
            for template_index, template in enumerate(templates):
                text = template.format(word=word)
                start = text.index(word)
                rows.append(
                    {
                        "id": row_id(f"{group_id}-{template_index}", text),
                        "group_id": group_id,
                        "text": text,
                        "entities": [
                            {"start": start, "end": start + len(word), "type": kind, "text": word}
                        ],
                        "source": "THUOCL MIT deterministic synthesis",
                    }
                )

    general_words = []
    negative_templates = (
        "系统正在处理关于{word}的常规信息。",
        "本次文档说明了{word}的基本情况。",
    )
    for filename, limit in GENERAL_SOURCES:
        for index, word in enumerate(words(thuocl_data / filename, limit, seen)):
            general_words.append(word)
            text = negative_templates[index % len(negative_templates)].format(word=word)
            rows.append(
                {
                    "id": row_id("thuocl-negative", text),
                    "group_id": row_id("thuocl-negative-group", word),
                    "text": text,
                    "entities": [],
                    "source": "THUOCL MIT deterministic synthesis",
                }
            )

    for raw in args.jd_labels.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        row["id"] = f"jd-{row['id']}"
        row["group_id"] = f"jd-{row['group_id']}"
        row["source"] = "JD comments Apache-2.0; DeepSeek-assisted labels"
        rows.append(row)

    rows.sort(key=lambda row: row["id"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out_dir / "release_labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    args.gazetteer.parent.mkdir(parents=True, exist_ok=True)
    gazetteer_words = set(entity_words + general_words)
    for _, filename, _, _ in ENTITY_SOURCES:
        gazetteer_words.update(
            words(thuocl_data / filename, GAZETTEER_LIMITS[filename], set())
        )
    # The runtime lexicon format reserves leading '#' lines for comments.
    gazetteer_words = {word for word in gazetteer_words if not word.startswith("#")}
    args.gazetteer.write_text(
        "".join(f"{word}\n" for word in sorted(gazetteer_words)),
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(license_path, args.out_dir / "THUOCL_LICENSE.txt")
    jd_readme_copy = args.out_dir / "JD_README.md"
    if jd_readme.resolve() != jd_readme_copy.resolve():
        shutil.copy2(jd_readme, jd_readme_copy)
    manifest = {
        "schema": "nexaloid.entity_release_labels.v1",
        "license_spdx": "Apache-2.0",
        "third_party": [
            {
                "name": "THUOCL",
                "license": "MIT",
                "license_sha256": sha256(license_path),
                "files": [
                    {
                        "name": filename,
                        "sha256": sha256(thuocl_data / filename),
                    }
                    for filename in sorted(
                        {item[1] for item in ENTITY_SOURCES}
                        | {item[0] for item in GENERAL_SOURCES}
                    )
                ],
            },
            {
                "name": "JD comments",
                "license": jd_license,
                "manifest_sha256": sha256(args.jd_manifest),
                "readme_sha256": sha256(jd_readme),
            },
        ],
        "generation": "deterministic templates plus existing DeepSeek-assisted JD labels",
        "sentence_count": len(rows),
        "entity_sentence_count": sum(bool(row["entities"]) for row in rows),
        "entity_count": sum(len(row["entities"]) for row in rows),
        "labels": {"path": labels_path.as_posix(), "sha256": sha256(labels_path)},
        "gazetteer": {
            "path": args.gazetteer.as_posix(),
            "sha256": sha256(args.gazetteer),
            "word_count": len(gazetteer_words),
            "source_limits": GAZETTEER_LIMITS,
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"sentences\t{len(rows)}")
    print(f"entities\t{manifest['entity_count']}")
    print(f"gazetteer\t{len(gazetteer_words)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
