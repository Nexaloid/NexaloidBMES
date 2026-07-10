from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "dev", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_hmm(path: Path, split: str) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        text, encoded = raw.split("\t", 1)
        spans = []
        if encoded != "-":
            for item in encoded.split(","):
                start, end, kind = item.split(":")
                spans.append([int(start), int(end), kind])
        digest = hashlib.sha256(f"{text}\0{encoded}".encode("utf-8")).hexdigest()[:20]
        rows.append(
            {
                "group_id": f"thuocl-hmm-{split}-{digest}",
                "source": path.as_posix(),
                "text": text,
                "spans": spans,
            }
        )
    return rows


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmm-dir", type=Path, default=Path("data/tasks/entity_hmm"))
    parser.add_argument(
        "--broad-dir", type=Path, default=Path("data/tasks/entity_release_bmes")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/tasks/entity_release_combined")
    )
    parser.add_argument(
        "--labels-manifest",
        type=Path,
        default=Path("data/tasks/entity_release_labels/manifest.json"),
    )
    args = parser.parse_args()

    labels_manifest = json.loads(args.labels_manifest.read_text(encoding="utf-8"))
    labels = labels_manifest["labels"]
    if sha256(Path(labels["path"])) != labels["sha256"]:
        raise ValueError("release labels manifest hash mismatch")
    broad_manifest_path = args.broad_dir / "manifest.json"
    broad_manifest = json.loads(broad_manifest_path.read_text(encoding="utf-8"))
    if not any(
        item["path"] == labels["path"] and item["sha256"] == labels["sha256"]
        for item in broad_manifest["inputs"]
    ):
        raise ValueError("BMES manifest does not reference the release labels")

    seen_texts: dict[str, str] = {}
    outputs = []
    stats = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        hmm_path = args.hmm_dir / f"{split}.tsv"
        broad_path = args.broad_dir / f"{split}.jsonl"
        rows = load_hmm(hmm_path, split)
        for row in rows:
            seen_texts.setdefault(row["text"], split)
        added_broad = 0
        for row in load_jsonl(broad_path):
            previous = seen_texts.get(row["text"])
            if previous is not None:
                if previous != split:
                    raise ValueError(f"cross-split duplicate: {row['text']!r}")
                continue
            seen_texts[row["text"]] = split
            rows.append(row)
            added_broad += 1
        out_path = args.out_dir / f"{split}.jsonl"
        out_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        outputs.append({"path": out_path.as_posix(), "sha256": sha256(out_path)})
        stats[split] = {"sentences": len(rows), "broad_sentences": added_broad}

    manifest = {
        "schema": "nexaloid.entity_release_combined.v1",
        "license_spdx": "Apache-2.0",
        "third_party": ["THUOCL MIT", "JD comments Apache-2.0"],
        "inputs": [
            {
                "path": (args.hmm_dir / "data_manifest.json").as_posix(),
                "sha256": sha256(args.hmm_dir / "data_manifest.json"),
            },
            {
                "path": broad_manifest_path.as_posix(),
                "sha256": sha256(broad_manifest_path),
            },
            {
                "path": args.labels_manifest.as_posix(),
                "sha256": sha256(args.labels_manifest),
            },
        ],
        "outputs": outputs,
        "splits": stats,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for split in SPLITS:
        print(f"{split}\t{stats[split]['sentences']}\t+{stats[split]['broad_sentences']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
