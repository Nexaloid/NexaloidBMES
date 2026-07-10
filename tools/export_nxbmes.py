from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path

from lexicon import load_lexicon
from nxdict_builder import build as build_nxdict


MAGIC = b"NXBMES01"
HEADER = "<8sIIIIII"
FEATURE = "<Q5fI"
STATES = ("O", "B", "M", "E", "S")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64(value: str) -> int:
    out = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        out ^= byte
        out = (out * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return out


def nxdict_blob(words: list[str], score: float) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "words.tsv"
        artifact = root / "words.nxdict"
        source.write_text(
            "".join(f"{word}\t{score}\n" for word in words),
            encoding="utf-8",
            newline="\n",
        )
        build_nxdict(source, artifact)
        blob = artifact.read_bytes()
        return blob + b"\0" * (-len(blob) % 8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/tasks/entity_llm_bmes/entity_llm_perceptron_generic.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/releases/bmes/entity_bmes_perceptron.nxbmes"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/releases/bmes/entity_bmes_perceptron.manifest.json"),
    )
    parser.add_argument(
        "--distribution",
        choices=("internal", "public"),
        default="internal",
    )
    parser.add_argument("--license-spdx", default="NOASSERTION")
    args = parser.parse_args()

    if args.distribution == "public" and args.license_spdx.strip() in {"", "NOASSERTION"}:
        raise SystemExit("public artifacts require --license-spdx")

    model = json.loads(args.model.read_text(encoding="utf-8"))
    if model.get("generic") is not True or model.get("states") != list(STATES):
        raise SystemExit("expected a generic O/B/M/E/S model")
    gazetteer = model.get("gazetteer") or {}
    max_word_len = int(gazetteer["max_word_len"])
    lexicon_path = Path(gazetteer["path"])
    if sha256(lexicon_path) != gazetteer["sha256"]:
        raise SystemExit("gazetteer hash mismatch")
    general_words = sorted(
        word for word in load_lexicon(lexicon_path) if 2 <= len(word) <= max_word_len
    )
    entity_words = sorted(set(gazetteer.get("training_entity_words", [])))

    records = []
    seen_hashes: dict[int, str] = {}
    for name, row in model["weights"].items():
        feature_hash = fnv1a64(name)
        collision = seen_hashes.setdefault(feature_hash, name)
        if collision != name:
            raise SystemExit(f"feature hash collision: {collision!r} and {name!r}")
        records.append(
            (
                feature_hash,
                *(float(row.get(state, 0.0)) for state in STATES),
                0,
            )
        )
    records.sort(key=lambda item: item[0])
    general_blob = nxdict_blob(general_words, 1.0)
    entity_blob = nxdict_blob(entity_words, 2.0)
    payload = bytearray(
        struct.pack(
            HEADER,
            MAGIC,
            1,
            len(records),
            max_word_len,
            len(general_blob),
            len(entity_blob),
            struct.calcsize(FEATURE),
        )
    )
    for record in records:
        payload += struct.pack(FEATURE, *record)
    payload += general_blob
    payload += entity_blob

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{digest}  {args.out.name}\n", encoding="utf-8", newline="\n"
    )
    manifest = {
        "schema": "nexaloid.bmes_manifest.v1",
        "artifact": args.out.as_posix(),
        "artifact_format": "nxbmes.v1",
        "artifact_sha256": digest,
        "feature_hash": "fnv1a64",
        "feature_count": len(records),
        "feature_record_size": struct.calcsize(FEATURE),
        "states": list(STATES),
        "distribution": {
            "scope": args.distribution,
            "license_spdx": args.license_spdx,
        },
        "max_word_len": max_word_len,
        "general_lexicon_size": len(general_words),
        "entity_lexicon_size": len(entity_words),
        "inputs": {
            "model": args.model.as_posix(),
            "model_sha256": sha256(args.model),
            "gazetteer": lexicon_path.as_posix(),
            "gazetteer_sha256": sha256(lexicon_path),
        },
        "quality": {
            split: model["metrics"][split]["overall"] for split in ("dev", "test")
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"features\t{len(records)}")
    print(f"lexicons\t{len(general_words)}\t{len(entity_words)}")
    print(f"sha256\t{digest}")
    print(f"wrote\t{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
