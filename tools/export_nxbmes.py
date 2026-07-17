from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import struct
import tempfile
from pathlib import Path

from lexicon import load_lexicon
from nxdict_builder import build as build_nxdict


MAGIC = b"NXBMES01"
HEADER_V1 = "<8sIIIIII"
HEADER = "<8s14I"
FEATURE = "<Q5fI"
NGRAM_WEIGHTS = "<10f"
STATES = ("O", "B", "M", "E", "S")
GATE_ARITY_BIT = 1 << 63
WEAK_PAIR_CONTEXT_DELTA = 0.4
GATE_BOUNDARIES = {
    "<BOS1>": 0x110000,
    "<BOS2>": 0x110001,
    "<EOS1>": 0x110002,
    "<EOS2>": 0x110003,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64(value: str) -> int:
    out = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        out ^= byte
        out = (out * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return out


def gate_hash(value: int) -> int:
    value ^= value >> 32
    value = (value * 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 32)


def gate_key(codepoints: tuple[int, ...]) -> int:
    if len(codepoints) == 2:
        raw = (codepoints[0] << 21) | codepoints[1]
    elif len(codepoints) == 3:
        raw = GATE_ARITY_BIT | (codepoints[0] << 42) | (codepoints[1] << 21) | codepoints[2]
    else:
        raise ValueError("proposal gate keys must contain two or three codepoints")
    return raw + 1


def gate_codepoints(value: str, expected_length: int) -> tuple[int, ...] | None:
    out = []
    while value and len(out) < expected_length:
        boundary = next((item for item in GATE_BOUNDARIES if value.startswith(item)), None)
        if boundary:
            out.append(GATE_BOUNDARIES[boundary])
            value = value[len(boundary) :]
        else:
            out.append(ord(value[0]))
            value = value[1:]
    return tuple(out) if not value and len(out) == expected_length else None


def build_proposal_gate(model: dict, min_delta: float) -> tuple[list[int], int, int]:
    pair_scores: dict[tuple[int, ...], float] = {}
    triple_scores: dict[tuple[int, ...], float] = {}
    prefixes = {
        "c-1c0=": (pair_scores, 2),
        "c0c+1=": (pair_scores, 2),
        "c-2c-1c0=": (triple_scores, 3),
        "c0c+1c+2=": (triple_scores, 3),
    }
    for name, row in model["weights"].items():
        outside = float(row.get("O", 0.0))
        delta = max(float(row.get("B", 0.0)), float(row.get("S", 0.0))) - outside
        for prefix, (scores, arity) in prefixes.items():
            if not name.startswith(prefix):
                continue
            codepoints = gate_codepoints(name[len(prefix) :], arity)
            if codepoints:
                scores[codepoints] = max(scores.get(codepoints, -float("inf")), delta)
            break

    pairs = {key for key, delta in pair_scores.items() if delta >= min_delta}
    triples = {key for key, delta in triple_scores.items() if delta >= min_delta}
    covered_pairs = {pair for triple in triples for pair in (triple[:2], triple[1:])}
    pairs -= covered_pairs
    for pair in sorted(pairs):
        if pair_scores[pair] >= min_delta + WEAK_PAIR_CONTEXT_DELTA:
            continue
        contexts = [
            (delta, triple)
            for triple, delta in triple_scores.items()
            if delta > 0 and pair in (triple[:2], triple[1:])
        ]
        if contexts:
            triples.add(max(contexts)[1])
            pairs.remove(pair)

    encoded = sorted(gate_key(key) for key in pairs | triples)
    capacity = 1
    while capacity < max(2, len(encoded) * 2):
        capacity *= 2
    slots = [0] * capacity
    for key in encoded:
        index = gate_hash(key) & (capacity - 1)
        while slots[index]:
            index = (index + 1) & (capacity - 1)
        slots[index] = key
    return slots, len(pairs), len(triples)


def build_ngram_table(
    model: dict,
    prefixes: tuple[str, str],
    arity: int,
) -> tuple[list[tuple], int]:
    values: dict[int, list[list[float]]] = {}
    for name, row in model["weights"].items():
        for side, prefix in enumerate(prefixes):
            if not name.startswith(prefix):
                continue
            codepoints = gate_codepoints(name[len(prefix) :], arity)
            if codepoints:
                weights = values.setdefault(
                    gate_key(codepoints),
                    [[0.0] * len(STATES), [0.0] * len(STATES)],
                )
                weights[side] = [float(row.get(state, 0.0)) for state in STATES]
            break
    capacity = 1
    while capacity < max(2, len(values) * 2):
        capacity *= 2
    empty = (0, *([0.0] * (len(STATES) * 2)))
    slots = [empty] * capacity
    for key, weights in values.items():
        index = gate_hash(key) & (capacity - 1)
        while slots[index][0]:
            index = (index + 1) & (capacity - 1)
        slots[index] = (key, *weights[0], *weights[1])
    return slots, len(values)


def build_entity_gate(blob: bytes, words: list[str], min_chars: int) -> tuple[list[int], bytes]:
    code_count, state_count, _ = struct.unpack_from("<III", blob, 8)
    offset = 20
    codepoints = struct.unpack_from(f"<{code_count}I", blob, offset)
    offset += code_count * 4
    base = struct.unpack_from(f"<{state_count}I", blob, offset)
    offset += state_count * 4
    check = struct.unpack_from(f"<{state_count}I", blob, offset)
    offset += state_count * 4
    nodes_offset = offset
    word_ids = [struct.unpack_from("<I", blob, nodes_offset + index * 8)[0] for index in range(state_count)]
    code_ids = {codepoint: index + 1 for index, codepoint in enumerate(codepoints)}
    children: list[dict[int, int]] = [dict() for _ in range(state_count)]
    for child in range(1, state_count):
        if not check[child]:
            continue
        parent = check[child] - 1
        code_id = child - base[parent]
        if not 1 <= code_id <= code_count:
            raise ValueError("invalid entity DAT transition")
        children[parent][codepoints[code_id - 1]] = child

    terminal_nodes = set()
    for word in words:
        state = 0
        for char in word:
            state = children[state][ord(char)]
        terminal_nodes.add(state)

    failure = [0] * state_count
    output = bytearray(state_count)
    depth = [0] * state_count
    queue = deque(children[0].values())
    for child in queue:
        depth[child] = 1
    while queue:
        parent = queue.popleft()
        output[parent] = int((parent in terminal_nodes and depth[parent] >= min_chars) or output[failure[parent]])
        for codepoint, child in children[parent].items():
            depth[child] = depth[parent] + 1
            fallback = failure[parent]
            while fallback and codepoint not in children[fallback]:
                fallback = failure[fallback]
            failure[child] = children[fallback].get(codepoint, 0)
            queue.append(child)
    if any(codepoint not in code_ids for edges in children for codepoint in edges):
        raise ValueError("entity DAT codepoint table is inconsistent")
    return failure, bytes(output)


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
        build_nxdict(source, artifact, allow_hash_words=True)
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
    parser.add_argument("--proposal-min-delta", type=float, default=7.0)
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
    gate_slots, gate_pair_count, gate_triple_count = build_proposal_gate(model, args.proposal_min_delta)
    pair_table, pair_count = build_ngram_table(model, ("c-1c0=", "c0c+1="), 2)
    triple_table, triple_count = build_ngram_table(model, ("c-2c-1c0=", "c0c+1c+2="), 3)
    general_blob = nxdict_blob(general_words, 1.0)
    entity_blob = nxdict_blob(entity_words, 2.0)
    entity_gate_min_chars = 3
    entity_gate_words = entity_words
    entity_gate_failure, entity_gate_output = build_entity_gate(
        entity_blob, entity_gate_words, entity_gate_min_chars
    )
    payload = bytearray(
        struct.pack(
            HEADER,
            MAGIC,
            2,
            len(records),
            max_word_len,
            len(general_blob),
            len(entity_blob),
            struct.calcsize(FEATURE),
            len(gate_slots),
            gate_pair_count + gate_triple_count,
            len(entity_gate_failure),
            entity_gate_min_chars,
            len(pair_table),
            pair_count,
            len(triple_table),
            triple_count,
        )
    )
    for record in records:
        payload += struct.pack(FEATURE, *record)
    payload += general_blob
    payload += entity_blob
    payload += struct.pack(f"<{len(gate_slots)}Q", *gate_slots)
    for record in pair_table:
        payload += struct.pack("<Q", record[0])
    for record in triple_table:
        payload += struct.pack("<Q", record[0])
    for record in pair_table:
        payload += struct.pack(NGRAM_WEIGHTS, *record[1:])
    for record in triple_table:
        payload += struct.pack(NGRAM_WEIGHTS, *record[1:])
    payload += struct.pack(f"<{len(entity_gate_failure)}I", *entity_gate_failure)
    payload += entity_gate_output

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{digest}  {args.out.name}\n", encoding="utf-8", newline="\n"
    )
    manifest = {
        "schema": "nexaloid.bmes_manifest.v1",
        "artifact": args.out.as_posix(),
        "artifact_format": "nxbmes.v2",
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
        "proposal_gate": {
            "min_delta": args.proposal_min_delta,
            "strategy": "context-refined",
            "weak_pair_context_delta": WEAK_PAIR_CONTEXT_DELTA,
            "pair_count": gate_pair_count,
            "triple_count": gate_triple_count,
            "capacity": len(gate_slots),
            "hash": "multiply-xorshift64",
        },
        "entity_gate": {
            "algorithm": "aho-corasick",
            "state_count": len(entity_gate_failure),
            "min_chars": entity_gate_min_chars,
            "word_count": len(entity_gate_words),
        },
        "ngram_tables": {
            "weight_record_size": struct.calcsize(NGRAM_WEIGHTS),
            "pair_capacity": len(pair_table),
            "pair_count": pair_count,
            "triple_capacity": len(triple_table),
            "triple_count": triple_count,
        },
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
    print(f"proposal_gate\t{gate_pair_count}\t{gate_triple_count}\t{len(gate_slots)}")
    print(f"sha256\t{digest}")
    print(f"wrote\t{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
