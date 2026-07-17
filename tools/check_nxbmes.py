from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from export_nxbmes import (
    FEATURE,
    HEADER,
    HEADER_V1,
    MAGIC,
    NGRAM_WEIGHTS,
    WEAK_PAIR_CONTEXT_DELTA,
    fnv1a64,
    gate_hash,
)


def check_nxdict(blob: bytes) -> int:
    if len(blob) < 20 or blob[:8] != b"NXDICT1\0":
        raise AssertionError("invalid embedded NXDICT")
    code_count, state_count, entry_count = struct.unpack_from("<III", blob, 8)
    minimum = 20 + code_count * 4 + state_count * 16
    assert entry_count > 0 and state_count > 0 and len(blob) >= minimum
    return entry_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/releases/bmes/entity_bmes_perceptron.nxbmes"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/releases/bmes/entity_bmes_perceptron.manifest.json"),
    )
    parser.add_argument("--min-feature-count", type=int, default=70_000)
    parser.add_argument("--min-general-count", type=int, default=10_000)
    parser.add_argument("--min-entity-count", type=int, default=100)
    args = parser.parse_args()

    payload = args.artifact.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = args.artifact.with_suffix(args.artifact.suffix + ".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    assert digest == expected
    magic, version = struct.unpack_from("<8sI", payload, 0)
    assert magic == MAGIC and version in {1, 2}
    if version == 1:
        header_size = struct.calcsize(HEADER_V1)
        _, _, count, max_len, general_len, entity_len, record_size = struct.unpack_from(
            HEADER_V1, payload, 0
        )
        gate_capacity = gate_count = 0
        entity_gate_count = entity_gate_min_chars = 0
        pair_capacity = pair_count = triple_capacity = triple_count = 0
    else:
        header_size = struct.calcsize(HEADER)
        (
            _,
            _,
            count,
            max_len,
            general_len,
            entity_len,
            record_size,
            gate_capacity,
            gate_count,
            entity_gate_count,
            entity_gate_min_chars,
            pair_capacity,
            pair_count,
            triple_capacity,
            triple_count,
        ) = struct.unpack_from(HEADER, payload, 0)
    assert count >= args.min_feature_count and max_len == 12
    assert record_size == struct.calcsize(FEATURE) == 32
    offset = header_size
    hashes = []
    wanted = {fnv1a64(name) for name in ("bias", "T=<START>", "lx=B", "ex=B")}
    found = set()
    for _ in range(count):
        record = struct.unpack_from(FEATURE, payload, offset)
        offset += record_size
        hashes.append(record[0])
        found.add(record[0])
        assert all(math.isfinite(value) for value in record[1:6])
    assert hashes == sorted(hashes) and len(hashes) == len(set(hashes))
    assert wanted <= found
    general_count = check_nxdict(payload[offset : offset + general_len])
    offset += general_len
    entity_count = check_nxdict(payload[offset : offset + entity_len])
    offset += entity_len
    if version == 2:
        assert gate_capacity >= 2 and gate_capacity & (gate_capacity - 1) == 0
        gate_slots = struct.unpack_from(f"<{gate_capacity}Q", payload, offset)
        offset += gate_capacity * 8
        assert sum(slot != 0 for slot in gate_slots) == gate_count
        for key in gate_slots:
            if not key:
                continue
            index = gate_hash(key) & (gate_capacity - 1)
            while gate_slots[index] != key:
                assert gate_slots[index] != 0
                index = (index + 1) & (gate_capacity - 1)
        ngram_tables = []
        for capacity, ngram_count in ((pair_capacity, pair_count), (triple_capacity, triple_count)):
            assert capacity >= 2 and capacity & (capacity - 1) == 0 and ngram_count <= capacity // 2
            table = struct.unpack_from(f"<{capacity}Q", payload, offset)
            offset += capacity * 8
            ngram_tables.append((table, ngram_count, capacity))
        for table, ngram_count, capacity in ngram_tables:
            assert sum(key != 0 for key in table) == ngram_count
            for key in table:
                if not key:
                    continue
                index = gate_hash(key) & (capacity - 1)
                while table[index] != key:
                    assert table[index] != 0
                    index = (index + 1) & (capacity - 1)
        ngram_weight_size = struct.calcsize(NGRAM_WEIGHTS)
        for capacity in (pair_capacity, triple_capacity):
            for index in range(capacity):
                weights = struct.unpack_from(NGRAM_WEIGHTS, payload, offset + index * ngram_weight_size)
                assert all(math.isfinite(value) for value in weights)
            offset += capacity * ngram_weight_size
        assert entity_gate_count > 0 and entity_gate_min_chars >= 2
        failure = struct.unpack_from(f"<{entity_gate_count}I", payload, offset)
        offset += entity_gate_count * 4
        outputs = payload[offset : offset + entity_gate_count]
        offset += entity_gate_count
        assert len(outputs) == entity_gate_count
        assert all(value < entity_gate_count for value in failure)
        assert set(outputs) <= {0, 1}
    assert offset == len(payload)
    assert general_count >= args.min_general_count
    assert entity_count >= args.min_entity_count

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"] == digest
    assert manifest["feature_count"] == count
    assert manifest["general_lexicon_size"] == general_count
    assert manifest["entity_lexicon_size"] == entity_count
    assert manifest["artifact_format"] == f"nxbmes.v{version}"
    if version == 2:
        gate = manifest["proposal_gate"]
        assert gate["capacity"] == gate_capacity
        assert gate["pair_count"] + gate["triple_count"] == gate_count
        assert gate["strategy"] == "context-refined"
        assert gate["weak_pair_context_delta"] == WEAK_PAIR_CONTEXT_DELTA
        entity_gate = manifest["entity_gate"]
        assert entity_gate["state_count"] == entity_gate_count
        assert entity_gate["min_chars"] == entity_gate_min_chars
        ngrams = manifest["ngram_tables"]
        assert ngrams["weight_record_size"] == ngram_weight_size
        assert ngrams["pair_capacity"] == pair_capacity and ngrams["pair_count"] == pair_count
        assert ngrams["triple_capacity"] == triple_capacity and ngrams["triple_count"] == triple_count
    distribution = manifest["distribution"]
    assert distribution["scope"] in {"internal", "public"}
    if distribution["scope"] == "public":
        assert distribution["license_spdx"] != "NOASSERTION"
    print("nxbmes_ok")
    print(f"sha256\t{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
