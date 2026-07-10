from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from export_nxbmes import FEATURE, HEADER, MAGIC, fnv1a64


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
    args = parser.parse_args()

    payload = args.artifact.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = args.artifact.with_suffix(args.artifact.suffix + ".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    assert digest == expected
    header_size = struct.calcsize(HEADER)
    magic, version, count, max_len, general_len, entity_len, record_size = struct.unpack_from(
        HEADER, payload, 0
    )
    assert magic == MAGIC and version == 1
    assert count > 70_000 and max_len == 12
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
    assert offset == len(payload)
    assert general_count > 10_000 and entity_count >= 100

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"] == digest
    assert manifest["feature_count"] == count
    assert manifest["general_lexicon_size"] == general_count
    assert manifest["entity_lexicon_size"] == entity_count
    print("nxbmes_ok")
    print(f"sha256\t{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
