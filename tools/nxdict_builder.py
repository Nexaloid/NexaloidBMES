from __future__ import annotations

import argparse
import struct
from collections import deque
from pathlib import Path


MAGIC = b"NXDICT1\0"


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            word = parts[0]
            try:
                score = float(parts[1]) if len(parts) > 1 else 1.0
            except ValueError:
                score = 1.0
            data = word.encode("utf-8")
            if len(data) <= 0xFFFF:
                yield word, data, score


def build_trie(rows):
    nodes = [{"word_id": 0, "score": 0.0, "children": {}}]
    entries = []
    for word_id, (word, data, score) in enumerate(rows, 1):
        node = 0
        for ch in word:
            children = nodes[node]["children"]
            cp = ord(ch)
            if cp not in children:
                children[cp] = len(nodes)
                nodes.append({"word_id": 0, "score": 0.0, "children": {}})
            node = children[cp]
        nodes[node]["word_id"] = word_id
        nodes[node]["score"] = score
        entries.append((data, score))
    return nodes, entries


def find_base(child_codes, used, start):
    base = start
    while True:
        for code in child_codes:
            pos = base + code
            if pos < len(used) and used[pos]:
                break
        else:
            return base
        base += 1


def ensure_len(items, size, fill):
    if len(items) < size:
        items.extend([fill] * (size - len(items)))


def build_dat(nodes):
    codepoints = sorted({cp for node in nodes for cp in node["children"]})
    code_id = {cp: i + 1 for i, cp in enumerate(codepoints)}
    trie_to_dat = {0: 0}
    base = [0]
    check = [0]
    used = [True]
    q = deque([0])
    next_base = 1

    while q:
        trie_node = q.popleft()
        dat_state = trie_to_dat[trie_node]
        children = nodes[trie_node]["children"]
        if not children:
            continue
        child_codes = sorted(code_id[cp] for cp in children)
        b = find_base(child_codes, used, next_base)
        next_base = b
        ensure_len(base, dat_state + 1, 0)
        base[dat_state] = b
        for cp, child in children.items():
            target = b + code_id[cp]
            ensure_len(base, target + 1, 0)
            ensure_len(check, target + 1, 0)
            ensure_len(used, target + 1, False)
            used[target] = True
            check[target] = dat_state + 1
            trie_to_dat[child] = target
            q.append(child)

    state_count = len(base)
    meta = [(0, 0.0)] * state_count
    for trie_node, dat_state in trie_to_dat.items():
        node = nodes[trie_node]
        meta[dat_state] = (node["word_id"], node["score"])
    return codepoints, base, check, meta


def build(in_path: Path, out_path: Path) -> tuple[int, int, int]:
    rows = list(read_rows(in_path))
    nodes, entries = build_trie(rows)
    codepoints, base, check, meta = build_dat(nodes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", len(codepoints), len(base), len(entries)))
        for cp in codepoints:
            f.write(struct.pack("<I", cp))
        for item in base:
            f.write(struct.pack("<I", item))
        for item in check:
            f.write(struct.pack("<I", item))
        for word_id, score in meta:
            f.write(struct.pack("<If", word_id, score))
        for data, score in entries:
            f.write(struct.pack("<HHf", len(data), 0, score))
            f.write(data)
    return len(entries), len(base), len(codepoints)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    entries, states, codes = build(args.input, args.output)
    print(f"wrote\tentries={entries}\tstates={states}\tcodes={codes}\t{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
