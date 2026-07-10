from __future__ import annotations

from collections import Counter
from pathlib import Path


def load_lexicon(path: Path) -> set[str]:
    words: Counter[str] = Counter()
    split_compounds: Counter[str] = Counter()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row = line.split()
        words.update(row)
        split_compounds.update(left + right for left, right in zip(row, row[1:]))
    return {
        word
        for word, count in words.items()
        if split_compounds.get(word, 0) < count
    }
