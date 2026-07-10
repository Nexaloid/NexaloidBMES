from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "entity_bmes_perceptron.nxbmes",
    "entity_bmes_perceptron.nxbmes.sha256",
    "entity_bmes_perceptron.manifest.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nexaloid-dir", type=Path, default=ROOT.parent / "Nexaloid")
    args = parser.parse_args()
    source = ROOT / "data" / "releases" / "bmes"
    target = args.nexaloid_dir / "data" / "entity"
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(source / name, target / name)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
