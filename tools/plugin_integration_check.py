from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def plugin_name() -> str:
    if sys.platform == "win32":
        return "nexaloid_plugin_entity_bmes.dll"
    if sys.platform == "darwin":
        return "nexaloid_plugin_entity_bmes.dylib"
    return "nexaloid_plugin_entity_bmes.so"


def ascii_boundary_ok(text: str, start: int, end: int) -> bool:
    def is_ascii_alnum(char: str) -> bool:
        return char.isascii() and char.isalnum()

    return not (
        start > 0 and is_ascii_alnum(text[start - 1]) and is_ascii_alnum(text[start])
    ) and not (
        end < len(text)
        and is_ascii_alnum(text[end - 1])
        and is_ascii_alnum(text[end])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nexaloid-dir", type=Path, default=ROOT.parent / "Nexaloid")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "data" / "releases" / "bmes" / "entity_bmes_perceptron.nxbmes",
    )
    args = parser.parse_args()

    python_src = args.nexaloid_dir / "bindings" / "python" / "src"
    sys.path.insert(0, str(python_src))
    if sys.platform == "win32":
        os.environ["NEXALOID_LIB"] = str(
            args.nexaloid_dir / "core" / "zig-out" / "bin" / "nexaloid.dll"
        )
    elif sys.platform == "darwin":
        os.environ["NEXALOID_LIB"] = str(
            args.nexaloid_dir / "core" / "zig-out" / "lib" / "libnexaloid.dylib"
        )
    else:
        os.environ["NEXALOID_LIB"] = str(
            args.nexaloid_dir / "core" / "zig-out" / "lib" / "libnexaloid.so"
        )

    from nexaloid import Tokenizer
    from lexicon import load_lexicon
    from train_entity_llm_perceptron import decode

    with tempfile.TemporaryDirectory() as tmp:
        plugin = Path(tmp) / plugin_name()
        subprocess.run(
            [
                "zig",
                "build-lib",
                "-dynamic",
                "-lc",
                f"-femit-bin={plugin}",
                str(ROOT / "plugins" / "entity_bmes_plugin.zig"),
            ],
            check=True,
        )
        tokenizer = Tokenizer(dict_path=Path(tmp) / "missing.tsv")
        try:
            tokenizer.load_plugin(plugin, json.dumps({"artifact": str(args.artifact)}))
            for text, expected in (
                ("患者服用阿司匹林治疗冠心病。", "阿司匹林"),
                ("苹果公司发布iPhone 16 Pro", "iPhone 16 Pro"),
                ("该系统采用Transformer和CUDA加速。", "Transformer"),
            ):
                entities = [
                    token.text
                    for token in tokenizer.tokenize(text)
                    if token.source == "plugin"
                ]
                assert expected in entities, (text, entities)

            model = json.loads(
                (ROOT / "data" / "tasks" / "entity_llm_bmes" / "entity_llm_perceptron_generic.json").read_text(
                    encoding="utf-8"
                )
            )
            gazetteer = model["gazetteer"]
            lexicon = {
                word
                for word in load_lexicon(ROOT / gazetteer["path"])
                if 2 <= len(word) <= gazetteer["max_word_len"]
            }
            entity_lexicon = set(gazetteer["training_entity_words"])
            for split in ("dev", "test"):
                path = ROOT / "data" / "tasks" / "entity_llm_bmes" / f"{split}.jsonl"
                for raw in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(raw)
                    text = row["text"]
                    expected = {
                        (start, end)
                        for start, end, _ in decode(
                            model["weights"],
                            text,
                            True,
                            lexicon,
                            entity_lexicon,
                            gazetteer["max_word_len"],
                        )
                        if end - start >= 2 and ascii_boundary_ok(text, start, end)
                    }
                    actual = {
                        (token.start_char, token.end_char)
                        for token in tokenizer.tokenize(text)
                        if token.source == "plugin"
                    }
                    assert actual == expected, (split, text, expected, actual)
        finally:
            tokenizer.close()

        bad_artifact = Path(tmp) / "bad.nxbmes"
        bad_artifact.write_bytes(args.artifact.read_bytes()[:64])
        tokenizer = Tokenizer(dict_path=Path(tmp) / "missing.tsv")
        try:
            try:
                tokenizer.load_plugin(plugin, str(bad_artifact))
            except Exception:
                pass
            else:
                raise AssertionError("truncated artifact was accepted")
        finally:
            tokenizer.close()
    print("plugin_integration_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
