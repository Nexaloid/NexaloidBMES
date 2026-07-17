from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIN_MARGIN = 35.0
HARD_BOUNDARY_FEATURE = "__runtime_hard_boundary__"
ALLOWED_CONNECTORS = {"·", "-", "‐", "‑", "&", "/"}


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


def is_entity_body(char: str) -> bool:
    cp = ord(char)
    return (
        0x3400 <= cp <= 0x9FFF
        or 0x20000 <= cp <= 0x2EBEF
        or char.isdigit()
        or "A" <= char <= "Z"
        or "a" <= char <= "z"
        or 0x00C0 <= cp <= 0x02AF
        or 0x0370 <= cp <= 0x052F
        or 0xFF21 <= cp <= 0xFF3A
        or 0xFF41 <= cp <= 0xFF5A
    )


def is_hard_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char in ALLOWED_CONNECTORS:
        return (
            index == 0
            or index + 1 == len(text)
            or not is_entity_body(text[index - 1])
            or not is_entity_body(text[index + 1])
        )
    return not is_entity_body(char)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nexaloid-dir", type=Path, default=ROOT.parent / "Nexaloid")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "data" / "releases" / "bmes-public" / "entity_bmes_perceptron.nxbmes",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "data" / "tasks" / "entity_release_combined" / "entity_release_perceptron.json",
    )
    args = parser.parse_args()

    for text, index in (("甲·乙", 1), ("A-B", 1), ("A‐B", 1), ("A‑B", 1), ("A&B", 1), ("A/B", 1)):
        assert not is_hard_boundary(text, index)
    for text, index in (("甲 乙", 1), ("甲：乙", 1), ("《甲》", 0), ("《甲》", 2), ("甲- 乙", 1)):
        assert is_hard_boundary(text, index)

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
    from train_entity_llm_perceptron import (
        decode_tags_with_features,
        emission_scores,
        sentence_features,
        tags_to_spans,
    )

    plugin_source = ROOT / "plugins" / "entity_bmes_plugin.zig"
    assert plugin_source.read_bytes() == (
        args.nexaloid_dir / "tools" / "entity_bmes_plugin.zig"
    ).read_bytes()

    with tempfile.TemporaryDirectory() as tmp:
        plugin = Path(tmp) / plugin_name()
        subprocess.run(
            [
                "zig",
                "build-lib",
                "-O",
                "ReleaseSafe",
                "-mcpu",
                "baseline",
                "-dynamic",
                "-lc",
                f"-femit-bin={plugin}",
                str(plugin_source),
            ],
            check=True,
        )
        tokenizer = Tokenizer(dict_path=Path(tmp) / "missing.tsv")
        try:
            tokenizer.load_plugin(plugin, json.dumps({"artifact": str(args.artifact)}))
            for text, expected in (
                ("团队计划前往云海数据研究院开展调研。", "云海数据研究院"),
                ("欧盟委员会发布公告。", "欧盟委员会"),
                ("韩国财政部公布数据。", "韩国财政部"),
                ("美国国务院发表声明。", "美国国务院"),
            ):
                entities = [
                    token.text
                    for token in tokenizer.tokenize(text)
                    if token.source == "plugin"
                ]
                assert expected in entities, (text, entities)

            for text, rejected in (
                ("央行票据 支持财政部", "央行票据 支持财政部"),
                ("事关国民经济", "事关国民经济"),
                ("国家开发银行湖南省", "国家开发银行湖南省"),
                ("超卓航科：控股股东", "超卓航科：控股股东"),
                ("公司上涨实现季度增长", "公司上涨实现"),
                ("限制数据中心", "限制数据中心"),
                ("随着数据中心", "随着数据中心"),
            ):
                entities = [token for token in tokenizer.tokenize(text) if token.source == "plugin"]
                assert rejected not in {token.text for token in entities}, (text, entities)
                assert all(token.score <= 400.0 for token in entities)
            for word in ("公司", "上涨", "实现", "季度"):
                assert not [token for token in tokenizer.tokenize(word) if token.source == "plugin"], word

            model = json.loads(args.model.read_text(encoding="utf-8"))
            runtime_weights = dict(model["weights"])
            runtime_weights[HARD_BOUNDARY_FEATURE] = {
                state: -float("inf") for state in ("B", "M", "E", "S")
            }
            gazetteer = model["gazetteer"]
            lexicon = {
                word
                for word in load_lexicon(ROOT / gazetteer["path"])
                if 2 <= len(word) <= gazetteer["max_word_len"]
            }
            entity_lexicon = set(gazetteer["training_entity_words"])
            for split in ("dev", "test"):
                true_positive = false_positive = false_negative = 0
                path = args.model.parent / f"{split}.jsonl"
                for raw in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(raw)
                    text = row["text"]
                    gold = {(start, end) for start, end, _ in row["spans"]}
                    features = sentence_features(text, lexicon, entity_lexicon, gazetteer["max_word_len"])
                    features = [
                        values + ((HARD_BOUNDARY_FEATURE,) if is_hard_boundary(text, index) else ())
                        for index, values in enumerate(features)
                    ]
                    tags = decode_tags_with_features(runtime_weights, features, True)
                    expected = set()
                    for start, end, _ in tags_to_spans(tags, True):
                        if (
                            end - start < 2
                            or not ascii_boundary_ok(text, start, end)
                            or text[start:end] in lexicon
                        ):
                            continue
                        margins = []
                        for index in range(start, end):
                            scores = emission_scores(runtime_weights, features[index], tuple(model["states"]))
                            margins.append(
                                scores[tags[index]]
                                - max(score for state, score in scores.items() if state != tags[index])
                            )
                        if sum(margins) / len(margins) >= DEFAULT_MIN_MARGIN:
                            expected.add((start, end))
                    actual = {
                        (token.start_char, token.end_char)
                        for token in tokenizer.tokenize(text)
                        if token.source == "plugin"
                    }
                    assert actual <= expected, (split, text, expected, actual)
                    true_positive += len(actual & gold)
                    false_positive += len(actual - gold)
                    false_negative += len(gold - actual)
                precision = true_positive / (true_positive + false_positive)
                assert precision >= 0.90, (split, precision)
                assert true_positive >= 700, (split, true_positive)
                print(
                    f"{split}_runtime\ttp={true_positive}\tfp={false_positive}\tfn={false_negative}"
                    f"\tprecision={precision:.6f}"
                )
        finally:
            tokenizer.close()

        artifact_bytes = args.artifact.read_bytes()
        misaligned = bytearray(artifact_bytes)
        struct.pack_into("<I", misaligned, 20, struct.unpack_from("<I", misaligned, 20)[0] + 1)
        feature_count = struct.unpack_from("<I", artifact_bytes, 12)[0]
        general_len = struct.unpack_from("<I", artifact_bytes, 20)[0]
        entity_offset = struct.calcsize("<8s14I") + feature_count * 32 + general_len
        code_count, state_count = struct.unpack_from("<2I", artifact_bytes, entity_offset + 8)
        check_offset = entity_offset + 20 + (code_count + state_count) * 4
        checks = struct.unpack_from(f"<{state_count}I", artifact_bytes, check_offset)
        check_index = next(index for index, value in enumerate(checks) if index and value)
        bad_parent = bytearray(artifact_bytes)
        struct.pack_into("<I", bad_parent, check_offset + check_index * 4, state_count + 1)
        nan_weight = bytearray(artifact_bytes)
        struct.pack_into("<f", nan_weight, struct.calcsize("<8s14I") + 8, float("nan"))
        for name, payload in (
            ("truncated", artifact_bytes[:64]),
            ("misaligned", misaligned),
            ("bad-dat-parent", bad_parent),
            ("nan-weight", nan_weight),
        ):
            bad_artifact = Path(tmp) / f"{name}.nxbmes"
            bad_artifact.write_bytes(payload)
            tokenizer = Tokenizer(dict_path=Path(tmp) / "missing.tsv")
            try:
                try:
                    tokenizer.load_plugin(plugin, str(bad_artifact))
                except Exception:
                    pass
                else:
                    raise AssertionError(f"{name} artifact was accepted")
            finally:
                tokenizer.close()
    print("plugin_integration_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
