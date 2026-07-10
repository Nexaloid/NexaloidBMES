from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from deepseek_client import call_deepseek


ENTITY_TYPES = (
    "PER",
    "LOC",
    "ORG",
    "FAC",
    "PRODUCT",
    "BRAND",
    "SOFTWARE",
    "HARDWARE",
    "MODEL",
    "DISEASE",
    "DRUG",
    "PATHOGEN",
    "GENE_PROTEIN",
    "CHEMICAL",
    "MATERIAL",
    "SPECIES",
    "EVENT",
    "WORK",
    "LAW",
    "TECH",
)
TYPE_ALIASES = {
    "PERSON": "PER",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "FACILITY": "FAC",
    "POLICY": "LAW",
    "STRATEGY": "LAW",
    "STANDARD": "LAW",
}
SKIPPED_TYPES = {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"}
SENTENCE_BREAK = re.compile(r"[\r\n]+|(?<=[。！？!?；;])")
TYPE_DEFINITIONS = {
    "PER": "具体人物、艺名、历史人物或明确的人物称谓名称",
    "LOC": "国家、行政区、自然地理区域及有专名的地点",
    "ORG": "公司、政府机构、学校、医院、社会组织、团队",
    "FAC": "机场、车站、道路、桥梁、建筑、场馆等设施专名",
    "PRODUCT": "完整产品名、设备名、车型、药械商品或命名商品",
    "BRAND": "独立出现的品牌名",
    "SOFTWARE": "软件、操作系统、应用、框架、数据库或云服务名称",
    "HARDWARE": "芯片、处理器、显卡、机器或硬件平台名称",
    "MODEL": "产品型号、版本号、芯片型号或设备代号",
    "DISEASE": "疾病、综合征及明确医学诊断名称",
    "DRUG": "药品、疫苗、生物制剂及明确药物名称",
    "PATHOGEN": "病毒、细菌、真菌、寄生虫等病原体名称",
    "GENE_PROTEIN": "基因、蛋白质、受体、酶等生命科学实体名称",
    "CHEMICAL": "化学元素、化合物、分子及明确化学物质名称",
    "MATERIAL": "金属、合金、高分子、半导体及命名材料名称",
    "SPECIES": "物种、品种、菌株、作物或动物植物名称",
    "EVENT": "会议、赛事、战争、事故、节庆或命名活动",
    "WORK": "书籍、影视、游戏、歌曲、报告、白皮书等作品或文献名",
    "LAW": "法律、法规、政策、标准、条约或正式制度文件名",
    "TECH": "算法、协议、编程语言、技术体系及命名技术概念",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentence_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_sentences(
    path: Path,
    scan_rows: int,
    limit: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    text_fields: tuple[str, ...] | None = None,
    category_pattern: str = "",
) -> list[dict[str, str]]:
    texts: list[tuple[str, str]] = []
    seen: set[str] = set()
    category_re = re.compile(category_pattern) if category_pattern else None

    def collect(row: dict, row_number: int) -> None:
        if category_re and not category_re.search(str(row.get("category", ""))):
            return
        source_group = row.get("id")
        if source_group in (None, ""):
            source_group = row_number
        group_id = sentence_id(f"{path.as_posix()}:{source_group}")
        fields = text_fields or (
            ("title", "desc", "answer")
            if path.suffix.lower() in (".json", ".jsonl")
            else ("headline", "content")
        )
        combined = "。".join(
            str(row.get(field, "")).strip()
            for field in fields
            if row.get(field) and str(row.get(field)).strip()
        )
        for raw in SENTENCE_BREAK.split(combined):
            text = " ".join(raw.split()).strip()
            if min_chars <= len(text) <= max_chars and text not in seen:
                seen.add(text)
                texts.append((text, group_id))

    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() in (".json", ".jsonl"):
            for row_number, raw in enumerate(handle, 1):
                collect(json.loads(raw), row_number)
                if row_number >= scan_rows:
                    break
        else:
            for row_number, row in enumerate(csv.DictReader(handle), 1):
                collect(row, row_number)
                if row_number >= scan_rows:
                    break
    random.Random(seed).shuffle(texts)
    return [
        {"id": sentence_id(text), "group_id": group_id, "text": text}
        for text, group_id in texts[:limit]
    ]


def prompt(rows: list[dict[str, str]]) -> str:
    return (
        "你是中文广义实体名词标注器。只输出合法 JSON，不要解释。\n"
        "输出格式：{\"rows\":[{\"id\":\"...\",\"text\":\"原文\","
        "\"entities\":[{\"start\":0,\"end\":2,\"type\":\"PER\",\"text\":\"张三\"}]}]}。\n"
        "start/end 使用 Python Unicode 字符下标，左闭右开，text 必须与原文切片完全一致。\n"
        "只标注有明确名称边界的专名或领域实体；普通名词、代词、泛称、修饰性短语不标。"
        "实体不得重叠；存在嵌套时选择语义最完整的跨度。品牌嵌入完整产品名时标 PRODUCT，"
        "品牌单独出现时标 BRAND。PER 只标个人，乐队、组合、团队标 ORG。"
        "日期、时间段和数量本身不是 EVENT；民族、方位或国别形容词不单独标 LOC。"
        "EVENT 只标有固定名称的活动，例如春运；羽泉等乐队组合标 ORG；"
        "丝绸之路等命名路线标 FAC；阿拉伯国家中的阿拉伯不标 LOC。"
        "DISEASE 不包含病原体，寨卡病毒应标 PATHOGEN。"
        "实体边界排除书名号、引号等外围标点。"
        "只能使用给定类型，不得创造新类型；战略、规划、政策和标准统一标 LAW。"
        "同一实体重复出现或作为定语出现时仍逐次标注，输出前再次扫描漏标。"
        "英文、数字和型号可以属于实体。\n"
        f"类型定义：{json.dumps(TYPE_DEFINITIONS, ensure_ascii=False)}\n"
        f"待标注数据：{json.dumps(rows, ensure_ascii=False)}"
    )


def review_prompt(rows: list[dict]) -> str:
    return (
        "你是中文广义实体标注的保守审核员。只输出合法 JSON，不要解释。\n"
        "输入包含初标 entities；请输出全部 rows 及纠正后的完整 entities。"
        "可以删除、改类型、改边界或补充高置信漏标。只能使用给定类型且实体不得重叠。\n"
        "重点删除：普通商品类别、普通名词、症状、日期数量、文件路径、章节标题、"
        "游戏职业/技能/宏、随机字母数字范围及缺乏名称证据的泛称。"
        "PRODUCT 必须是有名称的具体产品；MODEL 必须是正式型号；"
        "SOFTWARE 是命名软件，PDF/cookies/动态连接库等通用概念标 TECH 或不标；"
        "中药材按语境标 DRUG 或 SPECIES，不能标 CHEMICAL；"
        "DISEASE 不含症状，PER 不含职业、游戏类别或普通词；"
        "WORK 只标独立作品，不标章节标题；外围书名号和引号不进入跨度。\n"
        "以下泛称单独出现时不是实体：政府、警方、代表团、医院、文化馆、"
        "综合服务中心、高速、航空器、信息基础设施、冷链运输；只有带有唯一专名修饰、"
        "形成完整官方名称时才标。中阿、中德、中伊等双边关系简称及法方等国别立场词不标 LOC。"
        "职位不得进入 PER 边界，例如习近平总书记只标习近平。"
        "机构名、届次和会议名共同构成固定会议名称时，尽量标完整 EVENT。"
        "命名游戏标 WORK；普通项目或工程不得因缺少 PROJECT 类型而硬标 PRODUCT。"
        "一带一路等命名战略或倡议标 LAW。\n"
        "输出前执行两遍检查：第一遍逐个删除泛称、修正类型和边界；第二遍从原文扫描"
        "同一实体的重复出现，并补查 SPECIES、TECH 等容易漏标的实体。\n"
        f"类型定义：{json.dumps(TYPE_DEFINITIONS, ensure_ascii=False)}\n"
        "输出格式与初标相同：{\"rows\":[{\"id\":\"...\",\"text\":\"...\","
        "\"entities\":[{\"start\":0,\"end\":2,\"type\":\"PER\",\"text\":\"张三\"}]}]}。\n"
        f"待审核数据：{json.dumps(rows, ensure_ascii=False)}"
    )


def validate(rows: list[dict[str, str]], payload: object) -> list[dict]:
    expected = {row["id"]: row["text"] for row in rows}
    group_ids = {row["id"]: row.get("group_id") for row in rows}
    validated: dict[str, dict] = {}
    response_rows = payload if isinstance(payload, list) else payload.get("rows", []) if isinstance(payload, dict) else []
    for row in response_rows:
        row_id = str(row.get("id", ""))
        if row_id not in expected or row_id in validated:
            raise ValueError(f"invalid row identity: {row_id!r}")
        response_text = str(row.get("text", ""))
        text = expected[row_id]
        if response_text != text:
            print(f"restored_text\t{row_id}")
        normalized = []
        used_spans: set[tuple[int, int]] = set()
        for entity in row.get("entities", []):
            start = entity.get("start")
            end = entity.get("end")
            raw_kind = str(entity.get("type", ""))
            kind = TYPE_ALIASES.get(raw_kind, raw_kind)
            value = str(entity.get("text", ""))
            if raw_kind in SKIPPED_TYPES:
                continue
            if kind not in ENTITY_TYPES:
                print(f"skipped_type\t{row_id}\t{raw_kind}\t{value}")
                continue
            if not value:
                raise ValueError(f"invalid entity in {row_id}: {entity!r}")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or text[start:end] != value
                or (start, end) in used_spans
            ):
                matches = [
                    match.start()
                    for match in re.finditer(re.escape(value), text)
                    if (match.start(), match.start() + len(value)) not in used_spans
                ]
                if not matches:
                    if any(item["text"] == value and item["type"] == kind for item in normalized):
                        continue
                    print(f"skipped_entity\t{row_id}\t{value}")
                    continue
                start = min(matches, key=lambda position: abs(position - start)) if isinstance(start, int) else matches[0]
                end = start + len(value)
            used_spans.add((start, end))
            normalized.append({"start": start, "end": end, "type": kind, "text": value})

        entities = []
        for entity in sorted(
            normalized,
            key=lambda item: (-(item["end"] - item["start"]), item["start"], item["end"]),
        ):
            if all(
                entity["end"] <= other["start"] or entity["start"] >= other["end"]
                for other in entities
            ):
                entities.append(entity)
        entities.sort(key=lambda item: (item["start"], item["end"]))
        validated_row = {"id": row_id, "text": text, "entities": entities}
        if group_ids[row_id] is not None:
            validated_row["group_id"] = group_ids[row_id]
        validated[row_id] = validated_row
    if set(validated) != set(expected):
        raise ValueError("LLM response omitted or added rows")
    return [validated[row["id"]] for row in rows]


def self_test() -> None:
    assert tuple(TYPE_DEFINITIONS) == ENTITY_TYPES
    rows = [
        {
            "id": "demo",
            "group_id": "document-demo",
            "text": "会议修订《高新技术企业认定管理办法》。",
        }
    ]
    payload = [
        {
            "id": "demo",
            "text": rows[0]["text"],
            "entities": [
                {
                    "start": 4,
                    "end": 18,
                    "type": "LAW",
                    "text": "高新技术企业认定管理办法",
                }
            ],
        }
    ]
    entity = validate(rows, payload)[0]["entities"][0]
    assert rows[0]["text"][entity["start"] : entity["end"]] == entity["text"]
    restored = validate(
        rows,
        [{"id": "demo", "text": "被模型改写", "entities": []}],
    )[0]
    assert restored["text"] == rows[0]["text"]
    assert restored["group_id"] == rows[0]["group_id"]
    print("self_test\tok")


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            out[row["id"]] = row
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"G:\WordHub\MNBVC\20221224\baidu.20221224.18.新闻\chinese_news.csv"),
    )
    parser.add_argument("--review-input", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tasks/entity_llm/internal_news_labels.jsonl"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/tasks/entity_llm/internal_news_labels.manifest.json"),
    )
    parser.add_argument(
        "--prompt-out",
        type=Path,
        default=Path("data/tasks/entity_llm/internal_news_labels.prompt.txt"),
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--scan-rows", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--text-fields", default="")
    parser.add_argument("--category-pattern", default="")
    parser.add_argument("--source-id", default="mnbvc_chinese_news_internal")
    parser.add_argument("--source-license", default="unknown; internal training/evaluation only")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backfill-group-ids", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.review_input:
        rows = list(load_existing(args.review_input).values())[: args.limit]
        prompt_builder = review_prompt
        source_path = args.review_input
    else:
        rows = read_sentences(
            args.input,
            args.scan_rows,
            args.limit,
            args.seed,
            args.min_chars,
            args.max_chars,
            tuple(field.strip() for field in args.text_fields.split(",") if field.strip()) or None,
            args.category_pattern,
        )
        prompt_builder = prompt
        source_path = args.input
    if not rows:
        raise SystemExit("no sentences selected")
    args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_out.write_text(
        prompt_builder(rows[: args.batch_size]) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if args.backfill_group_ids:
        if args.review_input:
            raise SystemExit("--backfill-group-ids requires the original --input")
        existing = load_existing(args.out)
        if set(existing) != {row["id"] for row in rows}:
            raise SystemExit("backfill rows do not match the existing output")
        for row in rows:
            existing[row["id"]]["group_id"] = row["group_id"]
        write_rows(args.out, [existing[row["id"]] for row in rows])
        print(f"backfilled_group_ids\t{len(rows)}")
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if args.dry_run or not api_key:
            print(f"sentences\t{len(rows)}")
            print(f"prompt\t{args.prompt_out}")
            print("skipped\tdry_run" if args.dry_run else "skipped\tDEEPSEEK_API_KEY not set")
            return 0

        existing = {} if args.force else load_existing(args.out)
        pending = [row for row in rows if row["id"] not in existing]
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            for attempt in range(args.retries + 1):
                try:
                    payload = call_deepseek(
                        api_key,
                        args.base_url,
                        args.model,
                        prompt_builder(batch),
                        temperature=0.0,
                    )
                    for row in validate(batch, payload):
                        row["annotator"] = args.model
                        row["source"] = args.source_id
                        existing[row["id"]] = row
                    break
                except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                    if attempt >= args.retries:
                        raise SystemExit(f"annotation_validation_failed\t{exc}") from exc
                    print(f"retry\t{offset // args.batch_size + 1}\t{attempt + 1}\t{exc}")
            write_rows(args.out, [existing[row["id"]] for row in rows if row["id"] in existing])
            print(f"batch\t{min(offset + len(batch), len(pending))}/{len(pending)}")

    final_rows = [existing[row["id"]] for row in rows if row["id"] in existing]
    type_counts = Counter(entity["type"] for row in final_rows for entity in row["entities"])
    manifest = {
        "schema": "nexaloid.entity_llm_labels.v2",
        "source": str(source_path).replace("\\", "/"),
        "source_sha256": sha256(source_path),
        "source_license": args.source_license,
        "annotator": args.model,
        "seed": args.seed,
        "scan_rows": args.scan_rows,
        "sentence_count": len(final_rows),
        "entity_count": sum(type_counts.values()),
        "entity_types": list(ENTITY_TYPES),
        "type_counts": dict(sorted(type_counts.items())),
        "output_sha256": sha256(args.out),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"sentences\t{len(final_rows)}")
    print(f"entities\t{manifest['entity_count']}")
    print(f"types\t{json.dumps(manifest['type_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"wrote\t{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
