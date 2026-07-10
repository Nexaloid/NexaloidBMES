from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


HAN = re.compile(r"^[\u3400-\u9fff]+$")
COMMON_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华"
    "金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方"
    "俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于傅皮卞"
    "齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成"
    "戴宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭"
    "梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯管卢莫经房裘缪干"
    "解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆荣翁荀羊甄曲封"
    "芮储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋"
    "仲伊宫宁仇栾暴甘厉戎祖武符刘景詹束龙叶幸司韶黎乔苍双闻莘"
    "党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀浦尚"
    "农温别庄晏柴瞿阎充慕连茹习艾鱼容向古易慎戈廖庾终暨居衡步"
    "都耿满弘匡国文寇广禄阙东欧利蔚越夔隆师巩厍聂晁勾敖融冷訾"
    "辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益"
    "桓公仉督岳帅缑亢况郈有琴归海晋楚闫法汝鄢涂钦商牟佘佴伯赏"
    "墨哈谯笪年爱阳佟"
)

PERSON_STOP = {
    "陆军",
    "田间",
    "高昂",
    "元和",
    "王国",
    "林业",
    "马路",
    "于此",
    "方向",
    "和平",
    "华夏",
}

PLACE_SUFFIXES = (
    "特别行政区",
    "自治区",
    "自治州",
    "自治县",
    "半岛",
    "群岛",
    "省",
    "市",
    "县",
    "区",
    "镇",
    "乡",
    "村",
    "州",
    "盟",
    "旗",
    "岛",
    "山",
    "河",
    "湖",
    "湾",
    "港",
    "路",
    "街",
    "巷",
    "洲",
    "国",
    "郡",
    "府",
    "城",
)

PLACE_BAD_SUFFIXES = (
    "医院",
    "学校",
    "大学",
    "学院",
    "法院",
    "政府",
    "公安局",
    "派出所",
    "电视台",
    "图书馆",
    "公司",
    "委员会",
)

CORE_PLACES = {
    "中国",
    "北京",
    "上海",
    "天津",
    "重庆",
    "广州",
    "深圳",
    "杭州",
    "南京",
    "武汉",
    "成都",
    "西安",
    "长沙",
    "苏州",
    "日本",
    "东京",
    "美国",
    "英国",
    "法国",
    "德国",
    "加拿大",
    "澳大利亚",
    "新加坡",
    "亚洲",
    "欧洲",
    "非洲",
}

PLACE_STOP = {
    "医院",
    "学校",
    "法院",
    "小区",
    "市委",
    "公园",
    "政府",
    "公安",
    "公安局",
    "派出所",
    "酒店",
    "广场",
    "城区",
    "公交",
    "小学",
    "学院",
    "机场",
    "图书馆",
    "公路",
    "电视台",
    "大陆",
    "南方",
}

ORG_SUFFIXES = (
    "大学",
    "人民医院",
    "科技有限公司",
    "数据研究院",
    "商业银行",
    "智算中心",
    "联合实验室",
    "产业集团",
    "技术委员会",
    "证券交易所",
    "科学院",
    "计算所",
    "人民政府",
    "中心医院",
)

RESERVED_BADCASE_SURFACES = {
    "阿明",
    "星澜智算研究院",
    "长春",
    "云海新区",
    "海州市人民医院",
    "王小明",
    "东湖区",
    "小王",
    "雷锋",
    "日本",
    "东京",
    "上海证券交易所",
}

TRAIN_ANCHOR_SURFACES = {
    "阿强",
    "小李",
    "王大明",
    "赵云",
    "中国",
    "北京",
    "法国",
    "巴黎",
    "南湖区",
    "深圳证券交易所",
    "云海数据研究院",
    "南京",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_thuocl(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            freq = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            freq = 0
        rows.append((parts[0], freq))
    return rows


def keep_person(word: str) -> bool:
    return (
        2 <= len(word) <= 4
        and bool(HAN.fullmatch(word))
        and word[0] in COMMON_SURNAMES
        and word not in PERSON_STOP
    )


def keep_place(word: str) -> bool:
    if not 2 <= len(word) <= 8 or not HAN.fullmatch(word) or word in PLACE_STOP:
        return False
    if word.endswith(PLACE_BAD_SUFFIXES):
        return False
    if word in CORE_PLACES:
        return True
    return any(word.endswith(suffix) and len(word) > len(suffix) for suffix in PLACE_SUFFIXES)


def strip_place_suffix(word: str) -> str:
    for suffix in PLACE_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


def split_surfaces(items: list[str], limit: int, seed: int) -> dict[str, list[str]]:
    unique = sorted(set(items))
    random.Random(seed).shuffle(unique)
    unique = unique[:limit]
    train_end = int(len(unique) * 0.8)
    dev_end = int(len(unique) * 0.9)
    return {
        "train": unique[:train_end],
        "dev": unique[train_end:dev_end],
        "test": unique[dev_end:],
    }


def build_orgs(places: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for index, place in enumerate(places):
        stem = strip_place_suffix(place)
        if len(stem) < 2:
            continue
        value = stem + ORG_SUFFIXES[index % len(ORG_SUFFIXES)]
        if value in RESERVED_BADCASE_SURFACES or value in TRAIN_ANCHOR_SURFACES:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
        if len(out) >= limit:
            break
    return out


def render(parts: list[str | tuple[str, str]]) -> tuple[str, list[tuple[int, int, str]]]:
    text_parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for part in parts:
        if isinstance(part, tuple):
            value, kind = part
            start = cursor
            text_parts.append(value)
            cursor += len(value)
            spans.append((start, cursor, kind))
        else:
            text_parts.append(part)
            cursor += len(part)
    return "".join(text_parts), spans


TRAIN_TEMPLATES = {
    "PER": (
        lambda value: [("PER", value), "参加了项目评审"],
        lambda value: ["记者采访了", ("PER", value), "并发布报道"],
        lambda value: [("PER", value), "今天发表讲话"],
        lambda value: ["会议由", ("PER", value), "主持"],
        lambda value: ["我们联系了", ("PER", value)],
        lambda value: [("PER", value), "在技术团队工作"],
        lambda value: [("PER", value), "精神广为流传"],
    ),
    "LOC": (
        lambda value: ["团队前往", ("LOC", value), "开展调研"],
        lambda value: ["货物已经抵达", ("LOC", value)],
        lambda value: [("LOC", value), "发布了最新通知"],
        lambda value: ["项目位于", ("LOC", value), "附近"],
        lambda value: ["他从", ("LOC", value), "返回"],
    ),
    "ORG": (
        lambda value: [("ORG", value), "发布了新模型"],
        lambda value: ["我们与", ("ORG", value), "签署协议"],
        lambda value: [("ORG", value), "完成系统升级"],
        lambda value: ["报告由", ("ORG", value), "提交"],
        lambda value: ["他加入了", ("ORG", value)],
    ),
}

DEV_TEMPLATES = {
    "PER": (
        lambda value: [("PER", value), "负责本次系统升级"],
        lambda value: ["昨日", ("PER", value), "抵达会场"],
        lambda value: ["报告中提到了", ("PER", value)],
    ),
    "LOC": (
        lambda value: ["列车正在驶向", ("LOC", value)],
        lambda value: [("LOC", value), "今天出现降温"],
        lambda value: ["活动将在", ("LOC", value), "举行"],
    ),
    "ORG": (
        lambda value: [("ORG", value), "宣布开展合作"],
        lambda value: ["会议在", ("ORG", value), "召开"],
        lambda value: [("ORG", value), "正在招聘工程师"],
    ),
}

TEST_TEMPLATES = {
    "PER": (
        lambda value: [("PER", value), "加入了新的研究项目"],
        lambda value: [("PER", value), "明天参加会议"],
        lambda value: ["我们欢迎", ("PER", value), "到访"],
    ),
    "LOC": (
        lambda value: ["他长期居住在", ("LOC", value)],
        lambda value: [("LOC", value), "迎来了许多游客"],
        lambda value: ["考察队从", ("LOC", value), "出发"],
    ),
    "ORG": (
        lambda value: [("ORG", value), "发布了最新公告"],
        lambda value: ["我们访问了", ("ORG", value)],
        lambda value: [("ORG", value), "完成了平台部署"],
    ),
}

TEMPLATES_BY_SPLIT = {
    "train": TRAIN_TEMPLATES,
    "dev": DEV_TEMPLATES,
    "test": TEST_TEMPLATES,
}


def entity_rows(kind: str, surfaces: list[str], split: str) -> list[tuple[str, list[tuple[int, int, str]]]]:
    templates = TEMPLATES_BY_SPLIT[split][kind]
    rows: list[tuple[str, list[tuple[int, int, str]]]] = []
    for index, surface in enumerate(surfaces):
        parts = templates[index % len(templates)](surface)
        normalized: list[str | tuple[str, str]] = []
        for part in parts:
            if isinstance(part, tuple):
                part_kind, value = part
                normalized.append((value, part_kind))
            else:
                normalized.append(part)
        rows.append(render(normalized))
    return rows


def combined_rows(
    persons: list[str],
    places: list[str],
    orgs: list[str],
    limit: int,
    split: str,
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    count = min(len(persons), len(places), len(orgs), limit)
    rows = []
    for index in range(count):
        if split == "train":
            parts = [
                (persons[index], "PER"),
                "前往",
                (places[index], "LOC"),
                "并加入",
                (orgs[index], "ORG"),
            ]
        elif split == "dev":
            parts = [
                (orgs[index], "ORG"),
                "派",
                (persons[index], "PER"),
                "到",
                (places[index], "LOC"),
                "调研",
            ]
        else:
            parts = [
                (persons[index], "PER"),
                "代表",
                (orgs[index], "ORG"),
                "访问",
                (places[index], "LOC"),
            ]
        rows.append(render(parts))
    return rows


NEGATIVE_SAMPLES = {
    "train": (
        "你还小明天再说",
        "高昂的成本需要控制",
        "田间管理需要加强",
        "陆军正在组织训练",
        "研究生命起源",
        "系统正在正常运行",
        "这所大学今天开学",
        "研究院正在建设新平台",
        "负责人参加了项目评审",
        "团队前往现场开展调研",
        "报告由部门提交",
        "患者服用药物控制症状",
        "字符串优化知识库",
        "市长春节前发表讲话",
        "普通用户今天提交申请",
        "医院附近交通比较拥堵",
        "大学毕业以后继续深造",
        "市长今天发表讲话",
        "路由器出现网络故障",
        "汪洋大海一望无际",
        "证券交易今天恢复正常",
    ),
    "dev": (
        "部门负责人正在审核材料",
        "学校附近今天道路拥堵",
        "普通研究人员提交了报告",
        "公司完成了系统升级",
        "会议中心今天正常开放",
        "明天参加会议的人很多",
        "现场工作人员正在登记",
    ),
    "test": (
        "技术负责人明天参加会议",
        "这家医院附近正在施工",
        "大学毕业生提交了申请",
        "普通用户访问服务中心",
        "证券交易今天出现波动",
        "研究人员完成平台部署",
        "城市道路今天比较拥堵",
    ),
}


def negative_rows(split: str, repeat: int) -> list[tuple[str, list[tuple[int, int, str]]]]:
    samples = NEGATIVE_SAMPLES[split]
    return [(samples[index % len(samples)], []) for index in range(repeat)]


def train_guard_rows() -> list[tuple[str, list[tuple[int, int, str]]]]:
    rows = [
        render([("阿强", "PER"), "加入", ("云海数据研究院", "ORG")]),
        render([("小李", "PER"), "明天参加会议"]),
        render([("王大明", "PER"), "在", ("南湖区", "LOC"), "工作"]),
        render([("赵云", "PER"), "精神广为流传"]),
        render(["他从", ("法国", "LOC"), ("巴黎", "LOC"), "返回"]),
        render([("深圳证券交易所", "ORG"), "发布通知"]),
        render([("南京", "LOC"), "市长今天讲话"]),
        ("陆军正在组织训练", []),
        ("田间管理需要加强", []),
        ("高昂的成本需要控制", []),
        ("这所大学今天开学", []),
        ("医院附近交通拥堵", []),
        ("路由器出现故障", []),
    ]
    return rows


def encode_spans(spans: list[tuple[int, int, str]]) -> str:
    return ",".join(f"{start}:{end}:{kind}" for start, end, kind in spans) or "-"


def write_cases(path: Path, rows: list[tuple[str, list[tuple[int, int, str]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# text\tstart:end:type,..."]
    lines.extend(f"{text}\t{encode_spans(spans)}" for text, spans in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordhub", type=Path, default=Path(r"G:\WordHub"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tasks/entity_hmm"))
    parser.add_argument("--person-limit", type=int, default=6000)
    parser.add_argument("--place-limit", type=int, default=9000)
    parser.add_argument("--org-limit", type=int, default=9000)
    parser.add_argument("--combined-train", type=int, default=1500)
    parser.add_argument("--negative-train", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source_dir = args.wordhub / "THUOCL" / "data"
    person_path = source_dir / "THUOCL_lishimingren.txt"
    place_path = source_dir / "THUOCL_diming.txt"
    persons = [
        word
        for word, _ in read_thuocl(person_path)
        if keep_person(word)
        and word not in RESERVED_BADCASE_SURFACES
        and word not in TRAIN_ANCHOR_SURFACES
    ]
    nickname_persons = [
        value
        for surname in sorted(COMMON_SURNAMES)
        for prefix in ("小", "阿")
        if (value := f"{prefix}{surname}") not in RESERVED_BADCASE_SURFACES
        and value not in TRAIN_ANCHOR_SURFACES
    ]
    persons += nickname_persons
    places = [
        word
        for word, _ in read_thuocl(place_path)
        if keep_place(word)
        and word not in RESERVED_BADCASE_SURFACES
        and word not in TRAIN_ANCHOR_SURFACES
    ]
    person_splits = split_surfaces(persons, args.person_limit, args.seed)
    place_splits = split_surfaces(places, args.place_limit, args.seed + 1)
    org_splits = {
        split: build_orgs(values, args.org_limit)
        for split, values in place_splits.items()
    }

    datasets: dict[str, list[tuple[str, list[tuple[int, int, str]]]]] = {}
    for split in ("train", "dev", "test"):
        rows = []
        rows += entity_rows("PER", person_splits[split], split)
        rows += entity_rows("LOC", place_splits[split], split)
        rows += entity_rows("ORG", org_splits[split], split)
        rows += combined_rows(
            person_splits[split],
            place_splits[split],
            org_splits[split],
            args.combined_train if split == "train" else 300,
            split,
        )
        rows += negative_rows(split, args.negative_train if split == "train" else 500)
        if split == "train":
            rows += train_guard_rows() * 30
        random.Random(args.seed + {"train": 10, "dev": 20, "test": 30}[split]).shuffle(rows)
        datasets[split] = rows
        write_cases(args.out_dir / f"{split}.tsv", rows)

    manifest = {
        "schema": "nexaloid.entity_hmm_data.v1",
        "seed": args.seed,
        "source_ids": ["thuocl"],
        "license": "MIT",
        "sources": {
            "person": str(person_path).replace("\\", "/"),
            "person_sha256": sha256(person_path),
            "place": str(place_path).replace("\\", "/"),
            "place_sha256": sha256(place_path),
        },
        "filters": {
            "person_limit": args.person_limit,
            "place_limit": args.place_limit,
            "org_limit": args.org_limit,
            "person_rule": "2-4 Han chars; common Chinese surname; ambiguity stoplist",
            "place_rule": "2-8 Han chars; administrative/geographic suffix or core-place whitelist",
            "org_rule": "filtered place stem + deterministic organization suffix",
        },
        "surface_counts": {
            split: {
                "PER": len(person_splits[split]),
                "LOC": len(place_splits[split]),
                "ORG": len(org_splits[split]),
            }
            for split in ("train", "dev", "test")
        },
        "row_counts": {split: len(rows) for split, rows in datasets.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for split, rows in datasets.items():
        print(f"{split}_rows\t{len(rows)}")
    for split in ("train", "dev", "test"):
        for kind in ("PER", "LOC", "ORG"):
            print(f"{split}_{kind}\t{manifest['surface_counts'][split][kind]}")
    print(f"wrote\t{args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
