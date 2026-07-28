"""Build a deterministic 240-row synthetic draft for the second-Schema smoke test."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.keyword_evaluator import file_sha256, write_json_atomic, write_jsonl_atomic
from examples.intent_routing.schema import INTENTS, URGENCY, create_intent_routing_schema


SPLIT_COUNTS = {"dev": 120, "test": 100, "challenge": 20}

EVIDENCE_BANKS = {
    "dev": {
        "refund": ["申请退款", "要求退货退款", "还没收到退款", "取消订单并退款", "希望原路退回", "退款一直没到账", "想办理退货", "请退还货款"],
        "delivery": ["包裹还没送到", "物流一直没更新", "快递送错地址", "催一下发货", "显示签收但没收到", "配送时间太久", "包裹卡在中转站", "快递员没有联系我"],
        "product_quality": ["收到就是坏的", "商品有明显划痕", "用了两天就故障", "尺寸与描述不符", "配件少了一件", "衣服开线了", "屏幕有坏点", "味道非常刺鼻"],
        "account": ["无法登录账号", "验证码收不到", "账户被锁定", "修改不了手机号", "订单记录不见了", "会员权益没到账", "密码重置失败", "账号疑似被盗"],
        "other": ["想了解保修期限", "请问有没有发票", "需要修改收件人", "能否开具电子发票", "咨询商品使用方法", "想更换包装颜色", "请问门店在哪里", "想知道活动规则"],
    },
    "test": {
        "refund": ["麻烦把钱退回来", "这单我想退掉", "退货后没有返款", "请撤单并返还金额", "退款进度查不到"],
        "delivery": ["订单迟迟未发出", "货物没有按时到", "物流轨迹停了几天", "快件被投到别处", "签收状态不正确"],
        "product_quality": ["实物已经破损", "功能完全不能用", "颜色和页面不一致", "包装内缺少零件", "做工存在严重问题"],
        "account": ["登录时总提示错误", "手机验证无法通过", "账户突然被冻结", "绑定信息无法更新", "积分余额消失了"],
        "other": ["咨询延长保修", "需要纸质发票", "询问安装方式", "想了解优惠条件", "请告知营业时间"],
    },
}

URGENCY_CONTEXT = {
    "low": ("有空时帮我看看，", "暂时不着急。"),
    "normal": ("麻烦正常处理，", "请告知处理进度。"),
    "high": ("现在非常着急，", "请尽快处理，今天需要答复。"),
}

TEST_SUFFIXES = [
    "客服可以通过站内信回复我。",
    "相关订单信息已经在后台提交。",
    "如果需要照片我可以继续补充。",
    "这是我第一次反馈这个问题。",
]

CHALLENGE_ROWS = {
    "refund": [
        ("页面还在显示配送中，但我的核心诉求是申请退款，不需要继续催件。", "申请退款", "negated_distractor", "high"),
        ("质量我可以接受，只是重复下单了，请退还货款。", "退还货款", "mixed_intent", "normal"),
        ("物流后来恢复了，不过这单已经取消，钱还没退回来。", "钱还没退回来", "state_transition", "high"),
        ("帮我把这笔订单原路退回就行，其他问题不用处理。", "原路退回", "colloquial", "low"),
    ],
    "delivery": [
        ("我不是要退款，只想确认为什么包裹还没送到。", "包裹还没送到", "negated_distractor", "normal"),
        ("商品评价很好，但物流轨迹停了四天，请查一下。", "物流轨迹停了四天", "mixed_intent", "high"),
        ("昨天说已出库，今天又显示待发货，订单迟迟未发出。", "订单迟迟未发出", "state_transition", "high"),
        ("件儿不知道跑哪儿去了，麻烦找一下快递。", "件儿不知道跑哪儿去了", "colloquial", "normal"),
    ],
    "product_quality": [
        ("快递很准时，不需要查物流，但实物已经破损。", "实物已经破损", "negated_distractor", "high"),
        ("我能正常登录，真正的问题是屏幕有坏点。", "屏幕有坏点", "mixed_intent", "normal"),
        ("刚开箱可以使用，半小时后功能完全不能用。", "功能完全不能用", "state_transition", "high"),
        ("这做工也太糙了，边角全是毛刺。", "边角全是毛刺", "colloquial", "low"),
    ],
    "account": [
        ("订单已经收到，不用处理配送；现在是无法登录账号。", "无法登录账号", "negated_distractor", "high"),
        ("退款已经到账，但我的绑定信息无法更新。", "绑定信息无法更新", "mixed_intent", "normal"),
        ("上午还能进入账户，下午账户突然被冻结。", "账户突然被冻结", "state_transition", "high"),
        ("号登不上去了，老是说验证失败。", "号登不上去了", "colloquial", "normal"),
    ],
    "other": [
        ("我没有质量或退款问题，只想了解保修期限。", "了解保修期限", "negated_distractor", "low"),
        ("包裹已收到，现在需要纸质发票。", "需要纸质发票", "mixed_intent", "normal"),
        ("之前问过价格，现在主要想了解优惠条件。", "了解优惠条件", "state_transition", "low"),
        ("这个咋装呀，能发个使用说明不？", "咋装呀", "colloquial", "normal"),
    ],
}


def _row(
    sample_id: str,
    group_id: str,
    split: str,
    source: str,
    intent: str,
    urgency: str,
    evidence: str,
    challenge_slice: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "split": split,
        "source": source,
        "target": {"intent": intent, "urgency": urgency, "evidence": evidence},
        "challenge_slice": challenge_slice,
        "synthetic": True,
        "human_verified": False,
        "annotation_status": "draft",
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intent in INTENTS:
        for phrase_index, evidence in enumerate(EVIDENCE_BANKS["dev"][intent], 1):
            for urgency_index, urgency in enumerate(URGENCY, 1):
                prefix, suffix = URGENCY_CONTEXT[urgency]
                sample_id = f"ir-dev-{intent}-{phrase_index:02d}-{urgency_index}"
                source = f"{prefix}{evidence}。订单编号已隐藏，{suffix}"
                rows.append(_row(sample_id, f"ir-dev-{intent}-{phrase_index:02d}", "dev", source, intent, urgency, evidence))

        for phrase_index, evidence in enumerate(EVIDENCE_BANKS["test"][intent], 1):
            for variant_index, suffix_text in enumerate(TEST_SUFFIXES, 1):
                urgency = URGENCY[(phrase_index + variant_index) % len(URGENCY)]
                prefix, suffix = URGENCY_CONTEXT[urgency]
                sample_id = f"ir-test-{intent}-{phrase_index:02d}-{variant_index}"
                source = f"{prefix}{evidence}。{suffix_text}{suffix}"
                rows.append(_row(sample_id, f"ir-test-{intent}-{phrase_index:02d}", "test", source, intent, urgency, evidence))

        for challenge_index, (source, evidence, slice_name, urgency) in enumerate(CHALLENGE_ROWS[intent], 1):
            sample_id = f"ir-challenge-{intent}-{challenge_index:02d}"
            rows.append(_row(sample_id, f"ir-challenge-{intent}-{challenge_index:02d}", "challenge", source, intent, urgency, evidence, slice_name))
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    schema = create_intent_routing_schema()
    ids: set[str] = set()
    sources: set[str] = set()
    groups_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_COUNTS}
    split_counts = Counter()
    intent_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_COUNTS}
    urgency_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_COUNTS}

    for row in rows:
        sample_id = row["sample_id"]
        source = row["source"]
        split = row["split"]
        target = row["target"]
        if sample_id in ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        if source in sources:
            raise ValueError(f"duplicate source: {source}")
        ids.add(sample_id)
        sources.add(source)
        groups_by_split[split].add(row["group_id"])
        split_counts[split] += 1
        intent_counts[split][target["intent"]] += 1
        urgency_counts[split][target["urgency"]] += 1
        valid, errors = schema.validate(target)
        if not valid:
            raise ValueError(f"{sample_id} has invalid target: {errors}")
        if set(target) != {"intent", "urgency", "evidence"}:
            raise ValueError(f"{sample_id} target keys do not match the contract")
        if target["evidence"] not in source:
            raise ValueError(f"{sample_id} evidence is not copied from source")
        if row["synthetic"] is not True or row["human_verified"] is not False:
            raise ValueError(f"{sample_id} has incorrect provenance flags")

    if dict(split_counts) != SPLIT_COUNTS:
        raise ValueError(f"split counts differ from preregistration: {dict(split_counts)}")
    split_names = list(SPLIT_COUNTS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"group leakage between {left} and {right}: {sorted(overlap)[:3]}")
    return {
        "sample_count": len(rows),
        "split_counts": dict(split_counts),
        "intent_counts": {split: dict(sorted(counts.items())) for split, counts in intent_counts.items()},
        "urgency_counts": {split: dict(sorted(counts.items())) for split, counts in urgency_counts.items()},
        "unique_sample_ids": len(ids),
        "unique_sources": len(sources),
        "group_overlap_count": 0,
    }


def write_review_packet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "group_id", "split", "source", "intent", "urgency", "evidence",
        "challenge_slice", "approved", "correction_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "source": row["source"],
                **row["target"],
                "challenge_slice": row["challenge_slice"] or "",
                "approved": "",
                "correction_notes": "",
            })


def run(output_dir: Path) -> dict[str, Any]:
    rows = build_rows()
    validation = validate_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split in SPLIT_COUNTS:
        path = output_dir / f"{split}.jsonl"
        write_jsonl_atomic(path, [row for row in rows if row["split"] == split])
        files[split] = {"path": str(path.as_posix()), "sha256": file_sha256(path), "rows": SPLIT_COUNTS[split]}
    review_path = output_dir / "review_packet.csv"
    write_review_packet(review_path, rows)
    files["review_packet"] = {"path": str(review_path.as_posix()), "sha256": file_sha256(review_path), "rows": len(rows)}
    manifest = {
        "report_version": "intent-routing-draft-v1",
        "status": "DRAFT_REQUIRES_HUMAN_REVIEW",
        "synthetic": True,
        "human_verified": False,
        "frozen": False,
        "selection_use": "none",
        "generation": "deterministic_template_composition",
        "validation": validation,
        "files": files,
        "next_action": "review every row in review_packet.csv, set approved=true, then run freeze_second_schema_dataset.py",
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/intent_routing_v1_draft"))
    args = parser.parse_args()
    try:
        result = run(args.output_dir)
        print(json.dumps({"event": "second_schema_draft_complete", **result}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"event": "second_schema_draft_complete", "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
