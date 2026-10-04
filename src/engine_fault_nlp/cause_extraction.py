from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .text import normalize_text


CAUSE_TERMS = (
    "损坏", "失效", "异常", "漂移", "断裂", "磨损", "磨破", "短路", "开路",
    "断路", "泄压", "泄漏", "漏油", "漏水", "漏气", "堵塞", "卡滞", "打滑", "烧蚀",
    "破裂", "老化", "松动", "脱落", "击穿", "电压高", "电压低", "高于正常", "低于正常",
    "数据不稳定", "信号不稳定", "间隙异常", "无信号", "不工作", "不能工作",
    "数据问题", "失火", "设置有误", "效率低", "开裂", "接触不良", "密封不严",
    "压力值不够", "压力不足", "内部有水", "松旷", "质量问题",
)
SYMPTOM_ONLY = (
    "动力不足", "无力", "限扭", "限速", "故障灯亮", "启动困难", "无法启动", "抖动",
    "异响", "熄火", "高温", "气耗高", "白烟", "黑烟", "蓝烟",
)
NEGATION_TERMS = ("未见异常", "无异常", "没有异常", "正常", "排除", "未发现")
RESOLUTION_TERMS = (
    "故障排除", "故障解除", "故障已解除", "故障消失", "恢复正常", "试车正常",
    "维修完成", "维修完毕", "清除故障码", "更换后正常",
)
CAUSAL_PATTERNS = (
    re.compile(r"(?:故障原因为|原因为|原因是|判断(?:为|是)?|判定(?:为|是)?|确认为|确认是|检查(?:为|是)|得出结果为)(.{2,140}?)(?=导致|造成|引起|所致|[。；;\n,]|$)"),
    re.compile(r"(?<!原)(?:由于|因)(.{2,140}?)(?=导致|造成|引起|所致|[。；;\n,]|$)"),
    re.compile(r"(?:检查发现|经检查发现|检测发现|发现)(?:是|为)?(.{2,140}?)(?=导致|造成|引起|所致|[。；;\n,]|$)"),
    re.compile(r"(.{2,140}?)(?=导致|造成|引起|所致)"),
)


@dataclass
class CauseExtraction:
    text: str = ""
    raw: str = ""
    method: str = "none"
    confidence: float = 0.0


def _strip_boilerplate(text: str) -> str:
    text = normalize_text(text).strip(" ,，。;；:：")
    text = re.sub(r"^(?:到达现场|现场|经维修人员|经我站人员|维修人员|服务人员|经工作人员)", "", text)
    text = re.sub(r"^(?:检查|检测|排查|确认|判断|发现)(?:发现|后|为|是)?", "", text)
    text = re.sub(r"^(?:导致|造成|引起)", "", text)
    text = re.sub(r"^(?:该车|车辆|发动机)", "", text)
    text = re.sub(r"(?:(?:导致|造成|引起)(?:车辆|发动机)?(?:故障|故障发生)|所致故障)$", "", text)
    return text.strip(" ,，。;；:：")


def _cause_score(candidate: str, component_name: str = "") -> float:
    if not candidate:
        return -10.0
    score = 0.0
    score += 2.0 * sum(term in candidate for term in CAUSE_TERMS)
    if re.search(r"(?<!码)(?<!灯)故障(?!码|灯|排除|解除|消失)", candidate):
        score += 2.0
    score += 1.5 * sum(cue in candidate for cue in ("判断", "判定", "原因", "由于", "发现"))
    if component_name and component_name in candidate:
        score += 2.0
    if any(term in candidate for term in NEGATION_TERMS) and not any(term in candidate for term in CAUSE_TERMS):
        score -= 5.0
    if any(term in candidate for term in RESOLUTION_TERMS):
        score -= 6.0
    if candidate in SYMPTOM_ONLY or (len(candidate) <= 10 and any(candidate.endswith(s) for s in SYMPTOM_ONLY)):
        score -= 4.0
    if len(candidate) > 100:
        score -= (len(candidate) - 100) / 40
    return score


def _candidate_clauses(text: str) -> List[str]:
    clauses = []
    for clause in re.split(r"[。；;\n]|(?<!\d),(?!\d)", text):
        clause = _strip_boilerplate(clause)
        if clause:
            clauses.append(clause)
    return clauses


def extract_diagnostic_cause(diagnosis: object, component_name: str = "") -> CauseExtraction:
    raw = normalize_text(diagnosis)
    if not raw:
        return CauseExtraction(raw="")

    pattern_candidates = []
    for pattern_index, pattern in enumerate(CAUSAL_PATTERNS):
        for match in pattern.finditer(raw):
            candidate = _strip_boilerplate(match.group(1))
            if candidate:
                pattern_candidates.append((_cause_score(candidate, component_name), pattern_index, candidate))
    if pattern_candidates:
        score, pattern_index, candidate = max(
            pattern_candidates,
            key=lambda item: (item[0] + (3.0 if item[1] < 3 else 0.0), -item[1], -len(item[2])),
        )
        if score > 0 or pattern_index < 3:
            return CauseExtraction(candidate, raw, "causal_pattern_{}".format(pattern_index + 1), 0.96 if pattern_index < 3 else 0.91)

    compact = _strip_boilerplate(raw)
    if (
        len(compact) <= 80
        and not re.search(r"[。；;\n]", compact)
        and compact.count(",") <= 2
        and _cause_score(compact, component_name) >= 2.0
    ):
        return CauseExtraction(compact, raw, "short_diagnosis", 0.88)

    clauses = _candidate_clauses(raw)
    scored = [(_cause_score(clause, component_name), index, clause) for index, clause in enumerate(clauses)]
    if scored:
        score, _, candidate = max(scored, key=lambda item: (item[0], item[1]))
        if score >= 2.0:
            return CauseExtraction(candidate, raw, "diagnostic_clause", 0.86)

    return CauseExtraction("", raw, "not_causal", 0.0)
