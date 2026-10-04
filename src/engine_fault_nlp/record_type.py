from __future__ import annotations

import re

from .schemas import WorkOrder
from .text import normalize_text


RECORD_TYPE_RULE_VERSION = "record-type-rules-v1"


MAINTENANCE_PATTERNS = (
    r"主动升级", r"数据刷写", r"程序优化升级", r"发动机(?:及变速箱)?数据升级",
    r"新(?:销售|购|出厂).*升级", r"标定(?:主动)?升级", r"升级标定", r"ECM数据升级",
    r"应.*要求.*(?:刷写|标定)", r"整改", r"保养", r"赠送.*(?:机油|机滤)", r"照顾性支持",
    r"(?:更换|换)(?:机油|机滤|气滤)", r"采集数据", r"安装datalogger", r"拆除监控器",
    r"清(?:除)?故障码",
)
TEST_SERVICE_PATTERNS = (
    r"热平衡(?:测试|试验)", r"上门服务", r"协助.*工程师", r"采集数据",
    r"安装datalogger", r"拆除监控器", r"产品试验",
    r"申报.*(?:费用|费|扣款)", r"路试费用", r"清洁费", r"时效扣款",
)
STRONG_FAULT_TERMS = (
    "损坏", "断裂", "破裂", "开裂", "磨损", "漏油", "漏水", "漏气", "泄漏", "短路",
    "开路", "断路", "卡滞", "打滑", "接触不良", "密封不严", "无法启动", "烧蚀",
    "功能失效", "数据漂移", "电压高于", "电压低于", "高温故障", "压力不足",
    "设置有误", "标定错误", "标定不对",
)


def classify_record_type(order: WorkOrder) -> str:
    text = normalize_text("\n".join([
        order.fault_name, order.fault_description, order.diagnosis, order.solution,
        " ".join(part.part_name for part in order.parts),
    ]))
    if "非发动机原因" in text:
        return "OUT_OF_SCOPE"
    maintenance = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in MAINTENANCE_PATTERNS)
    test_service = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in TEST_SERVICE_PATTERNS)
    strong_fault = any(term in text for term in STRONG_FAULT_TERMS)
    if test_service and not strong_fault:
        return "TEST_SERVICE"
    if maintenance and not strong_fault:
        return "MAINTENANCE"
    if not text:
        return "UNKNOWN"
    return "FAULT"
