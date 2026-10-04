from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator


def flatten_predictions(rows: Iterable[Dict[str, object]]) -> Iterator[Dict[str, object]]:
    for row in rows:
        chains = row.get("fault_chains", []) or [{}]
        for index, chain in enumerate(chains, start=1):
            confidence = chain.get("confidence", {})
            evidence = chain.get("evidence", {})
            yield {
                "工单号": row.get("work_order_id", ""),
                "故障链序号": index,
                "工单类型": row.get("record_type", "FAULT"),
                "子系统ID": chain.get("subsystem_id", ""),
                "子系统": chain.get("subsystem_name", ""),
                "部件ID": chain.get("component_id", ""),
                "零部件": chain.get("component_name", ""),
                "故障模式ID": chain.get("failure_mode_id", ""),
                "故障模式": chain.get("failure_mode_name", ""),
                "原因ID": chain.get("cause_id", ""),
                "失效原因": chain.get("cause_text", ""),
                "故障原因原文": chain.get("cause_raw", ""),
                "原因抽取方法": chain.get("cause_method", ""),
                "根因分类ID": chain.get("cause_id", ""),
                "根因分类": chain.get("cause_name", ""),
                "症状": "|".join(chain.get("symptoms", [])),
                "故障码": "|".join(chain.get("dtc", [])),
                "部件置信度": confidence.get("component", ""),
                "模式置信度": confidence.get("failure_mode", ""),
                "原因置信度": confidence.get("cause", ""),
                "原因文本置信度": confidence.get("cause_text", ""),
                "故障链置信度": confidence.get("chain", ""),
                "决策": chain.get("decision", ""),
                "证据": json.dumps(evidence, ensure_ascii=False),
                "警告": "|".join(row.get("warnings", [])),
            }
