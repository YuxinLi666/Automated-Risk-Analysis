from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, Optional

from .ontology import KnowledgeBase
from .schemas import WorkOrder


PROMPT_VERSION = "fault-chain-candidate-v2"

SYSTEM_PROMPT = """你是重卡天然气发动机售后故障候选抽取器。请只依据给定工单输出JSON。
区分：症状是用户观察，故障模式是部件如何失效，失效原因是文本明确说明的“为何失效”。
严格规则：
1. 每条链必须给出可在原工单中逐字找到的证据片段；否定项不得输出。
2. “更换某部件”不能单独证明该部件失效；解决方案不能单独作为失效原因。
3. 没有明确原因证据时cause_id必须为UNKNOWN，不得用常识补全原因。
4. 允许一张工单有多条故障链；不要为了凑齐字段而猜测。
5. 仅输出work_order_id和fault_chains，fault_chains必须是数组。"""


class OpenAICompatibleTeacher:
    def __init__(self, endpoint: str, model: str, api_key: str = "", timeout: int = 120):
        base = endpoint.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout

    def label(
        self,
        order: WorkOrder,
        kb: KnowledgeBase,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        schema_hint = {
            "subsystems": kb.subsystems,
            "failure_modes": {key: value["name"] for key, value in kb.failure_modes.items()},
            "causes": {key: value["name"] for key, value in kb.causes.items()},
        }
        output_schema = {
            "work_order_id": order.work_order_id,
            "fault_chains": [
                {
                    "subsystem_id": "枚举ID或UNKNOWN",
                    "subsystem_name": "标准名称或原文名称",
                    "component_name": "原文部件名称",
                    "failure_mode_id": "枚举ID或UNKNOWN",
                    "failure_mode_name": "标准名称或原文名称",
                    "cause_id": "枚举ID或UNKNOWN",
                    "cause_name": "标准名称或未说明",
                    "symptoms": ["原文症状"],
                    "evidence": {
                        "component": {"field": "字段标签", "text": "原文逐字片段"},
                        "failure_mode": {"field": "字段标签", "text": "原文逐字片段"},
                        "root_cause": {"field": "字段标签", "text": "原文逐字片段或空"},
                    },
                    "confidence": {
                        "subsystem": 0.0,
                        "component": 0.0,
                        "failure_mode": 0.0,
                        "cause": 0.0,
                    },
                }
            ],
        }
        user = "本体枚举:\n{}\n\n输出结构示例:\n{}\n\n工单:\n{}".format(
            json.dumps(schema_hint, ensure_ascii=False),
            json.dumps(output_schema, ensure_ascii=False),
            order.candidate_text(),
        )
        payload = {
            "model": self.model,
            "temperature": max(0.0, float(temperature)),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            payload["seed"] = int(seed)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": "Bearer " + self.api_key} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError("Teacher API failed: {} {}".format(error.code, detail)) from error
        try:
            content = raw["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Teacher API returned invalid JSON response") from error
        if not isinstance(result, dict) or not isinstance(result.get("fault_chains"), list):
            raise RuntimeError("Teacher response must contain a fault_chains array")
        result["work_order_id"] = order.work_order_id
        return result
