from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from .ingest import load_work_orders
from .io_utils import read_json, write_json
from .schemas import WorkOrder
from .text import normalize_text, stable_id
from .xlsx import XlsxReader


SUBSYSTEMS = {
    "INTAKE_BOOST": "进气与增压系统",
    "GAS_SUPPLY": "燃气供给系统",
    "IGNITION": "点火系统",
    "ENGINE_MECHANICAL": "发动机机械系统",
    "VALVETRAIN": "配气机构",
    "LUBRICATION": "润滑系统",
    "COOLING": "冷却系统",
    "CRANKCASE_VENT": "曲轴箱通风系统",
    "START_CHARGE": "起动与充电系统",
    "ECM_ELECTRICAL": "ECM与电气系统",
    "AFTERTREATMENT": "排气后处理系统",
    "ACCESSORY": "附件驱动系统",
    "VEHICLE_SIDE": "车辆侧非发动机系统",
    "MAINTENANCE": "保养/整改非故障",
    "ENGINE_GENERAL": "发动机综合/未定位",
    "UNKNOWN": "待确认子系统",
}

SUBSYSTEM_KEYWORDS = [
    ("AFTERTREATMENT", ("后处理", "排温", "排气温度", "氧传感器", "三元催化", "催化器", "催化剂", "消声器", "排气管", "排气歧管", "隔热罩", "排气节流阀", "ETV阀")),
    ("CRANKCASE_VENT", ("曲轴箱通风", "曲轴通风", "呼吸器", "机油分离器", "泄压阀")),
    ("IGNITION", ("点火", "火花塞", "高压线", "缸线", "爆震", "失火", "缺火")),
    ("GAS_SUPPLY", ("燃气", "天然气", "燃油", "气轨", "减压器", "空气控制阀", "燃气喷射", "压力调节器", "燃油测量阀", "油水分离器", "气阀")),
    ("INTAKE_BOOST", ("增压器", "进气", "EGR", "节气门", "节流阀", "中冷器", "进气歧管")),
    ("COOLING", ("水泵", "水箱", "冷却", "水温", "高温", "节温器", "防冻液", "散热器", "出水管", "进水管")),
    ("LUBRICATION", ("机油", "油底壳", "机滤", "润滑", "机油泵", "油封", "O型圈", "O型密封圈", "漏油")),
    ("VALVETRAIN", ("气门", "摇臂", "凸轮", "挺杆", "推杆")),
    ("ENGINE_MECHANICAL", ("缸盖", "缸体", "活塞", "缸套", "曲轴", "连杆", "飞轮", "轴瓦", "止推瓦", "止推片", "整机", "密封垫", "螺栓", "螺柱", "管夹", "卡箍", "盖板")),
    ("START_CHARGE", ("起动机", "启动机", "发电机", "蓄电池")),
    ("ECM_ELECTRICAL", ("ECM", "控制模块", "电控模块", "模块支架", "线束", "继电器", "保险盒", "传感器", "执行器", "电子故障检修", "电子故障")),
    ("ACCESSORY", ("转向泵", "空压机", "空气压缩机", "风扇", "皮带", "多楔带")),
    ("MAINTENANCE", ("标定", "升级", "保养", "数据刷写", "整改")),
    ("VEHICLE_SIDE", ("非发动机", "变速箱", "离合器", "制动", "仪表", "车桥", "气瓶", "搭铁线")),
]

FAILURE_MODES = {
    "BROKEN": {"name": "断裂", "keywords": ["断裂", "折断", "破裂"]},
    "WORN": {"name": "磨损", "keywords": ["磨损", "磨破", "拉伤"]},
    "LEAK": {"name": "泄漏", "keywords": ["漏油", "漏水", "漏气", "泄漏", "渗漏"]},
    "BLOCKED": {"name": "堵塞", "keywords": ["堵塞", "阻塞", "不畅"]},
    "STUCK": {"name": "卡滞", "keywords": ["卡滞", "卡死", "发卡"]},
    "LOOSE": {"name": "松动/脱落", "keywords": ["松动", "脱落", "松脱"]},
    "OPEN_CIRCUIT": {"name": "开路", "keywords": ["开路", "断路", "线路断"]},
    "SHORT_GROUND": {"name": "对地短路", "keywords": ["对地短路", "低压电源短路"]},
    "SHORT_POWER": {"name": "对电源短路", "keywords": ["对高压电源短路", "电源短路"]},
    "SIGNAL_HIGH": {"name": "信号/电压高", "keywords": ["电压高于", "信号高", "高于正常"]},
    "SIGNAL_LOW": {"name": "信号/电压低", "keywords": ["电压低于", "信号低", "低于正常"]},
    "SIGNAL_UNSTABLE": {"name": "信号不稳定/漂移", "keywords": ["信号不稳定", "数据不稳定", "漂移", "间断"]},
    "FUNCTION_FAILED": {"name": "功能失效/损坏", "keywords": ["功能失效", "失效", "损坏", "故障", "异常"]},
    "CALIBRATION": {"name": "软件/标定异常", "keywords": ["标定", "程序", "数据刷写", "升级"]},
    "OVERHEAT": {"name": "过热", "keywords": ["高温", "过热", "温度过高"]},
}

CAUSES = {
    "MATERIAL_FATIGUE": {"name": "材料疲劳", "keywords": ["疲劳", "材料缺陷"]},
    "INSUFFICIENT_LUBRICATION": {"name": "润滑不足", "keywords": ["润滑不足", "缺机油", "机油不足"]},
    "CONTAMINATION": {"name": "污染/积碳", "keywords": ["污染", "积碳", "杂质", "脏污"]},
    "THERMAL_DAMAGE": {"name": "热损伤", "keywords": ["高温导致", "过热导致", "烧蚀"]},
    "MECHANICAL_RUBBING": {"name": "机械摩擦/干涉", "keywords": ["机械摩擦", "部件干涉", "相互摩擦"]},
    "SEAL_AGING": {"name": "密封老化", "keywords": ["密封老化", "密封失效", "密封垫损坏"]},
    "ASSEMBLY_ERROR": {"name": "装配不当", "keywords": ["装配不当", "安装不到位", "安装错误"]},
    "HARNESS_INTERFERENCE": {"name": "线束干涉/磨破", "keywords": ["线束磨破", "线路磨损", "线束干涉"]},
    "MAINTENANCE_ERROR": {"name": "维护不当", "keywords": ["保养不当", "未按期保养", "维护不当"]},
    "SOFTWARE_CALIBRATION": {"name": "软件/标定", "keywords": ["程序问题", "标定问题", "软件问题"]},
    "EXTERNAL_DAMAGE": {"name": "外部损伤", "keywords": ["外力", "碰撞", "异物"]},
    "UNKNOWN": {"name": "未说明", "keywords": []},
}

SYMPTOM_KEYWORDS = [
    "动力不足", "无力", "故障灯亮", "无法启动", "启动困难", "抖动", "失火", "缺火",
    "缺缸", "限扭", "限速", "熄火", "异响", "高温", "气耗高", "白烟", "黑烟", "蓝烟",
]

AMBIGUOUS_COMPONENT_NAMES = {
    "温度传感器", "压力传感器", "压力温度传感器", "传感器", "电磁阀",
    "O型密封圈", "O型圈", "成型管", "内螺纹接头", "外螺纹接头",
    "带O型圈接头", "其它零件", "其他零件",
}


def infer_subsystem(text: str) -> str:
    normalized = normalize_text(text)
    for subsystem_id, keywords in SUBSYSTEM_KEYWORDS:
        if any(keyword.lower() in normalized.lower() for keyword in keywords):
            return subsystem_id
    return "UNKNOWN"


@dataclass
class ComponentNode:
    id: str
    name: str
    subsystem_id: str
    aliases: List[str] = field(default_factory=list)
    part_numbers: List[str] = field(default_factory=list)
    subsystem_source: str = "derived_keyword"


@dataclass
class DtcNode:
    code: str
    description: str
    local_description: str = ""
    component_id: Optional[str] = None
    failure_mode_id: Optional[str] = None


@dataclass
class KnowledgeBase:
    subsystems: Dict[str, str] = field(default_factory=lambda: dict(SUBSYSTEMS))
    components: Dict[str, ComponentNode] = field(default_factory=dict)
    dtc: Dict[str, DtcNode] = field(default_factory=dict)
    failure_modes: Dict[str, Dict[str, Any]] = field(default_factory=lambda: dict(FAILURE_MODES))
    causes: Dict[str, Dict[str, Any]] = field(default_factory=lambda: dict(CAUSES))
    symptoms: List[str] = field(default_factory=lambda: list(SYMPTOM_KEYWORDS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subsystems": self.subsystems,
            "components": {key: asdict(value) for key, value in self.components.items()},
            "dtc": {key: asdict(value) for key, value in self.dtc.items()},
            "failure_modes": self.failure_modes,
            "causes": self.causes,
            "symptoms": self.symptoms,
        }

    def save(self, path: str) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        return cls.load_from_dict(read_json(path))

    @classmethod
    def load_from_dict(cls, raw: Dict[str, Any]) -> "KnowledgeBase":
        return cls(
            subsystems=raw.get("subsystems", dict(SUBSYSTEMS)),
            components={key: ComponentNode(**value) for key, value in raw.get("components", {}).items()},
            dtc={key: DtcNode(**value) for key, value in raw.get("dtc", {}).items()},
            failure_modes=raw.get("failure_modes", dict(FAILURE_MODES)),
            causes=raw.get("causes", dict(CAUSES)),
            symptoms=raw.get("symptoms", list(SYMPTOM_KEYWORDS)),
        )

    def component_by_part_no(self, part_no: str) -> Optional[ComponentNode]:
        target = normalize_text(part_no).upper()
        return next((c for c in self.components.values() if target in {p.upper() for p in c.part_numbers}), None)

    def component_by_name(self, name: str) -> Optional[ComponentNode]:
        target = normalize_text(name).lower()
        for component in self.components.values():
            if target == component.name.lower() or target in {a.lower() for a in component.aliases}:
                return component
        return None


def _infer_failure_mode(text: str) -> Optional[str]:
    for mode_id, entry in FAILURE_MODES.items():
        if any(keyword in text for keyword in entry["keywords"]):
            return mode_id
    return None


def build_knowledge_base(
    orders: Iterable[WorkOrder],
    workbook_path: Optional[str] = None,
    dtc_sheet: str = "故障码",
    symptom_sheet: str = "故障症状",
) -> KnowledgeBase:
    kb = KnowledgeBase()
    names: Dict[str, ComponentNode] = {}
    for order in orders:
        for part in order.parts:
            if not part.part_name:
                continue
            normalized_name = normalize_text(part.part_name)
            key = normalized_name.lower()
            if key not in names:
                component = ComponentNode(
                    id=stable_id("COMP", normalized_name),
                    name=normalized_name,
                    subsystem_id=infer_subsystem(normalized_name),
                    aliases=[normalized_name],
                    part_numbers=[],
                )
                names[key] = component
                kb.components[component.id] = component
            if part.part_no and part.part_no not in names[key].part_numbers:
                names[key].part_numbers.append(part.part_no)

    if workbook_path:
        with XlsxReader(workbook_path) as workbook:
            if dtc_sheet in workbook.sheet_names():
                for _, row in workbook.rows_as_dicts(dtc_sheet):
                    code = normalize_text(row.get("FC", ""))
                    if not code.isdigit():
                        continue
                    code = str(int(code))
                    description = normalize_text(row.get("故障码描述", ""))
                    local = normalize_text(row.get("Local Description _CDS&SEDB", ""))
                    component = _best_component_from_text(kb, local + " " + description)
                    kb.dtc[code] = DtcNode(
                        code=code,
                        description=description,
                        local_description=local,
                        component_id=component.id if component else None,
                        failure_mode_id=_infer_failure_mode(description + " " + local),
                    )
            if symptom_sheet in workbook.sheet_names():
                values: Set[str] = set(kb.symptoms)
                for _, row in workbook.rows_as_dicts(symptom_sheet):
                    symptom = normalize_text(row.get("故障症状", ""))
                    if symptom:
                        values.add(symptom)
                kb.symptoms = sorted(values)
    return kb


def _best_component_from_text(kb: KnowledgeBase, text: str) -> Optional[ComponentNode]:
    candidates = []
    normalized = normalize_text(text).lower()
    for component in kb.components.values():
        for alias in set(component.aliases + [component.name]):
            if alias and alias.lower() in normalized:
                candidates.append((len(alias), component))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def build_knowledge_base_from_workbook(path: str, report_sheet: str = "Report") -> KnowledgeBase:
    orders = load_work_orders(path, report_sheet)
    return build_knowledge_base(orders, workbook_path=path)


def merge_knowledge_bases(base: KnowledgeBase, update: KnowledgeBase) -> KnowledgeBase:
    """Merge newly observed parts into a curated base without losing DTC mappings."""
    merged = KnowledgeBase.load_from_dict(base.to_dict())
    by_name = {component.name.lower(): component for component in merged.components.values()}
    for component in update.components.values():
        existing = by_name.get(component.name.lower())
        if existing:
            existing.aliases = sorted(set(existing.aliases + component.aliases))
            existing.part_numbers = sorted(set(existing.part_numbers + component.part_numbers))
            if existing.subsystem_id == "UNKNOWN" and component.subsystem_id != "UNKNOWN":
                existing.subsystem_id = component.subsystem_id
                existing.subsystem_source = component.subsystem_source
        else:
            merged.components[component.id] = component
            by_name[component.name.lower()] = component
    merged.dtc.update(update.dtc)
    merged.subsystems.update(update.subsystems)
    merged.failure_modes.update(update.failure_modes)
    merged.causes.update(update.causes)
    merged.symptoms = sorted(set(merged.symptoms + update.symptoms))
    return merged
