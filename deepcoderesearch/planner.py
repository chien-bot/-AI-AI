from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import json
from .llm import BaseLLM, ChatMessage
from .memory import MemoryManager
from .tools import ToolRegistry
from .hooks import HookManager
from .rag.knowledge_base import KnowledgeBase


@dataclass
class PlanStep:
    description: str
    step_type: str = "generic"  # e.g. research / design / code / test / fix / doc
    tools: Optional[List[str]] = None  # suggested tool names like ["kb_search", "web_search"]
    role: Optional[str] = None  # optional agent role hint, e.g. "ResearchAgent"


@dataclass
class Plan:
    goal: str
    steps: List[PlanStep]


class Planner:
    """基于 LLM 的高层任务规划器。"""

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        kb: KnowledgeBase,
        memory: MemoryManager,
        hooks: HookManager,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._kb = kb
        self._memory = memory
        self._hooks = hooks

    def create_plan(self, goal: str, scenario: str = "code_review") -> Plan:
        """根据目标生成一个多步计划。

        scenario 用于给不同运行场景提供额外提示：
        - code_review: 更关注代码审查/修复流程设计
        - code_scratch: 更关注从零生成项目代码骨架和测试
        """

        self._hooks.emit("pre_planning", goal=goal)

        kb_hint = "有" if self._kb._chunks else "无"  # type: ignore[attr-defined]
        tool_names = ", ".join(t.name for t in self._tools.list_tools()) or "无"

        if scenario == "code_scratch":
            scenario_hint = (
                "当前运行场景为 code_scratch，重点是根据文档与需求生成项目代码骨架、模块划分和测试用例，"
                "而不是设计代码审查/修复系统本身。"
            )
        elif scenario == "code_review":
            scenario_hint = (
                "当前运行场景为 code_review，重点是设计和实现代码审查与自动修复相关的模块与流程。"
            )
        elif scenario == "doc_research":
            scenario_hint = (
                "当前运行场景为 doc_research，重点是对多份技术文档和知识库进行系统性深度研究、需求/架构分析与文档撰写，"
                "不需要生成项目代码或测试。"
            )
        else:
            scenario_hint = f"当前运行场景为 {scenario}。"

        system = ChatMessage(
            role="system",
            content=(
                "你是一名软件架构师级别的 AI Agent Planner。"
                "需要为复杂代码生成任务设计清晰的执行计划，并为每个步骤标注类型、建议工具和负责的子 Agent 角色。"
                + scenario_hint
            ),
        )
        user = ChatMessage(
            role="user",
            content=(
                "当前任务目标如下：\n"
                f"{goal}\n\n"
                "可用资源：\n"
                f"- 知识库: {kb_hint}\n"
                f"- 工具: {tool_names}\n\n"
                "请输出一个多步骤计划，并使用 JSON 结构返回，以便后续程序解析。\n"
                "要求：\n"
                "1. 步骤要覆盖：文档研究、需求归纳、架构设计、代码生成、测试/自我检查与文档撰写等。\n"
                "2. 每个步骤需包含以下字段：\n"
                '   - \"description\": 对步骤要做事情的自然语言描述；\n'
                '   - \"step_type\": 从 [\"research\", \"design\", \"code\", \"test\", \"fix\", \"doc\"] 中选择一个最合适的类型；\n'
                '   - \"tools\": 建议使用的工具名称列表，例如 [\"kb_search\", \"web_search\", \"run_python\"]，也可以为空列表；\n'
                '   - \"role\": 负责该步骤的子 Agent 角色名，例如 \"ResearchAgent\", \"DesignAgent\", \"CodeAgent\", \"TestAgent\", \"FixAgent\", \"DocAgent\" 等。\n'
                "3. 返回格式可以是一个 JSON 数组，或一个带 \"steps\" 字段的 JSON 对象，例如：\n"
                "[\n"
                "  {\"description\": \"文档研究：阅读参考资料…\", \"step_type\": \"research\", \"tools\": [\"kb_search\", \"web_search\"], \"role\": \"ResearchAgent\"},\n"
                "  {\"description\": \"根据研究结果整理需求…\", \"step_type\": \"design\", \"tools\": [], \"role\": \"DesignAgent\"}\n"
                "]\n"
                "4. 请只输出 JSON，本身不要再包裹代码块、前后说明文字。"
            ),
        )

        plan_text = self._llm.chat([system, user])
        candidate = _extract_json_candidate(plan_text)
        # 优先尝试解析 JSON 结构化计划，失败时回退到旧的纯文本解析。
        steps = _parse_plan_json(candidate)
        if not steps:
            steps = _parse_plan_text(plan_text)

        plan = Plan(goal=goal, steps=steps)

        self._hooks.emit("post_planning", goal=goal, plan=plan)
        self._memory.add_event("plan_created", {"goal": goal, "plan_text": plan_text})
        return plan


def _parse_plan_text(text: str) -> List[PlanStep]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    steps: List[PlanStep] = []
    for line in lines:
        # 兼容 "1." "1)" "步骤1:" 等多种格式
        cleaned = line
        for prefix in ["步骤", "Step "]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip()
        # 去掉前缀序号
        while cleaned and (cleaned[0].isdigit() or cleaned[0] in ".)、"):
            cleaned = cleaned[1:].lstrip()
        if cleaned:
            steps.append(PlanStep(description=cleaned))
    if not steps:
        steps.append(PlanStep(description=text.strip() or "研究并完成任务"))
    return steps


def _extract_json_candidate(raw: str) -> str:
    """从原始 LLM 输出中尽量提取 JSON 主体。

    许多模型会将 JSON 包裹在 ```json ... ``` 代码块中，
    这里尽量剥离这些包裹，只保留 { ... } 或 [ ... ]。
    """

    text = raw.strip()

    # Fast path: already looks like bare JSON object or array.
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return text

    if "```" in text:
        first = text.find("```")
        second = text.rfind("```")
        if second != -1 and second > first:
            inner = text[first + 3 : second].strip()
            # Possible leading language tag, e.g. "json\n{...}" or "json\n[...]"
            lower = inner.lower()
            if lower.startswith("json"):
                inner = inner[4:].lstrip("\r\n ")
            inner = inner.strip()
            if (inner.startswith("{") and inner.endswith("}")) or (
                inner.startswith("[") and inner.endswith("]")
            ):
                return inner

    return text


def _parse_plan_json(text: str) -> List[PlanStep]:
    """尝试将 LLM 输出解析为结构化 JSON 计划。

    支持两种格式：
    1) 直接是一个 JSON 数组: [{...}, {...}]
    2) 顶层是对象，包含 \"steps\" 字段: {\"steps\": [{...}, ...]}
    """

    text = text.strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except Exception:
        return []

    if isinstance(data, dict) and "steps" in data:
        items = data.get("steps")
    else:
        items = data

    if not isinstance(items, list):
        return []

    steps: List[PlanStep] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description") or "").strip()
        if not desc:
            continue
        step_type = str(item.get("step_type") or "generic").strip() or "generic"
        tools_val = item.get("tools")
        tools: Optional[List[str]] = None
        if isinstance(tools_val, list):
            filtered = [str(t).strip() for t in tools_val if str(t).strip()]
            if filtered:
                tools = filtered
        role_val = item.get("role")
        role = str(role_val).strip() or None if role_val is not None else None

        steps.append(PlanStep(description=desc, step_type=step_type, tools=tools, role=role))

    return steps
