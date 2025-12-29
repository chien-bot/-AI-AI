from __future__ import annotations

import os
from typing import Any, Dict
import time

from .llm import BaseLLM, ChatMessage
from .memory import MemoryManager
from .tools import ToolRegistry
from .hooks import HookManager
from .rag.knowledge_base import KnowledgeBase
from .planner import Plan, PlanStep
from .reflection import Reflector


class Executor:
    """执行器：顺序执行 Planner 生成的步骤。

    这里实现一个相对轻量的执行逻辑：
    - 对每个步骤，先从知识库中检索相关上下文
    - 再通过工具（kb_search / web_search 等）做补充检索
    - 然后调用 LLM，让其在上下文基础上完成该步骤
    - 将输出写入 Memory，遇到异常时调用 Reflector 做简单自我反思

    若要实现更强大的 Code Agent（自动写文件/跑测试等），
    可以在这里调用 ToolRegistry 中注册的工具。
    """

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
        self._reflector = Reflector(llm)

    def execute_plan(self, plan: Plan, goal: str) -> None:
        for idx, step in enumerate(plan.steps, start=1):
            step_label = f"Step {idx}: {step.description}"
            self._hooks.emit("pre_step", index=idx, step=step, goal=goal)
            print(f"[Executor] {step_label}")

            kb_context = self._kb.as_context(query=step.description, top_k=5)
            tool_context = self._gather_tool_context(step)

            start = time.perf_counter()
            try:
                output = self._run_step_llm(
                    goal=goal,
                    step=step,
                    kb_context=kb_context,
                    tool_context=tool_context,
                )
                duration = time.perf_counter() - start
                # 将耗时挂到 PlanStep 上，便于后续报告使用
                try:
                    step.duration_seconds = duration  # type: ignore[attr-defined]
                except Exception:
                    pass

                self._memory.add_event(
                    "step_executed",
                    {
                        "index": idx,
                        "description": step.description,
                        "step_type": getattr(step, "step_type", None),
                        "tools": getattr(step, "tools", None),
                        "role": getattr(step, "role", None),
                        "output": output,
                        "tool_context": tool_context,
                        "duration_seconds": duration,
                    },
                )
                self._hooks.emit("post_step", index=idx, step=step, goal=goal, output=output)

                print("[Executor] 步骤输出摘要:")
                print(output[:500])
            except Exception as exc:
                duration = time.perf_counter() - start
                try:
                    step.duration_seconds = duration  # type: ignore[attr-defined]
                except Exception:
                    pass

                err_msg = str(exc)
                print(f"[Executor] 步骤执行失败: {err_msg}")
                reflection = self._reflector.reflect(
                    goal=goal,
                    step=step.description,
                    error=err_msg,
                )
                self._memory.add_event(
                    "step_error",
                    {
                        "index": idx,
                        "description": step.description,
                        "step_type": getattr(step, "step_type", None),
                        "tools": getattr(step, "tools", None),
                        "role": getattr(step, "role", None),
                        "error": err_msg,
                        "reflection": reflection,
                        "duration_seconds": duration,
                    },
                )
                self._hooks.emit(
                    "on_error",
                    index=idx,
                    step=step,
                    goal=goal,
                    error=err_msg,
                    reflection=reflection,
                )
                # 这里选择不中断整个流程，根据需求你也可以 raise 终止

    # -------- 内部实现 --------
    def _gather_tool_context(self, step: PlanStep) -> str:
        """调用部分工具（kb_search / web_search），把结果作为额外上下文。

        这里不做复杂的工具规划，只是演示性地：
        - 总是尝试调用 kb_search
        - 在配置了 WEB_SEARCH_ENDPOINT 时调用 web_search
        工具结果会被拼接到 LLM 的输入中，真正参与推理。
        """

        snippets = []
        step_desc = step.description
        requested_tools = set(step.tools or []) if getattr(step, "tools", None) else set()

        # kb_search（如果未指定 tools，则默认启用；若指定则仅在包含时启用）
        use_kb = not requested_tools or "kb_search" in requested_tools
        if use_kb:
            try:
                kb_tool = self._tools.get("kb_search")
            except KeyError:
                kb_tool = None
            if kb_tool is not None:
                try:
                    kb_result: Dict[str, Any] = kb_tool.func({"query": step_desc, "top_k": 5})
                    context = kb_result.get("context")
                    if isinstance(context, str) and context.strip():
                        snippets.append("[kb_search 结果]\n" + context.strip())
                except Exception as exc:  # pragma: no cover - 工具失败不应打断主流程
                    snippets.append(f"[kb_search 调用失败: {exc}]")

        # web_search（只在配置了端点时尝试调用；同样受 tools 提示控制）
        use_web = (not requested_tools or "web_search" in requested_tools) and os.getenv(
            "WEB_SEARCH_ENDPOINT"
        )
        if use_web:
            try:
                web_tool = self._tools.get("web_search")
            except KeyError:
                web_tool = None
            if web_tool is not None:
                try:
                    web_result: Dict[str, Any] = web_tool.func({"query": step_desc, "top_k": 3})
                    snippets.append("[web_search 结果]\n" + str(web_result.get("result", web_result)))
                except Exception as exc:  # pragma: no cover
                    snippets.append(f"[web_search 调用失败: {exc}]")

        return "\n\n".join(snippets)

    def _run_step_llm(self, goal: str, step: PlanStep, kb_context: str, tool_context: str) -> str:
        # 根据步骤类型和角色构造更细粒度的 system prompt，支持多 Agent 角色风格。
        base_role = "你是一个执行复杂软件工程任务的高级 Agent 子模块"
        if getattr(step, "role", None):
            role_desc = f"你当前扮演的子 Agent 角色是：{step.role}。"
        else:
            role_desc = ""
        type_desc = ""
        if getattr(step, "step_type", None):
            type_desc = f"当前步骤类型为：{step.step_type}。"

        system = ChatMessage(
            role="system",
            content=(
                f"{base_role}，需要在给定上下文基础上完成当前步骤。"
                f"{role_desc}{type_desc}"
            ),
        )

        parts = []
        if kb_context:
            parts.append(f"相关文档片段:\n{kb_context}")
        if tool_context:
            parts.append(f"工具调用结果:\n{tool_context}")
        context_block = "\n\n".join(parts)
        if context_block:
            context_block += "\n\n"

        user = ChatMessage(
            role="user",
            content=(
                f"总体任务目标: {goal}\n"
                f"当前步骤: {step.description}\n"
                f"步骤类型: {getattr(step, 'step_type', 'generic')}\n"
                f"建议工具: {', '.join(step.tools) if getattr(step, 'tools', None) else '未指定'}\n"
                f"子 Agent 角色: {getattr(step, 'role', '未指定')}\n\n"
                f"{context_block}"
                "请基于以上信息完成当前步骤，输出：\n"
                "- 对本步骤要做事情的简要说明\n"
                "- 关键结论 / 设计要点 / 代码骨架建议等\n"
            ),
        )

        return self._llm.chat([system, user])
