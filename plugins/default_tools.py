from __future__ import annotations

from typing import Dict, Any

from deepcoderesearch.tools import Tool, ToolRegistry
from deepcoderesearch.hooks import HookManager
from deepcoderesearch.rag.knowledge_base import KnowledgeBase


# 为了简单演示，这里通过一个全局变量传入 KnowledgeBase 实例
_KB_REF: KnowledgeBase | None = None


def set_kb_ref(kb: KnowledgeBase) -> None:
    global _KB_REF
    _KB_REF = kb


def _echo_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"echo": args.get("text", "")}


def _kb_search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    if _KB_REF is None:
        return {"error": "知识库尚未初始化"}
    query = str(args.get("query", ""))
    context = _KB_REF.as_context(query=query, top_k=int(args.get("top_k", 5)))
    return {"context": context}


def register(tools: ToolRegistry, hooks: HookManager) -> None:
    """默认插件：注册一些基础工具与简单 Hook。"""

    tools.register(
        Tool(
            name="echo",
            description="返回输入文本，用于测试工具调用链路是否通畅。",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要回显的文本"},
                },
                "required": ["text"],
            },
            func=_echo_tool,
        )
    )

    tools.register(
        Tool(
            name="kb_search",
            description="在当前知识库中做简单检索，返回若干相关文档片段。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询问题"},
                    "top_k": {"type": "integer", "description": "返回的片段数量", "default": 5},
                },
                "required": ["query"],
            },
            func=_kb_search_tool,
        )
    )

    # 示例 Hook：打印每个步骤的开始/结束
    def _log_pre_step(index: int, step: Any, goal: str, **_: Any) -> None:
        print(f"[Hook] 即将执行步骤 {index}: {step.description}")

    def _log_post_step(index: int, step: Any, goal: str, output: str, **_: Any) -> None:
        print(f"[Hook] 步骤 {index} 完成，输出长度: {len(output)}")

    hooks.add("pre_step", _log_pre_step)
    hooks.add("post_step", _log_post_step)
