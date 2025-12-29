from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List


ToolFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Tool:
    """工具定义，兼容函数调用 / MCP 风格调用。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    func: ToolFunc


class ToolRegistry:
    """工具注册表，负责管理和查找工具。

    插件只需要拿到一个 registry 实例即可注册新工具。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已存在")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"未找到工具: {name}")
        return self._tools[name]

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def as_openai_tools(self) -> List[Dict[str, Any]]:
        """将工具列表转换为 OpenAI function calling 兼容格式。"""

        converted: List[Dict[str, Any]] = []
        for t in self._tools.values():
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
            )
        return converted
