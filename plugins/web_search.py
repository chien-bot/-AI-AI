from __future__ import annotations

import os
from typing import Dict, Any

from deepcoderesearch.tools import Tool, ToolRegistry
from deepcoderesearch.hooks import HookManager
from deepcoderesearch.mcp_client import MCPClient, MCPToolConfig


_client = MCPClient()


def _web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """通过 MCPClient 调用外部搜索服务的示例工具。

    实际搜索实现由外部服务负责，这里只做 HTTP 转发。\n
    约定环境变量：\n
    - `WEB_SEARCH_ENDPOINT`: 例如 https://api.example.com/search
    - `WEB_SEARCH_AUTH_HEADER`: 可选，形如 "Bearer xxx"，作为 Authorization 头
    """

    endpoint = os.getenv("WEB_SEARCH_ENDPOINT")
    if not endpoint:
        return {"error": "未配置 WEB_SEARCH_ENDPOINT 环境变量"}

    auth_header = os.getenv("WEB_SEARCH_AUTH_HEADER")
    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    config = MCPToolConfig(
        name="web_search",
        description="通用 Web 搜索服务",
        url=endpoint,
        method="POST",
        headers=headers,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    )

    payload = {
        "query": args.get("query", ""),
        "top_k": int(args.get("top_k", 5)),
    }

    return _client.call(config, payload)


def register(tools: ToolRegistry, hooks: HookManager) -> None:
    tools.register(
        Tool(
            name="web_search",
            description="调用外部 Web 搜索服务 (通过 MCP/HTTP 封装)",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回结果数", "default": 5},
                },
                "required": ["query"],
            },
            func=_web_search,
        )
    )
