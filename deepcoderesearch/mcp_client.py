from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass
class MCPToolConfig:
    """MCP/HTTP 工具配置。

    这里采用一个极简版抽象：
    - 通过 `url` 指定远程服务地址
    - 通过 `method` 指定 HTTP 方法
    - `headers` 用于携带认证信息（例如从环境变量拼装）
    - `input_schema` 与 Tool 中保持一致，方便自动透传
    """

    name: str
    description: str
    url: str
    method: str = "POST"
    headers: Optional[Dict[str, str]] = None
    input_schema: Dict[str, Any] | None = None


class MCPClient:
    """极简 MCP/远程工具调用客户端。"""

    def call(self, config: MCPToolConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
        method = config.method.upper()
        headers = config.headers or {"Content-Type": "application/json"}

        resp = requests.request(method, config.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            data = {"raw": resp.text}
        return {"result": data}


def load_openapi_schema(path: str | Path) -> Dict[str, Any]:
    """从本地或远程路径加载 OpenAPI JSON。

    仅用于示例，复杂场景可自行扩展。
    """

    import requests  # 局部导入以避免无网络时的开销

    p = str(path)
    if p.startswith("http://") or p.startswith("https://"):
        resp = requests.get(p, timeout=60)
        resp.raise_for_status()
        return resp.json()
    else:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
