from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Dict, Any

from deepcoderesearch.tools import Tool, ToolRegistry
from deepcoderesearch.hooks import HookManager


def _run_python_snippet(args: Dict[str, Any]) -> Dict[str, Any]:
    """在临时文件中执行一段 Python 代码的简单示例。

    出于安全考虑，只建议在受控环境/本地实验使用。比赛中可根据需求调整。
    """

    code = str(args.get("code", ""))
    workdir = Path(args.get("workdir") or ".").resolve()

    workdir.mkdir(parents=True, exist_ok=True)
    tmp_file = workdir / "_agent_snippet.py"
    tmp_file.write_text(code, encoding="utf-8")

    try:
        proc = subprocess.run(
            ["python", str(tmp_file)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return {"error": f"执行失败: {exc}"}

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def register(tools: ToolRegistry, hooks: HookManager) -> None:
    tools.register(
        Tool(
            name="run_python",
            description=(
                "在本地以子进程形式执行一段 Python 代码，仅用于实验性 Code Agent 能力演示。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"},
                    "workdir": {
                        "type": "string",
                        "description": "可选，代码执行的工作目录，默认当前目录",
                    },
                },
                "required": ["code"],
            },
            func=_run_python_snippet,
        )
    )
