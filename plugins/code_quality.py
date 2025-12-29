from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

from deepcoderesearch.tools import Tool, ToolRegistry
from deepcoderesearch.hooks import HookManager


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    tool_name: str,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Execute an external command and capture stdout/stderr.

    This helper is defensive: if工具未安装或执行失败，会在结果中返回 error 字段，
    而不是抛异常中断主流程。
    """

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"error": f"{tool_name} not found", "issues": []}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{tool_name} execution failed: {exc}", "issues": []}

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _parse_flake8(stdout: str) -> List[Dict[str, Any]]:
    """Parse flake8 text output into a unified issues list."""

    issues: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected format: file:line:col: code message
        parts = line.split(":", 3)
        if len(parts) < 4:
            continue
        file_path, line_str, col_str, rest = parts
        file_path = file_path.strip()
        try:
            line_no = int(line_str.strip())
        except ValueError:
            line_no = 0
        try:
            col_no = int(col_str.strip())
        except ValueError:
            col_no = 0
        rest = rest.strip()
        if " " in rest:
            code, message = rest.split(" ", 1)
        else:
            code, message = rest, ""
        issues.append(
            {
                "file": file_path,
                "line": line_no,
                "column": col_no,
                "tool": "flake8",
                "code": code,
                "message": message,
            }
        )
    return issues


def _parse_mypy(stdout: str) -> List[Dict[str, Any]]:
    """Parse mypy text output into a unified issues list."""

    issues: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        # Typical format: file:line: column: message
        parts = line.split(":", 3)
        if len(parts) < 4:
            continue
        file_path, line_str, col_str, message = parts
        file_path = file_path.strip()
        try:
            line_no = int(line_str.strip())
        except ValueError:
            line_no = 0
        try:
            col_no = int(col_str.strip())
        except ValueError:
            col_no = 0
        issues.append(
            {
                "file": file_path,
                "line": line_no,
                "column": col_no,
                "tool": "mypy",
                "code": "",
                "message": message.strip(),
            }
        )
    return issues


def _run_flake8(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run flake8 on a given path and return unified issues."""

    path_str = str(args.get("path") or ".")
    cwd = Path(path_str).resolve()
    # Use python -m flake8 to ensure we use the venv Python
    result = _run_cmd(
        [sys.executable, "-m", "flake8", "."],
        cwd=cwd,
        tool_name="flake8",
    )
    stdout = str(result.get("stdout", ""))
    issues = _parse_flake8(stdout) if stdout else []
    result["issues"] = issues
    return result


def _run_pylint(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run pylint; return raw output in unified format."""

    path_str = str(args.get("path") or ".")
    cwd = Path(path_str).resolve()
    result = _run_cmd(
        [sys.executable, "-m", "pylint", cwd.as_posix()],
        cwd=cwd,
        tool_name="pylint",
    )
    stdout = str(result.get("stdout", ""))
    issues: List[Dict[str, Any]] = []
    if stdout:
        issues.append(
            {
                "file": "",
                "line": 0,
                "column": 0,
                "tool": "pylint",
                "code": "",
                "message": stdout.strip(),
            }
        )
    result["issues"] = issues
    return result


def _run_bandit(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run bandit security scanner on the given path."""

    path_str = str(args.get("path") or ".")
    cwd = Path(path_str).resolve()
    # -q to reduce noise, -r to recurse
    result = _run_cmd(
        [sys.executable, "-m", "bandit", "-q", "-r", "."],
        cwd=cwd,
        tool_name="bandit",
    )
    stdout = str(result.get("stdout", ""))
    issues: List[Dict[str, Any]] = []
    if stdout:
        issues.append(
            {
                "file": "",
                "line": 0,
                "column": 0,
                "tool": "bandit",
                "code": "",
                "message": stdout.strip(),
            }
        )
    result["issues"] = issues
    return result


def _run_mypy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run mypy on the given path."""

    path_str = str(args.get("path") or ".")
    cwd = Path(path_str).resolve()
    result = _run_cmd(
        [sys.executable, "-m", "mypy", "."],
        cwd=cwd,
        tool_name="mypy",
    )
    stdout = str(result.get("stdout", ""))
    issues = _parse_mypy(stdout) if stdout else []
    result["issues"] = issues
    return result


def register(tools: ToolRegistry, hooks: HookManager) -> None:
    """Register static code quality tools as callable Agent tools."""

    tools.register(
        Tool(
            name="run_flake8",
            description="Run flake8 on a given path and return lint issues.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to lint; default: current directory.",
                    },
                },
                "required": [],
            },
            func=_run_flake8,
        )
    )

    tools.register(
        Tool(
            name="run_pylint",
            description="Run pylint on a given path and return issues summary.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to analyze; default: current directory.",
                    },
                },
                "required": [],
            },
            func=_run_pylint,
        )
    )

    tools.register(
        Tool(
            name="run_bandit",
            description="Run bandit security analysis on a given path and return issues summary.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to scan; default: current directory.",
                    },
                },
                "required": [],
            },
            func=_run_bandit,
        )
    )

    tools.register(
        Tool(
            name="run_mypy",
            description="Run mypy type checker on a given path and return issues.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to type-check; default: current directory.",
                    },
                },
                "required": [],
            },
            func=_run_mypy,
        )
    )

