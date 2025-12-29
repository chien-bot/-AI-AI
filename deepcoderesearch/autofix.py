from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import ast
import json

from .llm import BaseLLM, ChatMessage
from .memory import MemoryManager


@dataclass
class AutoFixResult:
    """Result of a single auto-fix attempt."""

    changed_files: List[str]
    raw_response: str
    parse_error: Optional[str] = None


def _collect_repo_context(repo_root: Path, max_chars_per_file: int = 2000) -> str:
    """Collect a compact textual view of the repo for the LLM.

    We only read a subset of files (.py and tests) and truncate each file
    to avoid over-long prompts.
    """

    lines: List[str] = []
    for path in sorted(repo_root.rglob("*.py")):
        rel = path.relative_to(repo_root)
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        snippet = content[:max_chars_per_file]
        lines.append(f"[File: {rel}]")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


def _format_issues_for_llm(issues: List[Dict[str, Any]], max_issues: int = 100) -> str:
    """Render a unified issues list into a compact human-readable text block.

    期望 issues 中的字段包括：file, line, column, tool, code, message。
    即使部分字段缺失，也尽量容错展示。
    """

    lines: List[str] = []
    for idx, issue in enumerate(issues[:max_issues], start=1):
        file = str(issue.get("file", "") or "").strip()
        line = issue.get("line")
        column = issue.get("column")
        tool = str(issue.get("tool", "") or "").strip()
        code = str(issue.get("code", "") or "").strip()
        message = str(issue.get("message", "") or "").strip()

        loc = file or "<unknown>"
        if isinstance(line, int) and line > 0:
            loc += f":{line}"
            if isinstance(column, int) and column > 0:
                loc += f":{column}"

        prefix_parts = []
        if tool:
            prefix_parts.append(tool)
        if code:
            prefix_parts.append(code)
        prefix = " ".join(prefix_parts) if prefix_parts else "issue"

        lines.append(f"[{idx}] {loc} - {prefix}: {message}")

    if len(issues) > max_issues:
        lines.append(f"... ({len(issues) - max_issues} more issues truncated)")

    return "\n".join(lines)


def attempt_autofix(
    llm: BaseLLM,
    repo_root: Path,
    coding_task: str,
    test_stdout: str,
    test_stderr: str,
    memory: Optional[MemoryManager] = None,
) -> AutoFixResult:
    """Try to automatically repair the demo repo based on failing tests.

    The LLM is asked to output a Python dict literal with a `changes` list,
    where each item has `path` and `content` fields. Paths are repo-relative
    file paths whose full contents will be replaced by `content`.
    """

    repo_root = repo_root.resolve()
    context = _collect_repo_context(repo_root)

    system = ChatMessage(
        role="system",
        content=(
            "你是一名经验丰富的 Python 工程师，当前需要根据单元测试失败信息来修复代码。"
            "请只返回可执行的修改方案，不要解释性的文字。"
        ),
    )

    user = ChatMessage(
        role="user",
        content=(
            "当前要优化的 Coding 题目是:\n"
            f"{coding_task}\n\n"
            "单元测试失败的输出如下(stdout + stderr):\n"
            f"STDOUT:\n{test_stdout}\n\nSTDERR:\n{test_stderr}\n\n"
            "下面是当前代码仓库的部分文件内容（请根据需要选择性修改）：\n"
            f"{context}\n\n"
            "请根据以上信息给出修复方案，直接返回一个 Python 字典字面量，不要添加解释文字，"
            "也不要使用代码块。格式示例:\n"
            "{\n"
            "  'changes': [\n"
            "    {'path': 'multimodal_code_gen/agent.py', 'content': '新的完整文件内容...'},\n"
            "    {'path': 'tests/test_pipeline.py', 'content': '新的完整文件内容...'}\n"
            "  ]\n"
            "}\n"
        ),
    )

    raw = llm.chat([system, user])

    changed: List[str] = []
    parse_error: Optional[str] = None

    try:
        data = ast.literal_eval(raw)
        changes = data.get("changes", []) if isinstance(data, dict) else []
        if isinstance(changes, list):
            for item in changes:
                if not isinstance(item, dict):
                    continue
                rel = str(item.get("path", "")).strip()
                content = str(item.get("content", ""))
                if not rel:
                    continue
                file_path = repo_root / rel
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                changed.append(rel)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)

    result = AutoFixResult(changed_files=changed, raw_response=raw, parse_error=parse_error)

    if memory is not None:
        memory.add_event(
            "autofix_attempt",
            {
                "repo_root": str(repo_root),
                "changed_files": changed,
                "parse_error": parse_error,
            },
        )

    return result


def attempt_autofix_from_issues(
    llm: BaseLLM,
    repo_root: Path,
    coding_task: str,
    issues: List[Dict[str, Any]],
    memory: Optional[MemoryManager] = None,
) -> AutoFixResult:
    """Try to automatically repair the repo based on static analysis issues.

    这里不依赖测试输出，而是依赖统一 schema 的 issues 列表，例如：
    {
      "file": "xxx.py",
      "line": 42,
      "column": 1,
      "tool": "flake8",
      "code": "E302",
      "message": "expected 2 blank lines"
    }

    LLM 会被要求输出和 attempt_autofix 相同结构的 Python 字典，包含
    changes 列表，每个元素给出需要整体替换的文件内容。
    """

    repo_root = repo_root.resolve()
    context = _collect_repo_context(repo_root)
    issues_text = _format_issues_for_llm(issues)
    issues_json = json.dumps(issues, ensure_ascii=False, indent=2)

    system = ChatMessage(
        role="system",
        content=(
            "你是一名经验丰富的 Python 代码审查与修复工程师，当前需要根据静态分析工具"
            "发现的问题来修复代码。请只返回可执行的修改方案，不要解释性的文字。"
        ),
    )

    user = ChatMessage(
        role="user",
        content=(
            "当前要优化的 Coding 题目是:\n"
            f"{coding_task}\n\n"
            "静态分析工具发现的问题列表如下（人类可读形式）:\n"
            f"{issues_text}\n\n"
            "同样的问题列表也以 JSON 形式提供，方便你精确定位：\n"
            f"{issues_json}\n\n"
            "下面是当前代码仓库的部分文件内容（请根据需要选择性修改）：\n"
            f"{context}\n\n"
            "请根据以上信息给出修复方案，直接返回一个 Python 字典字面量，不要添加解释文字，"
            "也不要使用代码块。格式示例:\n"
            "{\n"
            "  'changes': [\n"
            "    {'path': 'some_module.py', 'content': '新的完整文件内容...'}\n"
            "  ]\n"
            "}\n"
        ),
    )

    raw = llm.chat([system, user])

    changed: List[str] = []
    parse_error: Optional[str] = None

    try:
        data = ast.literal_eval(raw)
        changes = data.get("changes", []) if isinstance(data, dict) else []
        if isinstance(changes, list):
            for item in changes:
                if not isinstance(item, dict):
                    continue
                rel = str(item.get("path", "")).strip()
                content = str(item.get("content", ""))
                if not rel:
                    continue
                file_path = repo_root / rel
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                changed.append(rel)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)

    result = AutoFixResult(changed_files=changed, raw_response=raw, parse_error=parse_error)

    if memory is not None:
        memory.add_event(
            "autofix_from_issues",
            {
                "repo_root": str(repo_root),
                "changed_files": changed,
                "parse_error": parse_error,
                "issue_count": len(issues),
            },
        )

    return result
