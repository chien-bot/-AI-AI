import argparse
from pathlib import Path
import json
import concurrent.futures
import time

from deepcoderesearch.config import AgentConfig, LLMConfig
from typing import Callable, List, Dict, Any
from deepcoderesearch.llm import build_llm
from deepcoderesearch.memory import MemoryManager
from deepcoderesearch.rag.knowledge_base import KnowledgeBase
from deepcoderesearch.tools import ToolRegistry
from deepcoderesearch.hooks import HookManager
from deepcoderesearch.planner import Planner
from deepcoderesearch.executor import Executor
from deepcoderesearch.codegen import generate_demo_repo
from deepcoderesearch.reflection import Reflector
from deepcoderesearch.autofix import attempt_autofix, attempt_autofix_from_issues
from plugins import load_plugins, default_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepCodeResearch: 复杂代码生成 Agent 参考实现",
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="本次任务的目标，例如：为某业务设计代码生成方案",
    )
    parser.add_argument(
        "--coding-task",
        default="实现一个将自然语言需求转换为 Python 函数骨架的多 Agent 代码生成 demo 框架",
        help="本次要解决的具体 Coding 题目描述，将用于指导代码仓库与测试生成",
    )
    parser.add_argument(
        "--docs",
        nargs="*",
        default=[],
        help="技术文档路径列表（PDF/PPTX/DOCX/MD/TXT），可选",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="会话 ID，用于在多次运行间共享记忆；默认自动生成",
    )
    parser.add_argument(
        "--scenario",
        default="code_review",
        choices=["code_review", "code_scratch", "doc_research"],
        help="运行场景：code_review(默认)、code_scratch、doc_research",
    )
    return parser.parse_args()


def _slugify_task(text: str) -> str:
    """将任意 Coding 题目描述转成安全的目录名（snake_case）。"""
    import re

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "multimodal_deepresearch_demo"
    # 避免目录名过长
    return text[:80]


def _build_openai_embedder(llm_config: LLMConfig) -> Callable[[str], List[float]] | None:
    """尝试基于当前 LLM 配置构建一个 OpenAI / ModelScope embeddings 客户端。

    若环境中缺少 openai 依赖或未配置 API Key，则返回 None，保持 RAG 回退为词重叠检索。
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    if not llm_config.api_key:
        return None

    import os

    client_kwargs: dict = {"api_key": llm_config.api_key}
    if llm_config.base_url:
        client_kwargs["base_url"] = llm_config.base_url

    client = OpenAI(**client_kwargs)

    # 允许通过环境变量覆盖 embedding 模型名称，默认使用 OpenAI 的轻量模型
    embed_model = (
        os.getenv("EMBEDDING_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )

    def embed(text: str) -> List[float]:
        resp = client.embeddings.create(model=embed_model, input=[text])
        return list(resp.data[0].embedding)

    return embed


def _run_static_tool(tool, path: Path) -> Dict[str, Any]:
    """Helper to run a static analysis tool and measure its duration."""

    start = time.perf_counter()
    result = tool.func({"path": str(path)})
    duration = time.perf_counter() - start
    if not isinstance(result, dict):
        result = {"raw": result}
    # Make a shallow copy to avoid mutating internal structures
    out: Dict[str, Any] = dict(result)
    out["_duration"] = duration
    return out


def _write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    """Render a human-friendly Markdown report from the JSON payload."""

    goal = report.get("goal", "")
    coding_task = report.get("coding_task", "")
    session_id = report.get("session_id", "")
    output_dir = report.get("output_dir", "")

    llm = report.get("llm") or {}
    llm_provider = llm.get("provider", "")
    llm_model = llm.get("model", "")

    plan = report.get("plan") or {}
    steps = plan.get("steps") or []

    static_analysis = report.get("static_analysis") or {}
    static_tools = static_analysis.get("tools") or []

    tests = report.get("tests") or {}
    initial_tests = tests.get("initial") or {}

    autofix = report.get("autofix") or {}
    autofix_static = autofix.get("static_issues") or {}
    autofix_tests = autofix.get("tests") or {}

    lines: List[str] = []
    lines.append(f"# DeepCodeResearch 运行报告 ({session_id})")
    lines.append("")
    lines.append(f"- 目标: `{goal}`")
    lines.append(f"- Coding Task: `{coding_task}`")
    lines.append(f"- 输出目录: `{output_dir}`")
    lines.append("")

    lines.append("## LLM 信息")
    lines.append("")
    lines.append(f"- Provider: `{llm_provider}`")
    lines.append(f"- Model: `{llm_model}`")
    if llm.get("base_url"):
        lines.append(f"- Base URL: `{llm.get('base_url')}`")
    lines.append("")

    lines.append("## 规划步骤耗时")
    lines.append("")
    if steps:
        lines.append("| # | 类型 | 角色 | 描述 | 耗时 (s) |")
        lines.append("|---|------|------|------|----------|")
        for step in steps:
            idx = step.get("index")
            step_type = step.get("step_type") or ""
            role = step.get("role") or ""
            desc = step.get("description") or ""
            dur = step.get("duration_seconds")
            dur_str = f"{dur:.3f}" if isinstance(dur, (int, float)) else ""
            # 简要截断描述，避免过长
            short_desc = desc.replace("\n", " ")
            if len(short_desc) > 40:
                short_desc = short_desc[:37] + "..."
            lines.append(f"| {idx} | {step_type} | {role} | {short_desc} | {dur_str} |")
    else:
        lines.append("本次未记录到规划步骤信息。")
    lines.append("")

    lines.append("## 静态分析工具耗时与结果")
    lines.append("")
    if static_tools:
        lines.append("| 工具 | 耗时 (s) | 问题数 | 错误 |")
        lines.append("|------|----------|--------|------|")
        for tinfo in static_tools:
            name = tinfo.get("name") or ""
            dur = tinfo.get("duration_seconds")
            dur_str = f"{dur:.3f}" if isinstance(dur, (int, float)) else ""
            issue_count = tinfo.get("issue_count", 0)
            err = tinfo.get("error") or ""
            short_err = err.replace("\n", " ")
            if len(short_err) > 40:
                short_err = short_err[:37] + "..."
            lines.append(f"| {name} | {dur_str} | {issue_count} | {short_err} |")
    else:
        lines.append("本次未运行静态分析工具。")
    lines.append("")

    lines.append("## 测试结果")
    lines.append("")
    rc = initial_tests.get("returncode")
    lines.append(f"- 初始测试返回码: `{rc}`")
    stderr = initial_tests.get("stderr") or ""
    if stderr:
        lines.append("")
        lines.append("```text")
        lines.append(str(stderr).strip())
        lines.append("```")
    lines.append("")

    lines.append("## 自动修复情况")
    lines.append("")
    lines.append("- 静态分析自动修复:")
    lines.append(f"  - 启用: `{autofix_static.get('enabled', False)}`")
    lines.append(f"  - 已尝试: `{autofix_static.get('attempted', False)}`")
    lines.append(f"  - 修改文件: `{autofix_static.get('changed_files', [])}`")
    lines.append("")
    lines.append("- 测试失败自动修复:")
    lines.append(f"  - 启用: `{autofix_tests.get('enabled', False)}`")
    lines.append(f"  - 已尝试: `{autofix_tests.get('attempted', False)}`")
    lines.append(f"  - 修改文件: `{autofix_tests.get('changed_files', [])}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    # 优先从项目级配置文件（code_review_agent.yml）加载配置，
    # 若文件不存在则回退到默认配置，仅依赖当前工作目录与环境变量。
    config = AgentConfig.from_file()
    memory = MemoryManager(config=config, session_id=args.session_id)

    kb = KnowledgeBase()

    # 1) 如果命令行显式给了 --docs，就按用户指定的路径加载
    doc_paths = []
    if args.docs:
        doc_paths = [Path(p) for p in args.docs]
    else:
        # 2) 否则，如果当前目录下存在 references/，默认加载其中的参考文档
        default_dir = Path("references")
        if default_dir.exists():
            patterns = ["*.pdf", "*.docx", "*.pptx", "*.txt", "*.md"]
            for pattern in patterns:
                doc_paths.extend(default_dir.rglob(pattern))

    if doc_paths:
        print(f"[Agent] 检测到 {len(doc_paths)} 个参考文档，将作为知识库载入。")
        kb.ingest_documents(doc_paths)
    else:
        print("[Agent] 未检测到参考文档，将仅基于模型自身知识进行推理。")

    # 尝试基于当前 LLM 配置启用 OpenAI / ModelScope 向量检索（如果可用）
    embedder = _build_openai_embedder(config.llm)
    if embedder is not None:
        kb.set_embedder(embedder)
        print("[RAG] 已启用 OpenAI/ModelScope 向量检索（可用时优先使用 embedding，相似度检索）。")
    else:
        print("[RAG] 未启用向量检索，使用基于词重叠的简化检索策略。")

    # 将知识库引用传入默认工具插件，便于通过工具进行检索
    default_tools.set_kb_ref(kb)

    tools = ToolRegistry()
    hooks = HookManager()

    # 动态加载 plugins 目录中的插件
    load_plugins(tools, hooks)

    llm = build_llm(config.llm)

    scenario = getattr(args, "scenario", "code_review")
    if scenario == "code_review":
        run_code_review_scenario(
            args=args,
            config=config,
            memory=memory,
            kb=kb,
            tools=tools,
            hooks=hooks,
            llm=llm,
        )
    elif scenario == "code_scratch":
        run_code_scratch_scenario(
            args=args,
            config=config,
            memory=memory,
            kb=kb,
            tools=tools,
            hooks=hooks,
            llm=llm,
        )
    elif scenario == "doc_research":
        run_doc_research_scenario(
            args=args,
            config=config,
            memory=memory,
            kb=kb,
            tools=tools,
            hooks=hooks,
            llm=llm,
        )
    else:
        print(f"[Agent] 未知场景: {scenario}，暂不执行。")


def run_code_review_scenario(
    args: argparse.Namespace,
    config: AgentConfig,
    memory: MemoryManager,
    kb: KnowledgeBase,
    tools: ToolRegistry,
    hooks: HookManager,
    llm: Any,
) -> None:
    """完整的 code_review 场景流水线：规划 → 执行 → 代码生成 → 静态分析 → 测试 → 报告。"""

    planner = Planner(llm=llm, tools=tools, kb=kb, memory=memory, hooks=hooks)
    executor = Executor(llm=llm, tools=tools, kb=kb, memory=memory, hooks=hooks)

    print("[Agent] 开始规划任务……")
    plan = planner.create_plan(goal=args.goal)

    print("[Agent] 规划完成，共 {} 步：".format(len(plan.steps)))
    for idx, step in enumerate(plan.steps, start=1):
        step_type = getattr(step, "step_type", "generic")
        role = getattr(step, "role", None) or "UnknownAgent"
        step_tools = getattr(step, "tools", None)
        tools_str = ", ".join(step_tools) if step_tools else "未指定"
        print(f"  - Step {idx} [{step_type} / {role} / tools: {tools_str}]: {step.description}")

    # Human-in-the-loop：让用户确认是否继续执行与代码生成
    choice = input("是否继续执行这些步骤并生成代码仓库？(Y/n) ").strip().lower()
    if choice in {"n", "no"}:
        print("[Agent] 用户选择终止执行，仅保留规划结果。")
        return

    print("[Agent] 开始执行任务……")
    executor.execute_plan(plan, goal=args.goal)

    # 基于研究与规划结果，按具体 Coding 题目生成对应的 demo 代码仓库
    task_slug = _slugify_task(args.coding_task) if args.coding_task else "multimodal_deepresearch_demo"
    output_dir = config.workspace_dir / "outputs" / task_slug
    print(f"[Agent] 开始基于本次研究结果生成 demo 代码仓库到: {output_dir}")
    generate_demo_repo(
        llm=llm,
        kb=kb,
        goal=args.goal,
        output_dir=output_dir,
        memory=memory,
        coding_task=args.coding_task,
    )

    # 在生成的 demo 仓库上运行静态代码质量检查，并尝试基于 issues 自动修复
    static_issues: List[Dict[str, Any]] = []
    static_autofix_changed: List[str] = []

    static_tool_names = config.get_static_tool_names()
    if static_tool_names:
        print(
            "[Agent] 将使用以下静态分析工具进行检查: "
            + ", ".join(static_tool_names)
        )

    static_tool_reports: List[Dict[str, Any]] = []

    if static_tool_names:
        # 并行执行静态分析工具，加快在大仓库上的检查速度
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(static_tool_names)) as executor:
            future_to_tool: Dict[concurrent.futures.Future, str] = {}
            for tool_name in static_tool_names:
                try:
                    tool = tools.get(tool_name)
                except KeyError:
                    continue
                future = executor.submit(_run_static_tool, tool, output_dir)
                future_to_tool[future] = tool_name

            for future in concurrent.futures.as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[Agent] {tool_name} 调用异常: {exc}")
                    static_tool_reports.append(
                        {
                            "name": tool_name,
                            "duration_seconds": None,
                            "issue_count": 0,
                            "error": str(exc),
                        }
                    )
                    continue

                issues = result.get("issues") or []
                error = result.get("error")
                duration = float(result.get("_duration", 0.0))
                if error:
                    print(f"[Agent] {tool_name} 调用提示: {error}")
                if issues:
                    print(f"[Agent] {tool_name} 发现 {len(issues)} 个问题。")
                    static_issues.extend(issues)

                static_tool_reports.append(
                    {
                        "name": tool_name,
                        "duration_seconds": duration,
                        "issue_count": len(issues),
                        "error": error,
                    }
                )

    if static_issues and config.enable_static_autofix:
        print(f"[Agent] 静态分析共发现 {len(static_issues)} 个问题，尝试基于 issues 进行自动修复……")
        autofix_result = attempt_autofix_from_issues(
            llm=llm,
            repo_root=output_dir,
            coding_task=args.coding_task,
            issues=static_issues,
            memory=memory,
        )
        static_autofix_changed = autofix_result.changed_files
        print(f"[Agent] 静态分析自动修复修改的文件: {autofix_result.changed_files}")

    # 运行 demo 仓库中的单元测试
    print("[Agent] 开始在 demo 仓库中运行自动测试……")
    try:
        run_python_tool = tools.get("run_python")
    except KeyError:
        print("[Agent] 未找到 run_python 工具，跳过自动测试。")
    else:
        test_snippet = """import unittest
import sys
from pathlib import Path

root = Path('.').resolve()
sys.path.insert(0, str(root))

suite = unittest.defaultTestLoader.discover('tests')
runner = unittest.TextTestRunner()
result = runner.run(suite)

if not result.wasSuccessful():
    raise SystemExit(1)
"""

        result = run_python_tool.func({"code": test_snippet, "workdir": str(output_dir)})
        returncode = result.get("returncode", 0)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if stdout:
            print(stdout)
        if stderr:
            print("[Agent][tests stderr]", stderr)

        tests_initial: Dict[str, Any] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

        tests_after_autofix: Dict[str, Any] | None = None
        tests_autofix_changed: List[str] = []

        if returncode != 0:
            print("[Agent] 自动测试失败，将触发自我反思与自动修复。")
            reflector = Reflector(llm)
            reflection = reflector.reflect(
                goal=args.goal,
                step="run demo repo tests",
                error=f"returncode={returncode}, stderr={stderr}",
                partial_output=stdout,
            )
            print("[Agent] 自我反思结果：")
            print(reflection)
            memory.add_event(
                "tests_failed",
                {
                    "output_dir": str(output_dir),
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "reflection": reflection,
                },
            )

            tests_after_autofix = None
            tests_autofix_changed = []

            if config.enable_test_autofix:
                # 自动修复闭环：尝试修改代码并重新运行测试
                print("[Agent] 尝试根据反思与测试输出来自动修复 demo 仓库代码……")
                fix_result = attempt_autofix(
                    llm=llm,
                    repo_root=output_dir,
                    coding_task=args.coding_task,
                    test_stdout=stdout,
                    test_stderr=stderr,
                    memory=memory,
                )
                print(f"[Agent] 自动修复修改的文件: {fix_result.changed_files}")

                print("[Agent] 自动修复后重新运行测试……")
                result2 = run_python_tool.func({"code": test_snippet, "workdir": str(output_dir)})
                rc2 = result2.get("returncode", 0)
                stdout2 = result2.get("stdout", "")
                stderr2 = result2.get("stderr", "")

                if stdout2:
                    print(stdout2)
                if stderr2:
                    print("[Agent][tests stderr after autofix]", stderr2)

                tests_after_autofix = {
                    "returncode": rc2,
                    "stdout": stdout2,
                    "stderr": stderr2,
                }
                tests_autofix_changed = fix_result.changed_files

                if rc2 != 0:
                    print("[Agent] 自动修复后测试仍然失败。")
                    memory.add_event(
                        "tests_failed_after_autofix",
                        {
                            "output_dir": str(output_dir),
                            "returncode": rc2,
                            "stdout": stdout2,
                            "stderr": stderr2,
                        },
                    )
                else:
                    print("[Agent] 自动修复后测试通过。")
                    memory.add_event(
                        "tests_passed_after_autofix",
                        {
                            "output_dir": str(output_dir),
                            "stdout": stdout2,
                        },
                    )

        else:
            print("[Agent] 自动测试通过。")
            memory.add_event(
                "tests_passed",
                {
                    "output_dir": str(output_dir),
                    "stdout": stdout,
                },
            )
            tests_after_autofix = None
            tests_autofix_changed = []

    # 生成本次运行的 JSON 报告，便于企业化集成与审计
    try:
        if not config.reports_enabled:
            print("[Agent] 已根据配置关闭 JSON 报告生成。")
        else:
            reports_dir = config.report_dir or (config.workspace_dir / "reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"run-{memory.session_id}.json"

            plan_steps_payload = []
            for idx, step in enumerate(plan.steps, start=1):
                plan_steps_payload.append(
                    {
                        "index": idx,
                        "description": step.description,
                        "step_type": getattr(step, "step_type", "generic"),
                        "tools": getattr(step, "tools", None),
                        "role": getattr(step, "role", None),
                        "duration_seconds": getattr(step, "duration_seconds", None),
                    }
                )

            report: Dict[str, Any] = {
                "goal": args.goal,
                "coding_task": args.coding_task,
                "session_id": memory.session_id,
                "output_dir": str(output_dir),
                "llm": {
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "base_url": config.llm.base_url,
                },
                "plan": {
                    "step_count": len(plan.steps),
                    "steps": plan_steps_payload,
                },
                "static_analysis": {
                    "issue_count": len(static_issues),
                    "issues": static_issues,
                    "autofix_changed_files": static_autofix_changed,
                    "tools": static_tool_reports,
                },
                "tests": {
                    "initial": tests_initial if "tests_initial" in locals() else None,
                    "after_autofix": tests_after_autofix,
                    "autofix_changed_files": tests_autofix_changed,
                },
                "autofix": {
                    "static_issues": {
                        "enabled": config.enable_static_autofix,
                        "attempted": config.enable_static_autofix and bool(static_issues),
                        "changed_files": static_autofix_changed,
                    },
                    "tests": {
                        "enabled": config.enable_test_autofix,
                        "attempted": (
                            config.enable_test_autofix
                            and "tests_initial" in locals()
                            and bool(tests_initial.get("returncode"))
                        ),
                        "changed_files": tests_autofix_changed,
                    },
                },
            }

            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_markdown_report(report, report_path.with_suffix(".md"))
            print(f"[Agent] 本次运行报告已写入: {report_path}")
    except Exception as exc:  # pragma: no cover - 报告生成失败不影响主流程
        print(f"[Agent] 生成 JSON 报告时出错: {exc}")

    print("[Agent] 任务执行完成，记忆已写入 memory/ 目录，demo 代码仓库已生成并完成测试/修复流程。")


def run_code_scratch_scenario(
    args: argparse.Namespace,
    config: AgentConfig,
    memory: MemoryManager,
    kb: KnowledgeBase,
    tools: ToolRegistry,
    hooks: HookManager,
    llm: Any,
) -> None:
    """纯代码生成场景：研究/设计 → 代码生成 → 测试 → 报告，不执行静态代码审查工具。"""

    planner = Planner(llm=llm, tools=tools, kb=kb, memory=memory, hooks=hooks)
    executor = Executor(llm=llm, tools=tools, kb=kb, memory=memory, hooks=hooks)

    print("[Agent] 开始规划任务（code_scratch 场景）……")
    plan = planner.create_plan(goal=args.goal)

    print("[Agent] 规划完成，共 {} 步：".format(len(plan.steps)))
    for idx, step in enumerate(plan.steps, start=1):
        step_type = getattr(step, "step_type", "generic")
        role = getattr(step, "role", None) or "UnknownAgent"
        step_tools = getattr(step, "tools", None)
        tools_str = ", ".join(step_tools) if step_tools else "未指定"
        print(f"  - Step {idx} [{step_type} / {role} / tools: {tools_str}]: {step.description}")

    choice = input("是否继续执行这些步骤并生成代码仓库？(Y/n) ").strip().lower()
    if choice in {"n", "no"}:
        print("[Agent] 用户选择终止执行，仅保留规划结果。")
        return

    print("[Agent] 开始执行任务……")
    executor.execute_plan(plan, goal=args.goal)

    # 基于研究与规划结果，按具体 Coding 题目生成对应的 demo 代码仓库
    task_slug = _slugify_task(args.coding_task) if args.coding_task else "multimodal_deepresearch_demo"
    output_dir = config.workspace_dir / "outputs" / task_slug
    print(f"[Agent] 开始基于本次研究结果生成 demo 代码仓库到: {output_dir}")
    generate_demo_repo(
        llm=llm,
        kb=kb,
        goal=args.goal,
        output_dir=output_dir,
        memory=memory,
        coding_task=args.coding_task,
    )

    # code_scratch 场景下，不额外执行静态代码审查工具，仅保留占位字段
    static_issues: List[Dict[str, Any]] = []
    static_autofix_changed: List[str] = []
    static_tool_reports: List[Dict[str, Any]] = []

    # 运行 demo 仓库中的单元测试
    print("[Agent] 开始在 demo 仓库中运行自动测试……")
    try:
        run_python_tool = tools.get("run_python")
    except KeyError:
        print("[Agent] 未找到 run_python 工具，跳过自动测试。")
        tests_initial: Dict[str, Any] | None = None
        tests_after_autofix: Dict[str, Any] | None = None
        tests_autofix_changed: List[str] = []
    else:
        test_snippet = """import unittest
import sys
from pathlib import Path

root = Path('.').resolve()
sys.path.insert(0, str(root))

suite = unittest.defaultTestLoader.discover('tests')
runner = unittest.TextTestRunner()
result = runner.run(suite)

if not result.wasSuccessful():
    raise SystemExit(1)
"""

        result = run_python_tool.func({"code": test_snippet, "workdir": str(output_dir)})
        returncode = result.get("returncode", 0)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if stdout:
            print(stdout)
        if stderr:
            print("[Agent][tests stderr]", stderr)

        tests_initial = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

        tests_after_autofix = None
        tests_autofix_changed = []

        if returncode != 0 and config.enable_test_autofix:
            print("[Agent] 自动测试失败，将触发自我反思与自动修复。")
            reflector = Reflector(llm)
            reflection = reflector.reflect(
                goal=args.goal,
                step="run demo repo tests",
                error=f"returncode={returncode}, stderr={stderr}",
                partial_output=stdout,
            )
            print("[Agent] 自我反思结果：")
            print(reflection)
            memory.add_event(
                "tests_failed",
                {
                    "output_dir": str(output_dir),
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "reflection": reflection,
                },
            )

            print("[Agent] 尝试根据反思与测试输出来自动修复 demo 仓库代码……")
            fix_result = attempt_autofix(
                llm=llm,
                repo_root=output_dir,
                coding_task=args.coding_task,
                test_stdout=stdout,
                test_stderr=stderr,
                memory=memory,
            )
            print(f"[Agent] 自动修复修改的文件: {fix_result.changed_files}")

            print("[Agent] 自动修复后重新运行测试……")
            result2 = run_python_tool.func({"code": test_snippet, "workdir": str(output_dir)})
            rc2 = result2.get("returncode", 0)
            stdout2 = result2.get("stdout", "")
            stderr2 = result2.get("stderr", "")

            if stdout2:
                print(stdout2)
            if stderr2:
                print("[Agent][tests stderr after autofix]", stderr2)

            tests_after_autofix = {
                "returncode": rc2,
                "stdout": stdout2,
                "stderr": stderr2,
            }
            tests_autofix_changed = fix_result.changed_files

            if rc2 != 0:
                print("[Agent] 自动修复后测试仍然失败。")
                memory.add_event(
                    "tests_failed_after_autofix",
                    {
                        "output_dir": str(output_dir),
                        "returncode": rc2,
                        "stdout": stdout2,
                        "stderr": stderr2,
                    },
                )
            else:
                print("[Agent] 自动修复后测试通过。")
                memory.add_event(
                    "tests_passed_after_autofix",
                    {
                        "output_dir": str(output_dir),
                        "stdout": stdout2,
                    },
                )
        elif returncode == 0:
            print("[Agent] 自动测试通过。")
            memory.add_event(
                "tests_passed",
                {
                    "output_dir": str(output_dir),
                    "stdout": stdout,
                },
            )

    # 生成本次运行的 JSON 报告，便于企业化集成与审计
    try:
        if not config.reports_enabled:
            print("[Agent] 已根据配置关闭 JSON 报告生成。")
        else:
            reports_dir = config.report_dir or (config.workspace_dir / "reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"run-{memory.session_id}.json"

            plan_steps_payload = []
            for idx, step in enumerate(plan.steps, start=1):
                plan_steps_payload.append(
                    {
                        "index": idx,
                        "description": step.description,
                        "step_type": getattr(step, "step_type", "generic"),
                        "tools": getattr(step, "tools", None),
                        "role": getattr(step, "role", None),
                        "duration_seconds": getattr(step, "duration_seconds", None),
                    }
                )

            report: Dict[str, Any] = {
                "goal": args.goal,
                "coding_task": args.coding_task,
                "session_id": memory.session_id,
                "output_dir": str(output_dir),
                "llm": {
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "base_url": config.llm.base_url,
                },
                "plan": {
                    "step_count": len(plan.steps),
                    "steps": plan_steps_payload,
                },
                "static_analysis": {
                    "issue_count": len(static_issues),
                    "issues": static_issues,
                    "autofix_changed_files": static_autofix_changed,
                    "tools": static_tool_reports,
                },
                "tests": {
                    "initial": tests_initial if "tests_initial" in locals() else None,
                    "after_autofix": tests_after_autofix if "tests_after_autofix" in locals() else None,
                    "autofix_changed_files": tests_autofix_changed if "tests_autofix_changed" in locals() else [],
                },
                "autofix": {
                    "static_issues": {
                        "enabled": config.enable_static_autofix,
                        "attempted": False,
                        "changed_files": static_autofix_changed,
                    },
                    "tests": {
                        "enabled": config.enable_test_autofix,
                        "attempted": (
                            config.enable_test_autofix
                            and "tests_initial" in locals()
                            and bool(tests_initial and tests_initial.get("returncode"))
                        ),
                        "changed_files": tests_autofix_changed if "tests_autofix_changed" in locals() else [],
                    },
                },
            }

            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_markdown_report(report, report_path.with_suffix(".md"))
            print(f"[Agent] 本次运行报告已写入: {report_path}")
    except Exception as exc:  # pragma: no cover - 报告生成失败不影响主流程
        print(f"[Agent] 生成 JSON 报告时出错: {exc}")

    print("[Agent] 任务执行完成，记忆已写入 memory/ 目录，demo 代码仓库已生成并完成测试流程。")


def run_doc_research_scenario(
    args: argparse.Namespace,
    config: AgentConfig,
    memory: MemoryManager,
    kb: KnowledgeBase,
    tools: ToolRegistry,
    hooks: HookManager,
    llm: Any,
) -> None:
    """文档 DeepResearch 场景：只做研究/设计/文档撰写，不生成代码仓库。"""

    planner = Planner(llm=llm, tools=tools, kb=kb, memory=memory, hooks=hooks)
    executor = Executor(llm=llm, tools=tools, kb=kb, memory=memory, hooks=hooks)

    print("[Agent] 开始规划任务（doc_research 场景）……")
    plan = planner.create_plan(goal=args.goal, scenario="doc_research")

    # 仅保留 research/design/doc 类型的步骤
    filtered_steps = [
        step
        for step in plan.steps
        if getattr(step, "step_type", "generic") in ("research", "design", "doc")
    ]
    if filtered_steps:
        plan.steps = filtered_steps

    print("[Agent] 规划完成，共 {} 步：".format(len(plan.steps)))
    for idx, step in enumerate(plan.steps, start=1):
        step_type = getattr(step, "step_type", "generic")
        role = getattr(step, "role", None) or "UnknownAgent"
        step_tools = getattr(step, "tools", None)
        tools_str = ", ".join(step_tools) if step_tools else "未指定"
        print(f"  - Step {idx} [{step_type} / {role} / tools: {tools_str}]: {step.description}")

    choice = input("是否继续执行这些研究/设计/文档步骤？(Y/n) ").strip().lower()
    if choice in {"n", "no"}:
        print("[Agent] 用户选择终止执行，仅保留规划结果。")
        return

    print("[Agent] 开始执行任务……")
    executor.execute_plan(plan, goal=args.goal)

    # doc_research 场景下不生成代码仓库，也不运行静态分析和测试
    static_issues: List[Dict[str, Any]] = []
    static_autofix_changed: List[str] = []
    static_tool_reports: List[Dict[str, Any]] = []

    tests_initial: Dict[str, Any] | None = None
    tests_after_autofix: Dict[str, Any] | None = None
    tests_autofix_changed: List[str] = []

    # 生成本次运行的 JSON 报告，便于企业化集成与审计
    try:
        if not config.reports_enabled:
            print("[Agent] 已根据配置关闭 JSON 报告生成。")
        else:
            reports_dir = config.report_dir or (config.workspace_dir / "reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"run-{memory.session_id}.json"

            plan_steps_payload = []
            for idx, step in enumerate(plan.steps, start=1):
                plan_steps_payload.append(
                    {
                        "index": idx,
                        "description": step.description,
                        "step_type": getattr(step, "step_type", "generic"),
                        "tools": getattr(step, "tools", None),
                        "role": getattr(step, "role", None),
                        "duration_seconds": getattr(step, "duration_seconds", None),
                    }
                )

            report: Dict[str, Any] = {
                "goal": args.goal,
                "coding_task": args.coding_task,
                "session_id": memory.session_id,
                # doc_research 场景下没有特定的输出代码目录，记录为工作区目录
                "output_dir": str(config.workspace_dir),
                "llm": {
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "base_url": config.llm.base_url,
                },
                "plan": {
                    "step_count": len(plan.steps),
                    "steps": plan_steps_payload,
                },
                "static_analysis": {
                    "issue_count": len(static_issues),
                    "issues": static_issues,
                    "autofix_changed_files": static_autofix_changed,
                    "tools": static_tool_reports,
                },
                "tests": {
                    "initial": tests_initial,
                    "after_autofix": tests_after_autofix,
                    "autofix_changed_files": tests_autofix_changed,
                },
                "autofix": {
                    "static_issues": {
                        "enabled": config.enable_static_autofix,
                        "attempted": False,
                        "changed_files": static_autofix_changed,
                    },
                    "tests": {
                        "enabled": config.enable_test_autofix,
                        "attempted": False,
                        "changed_files": tests_autofix_changed,
                    },
                },
            }

            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_markdown_report(report, report_path.with_suffix(".md"))
            print(f"[Agent] 本次运行报告已写入: {report_path}")
    except Exception as exc:  # pragma: no cover - 报告生成失败不影响主流程
        print(f"[Agent] 生成 JSON 报告时出错: {exc}")

    print("[Agent] doc_research 场景执行完成，记忆已写入 memory/ 目录。")


if __name__ == "__main__":
    main()
