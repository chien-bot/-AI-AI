from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .llm import BaseLLM, ChatMessage
from .rag.knowledge_base import KnowledgeBase
from .memory import MemoryManager


@dataclass
class GeneratedFile:
    """Represents a single file that should be written to disk."""

    path: str
    content: str


def _extract_json_candidate(raw: str) -> str:
    """Try to extract a JSON object string from raw LLM output.

    Many chat models会错误地把 JSON 包在 ```json ... ``` 代码块里，
    这里做一次简单的提取，尽量拿到纯 JSON 再去解析。
    """

    text = raw.strip()

    # Fast path: already looks like plain JSON object.
    if text.startswith("{") and text.endswith("}"):
        return text

    # Try to strip markdown code fences: ```json ... ``` 或 ``` ... ```
    if "```" in text:
        first = text.find("```")
        second = text.rfind("```")
        if second != -1 and second > first:
            inner = text[first + 3 : second].strip()
            # Possible leading language tag, e.g. "json\n{...}"
            if inner.lower().startswith("json"):
                # Drop leading "json" 和紧跟的换行
                inner = inner[4:].lstrip("\r\n ")
            inner = inner.strip()
            if inner.startswith("{") and inner.endswith("}"):
                return inner

    return text


def _plan_repo_with_llm(
    llm: BaseLLM,
    goal: str,
    kb: KnowledgeBase,
    coding_task: str | None = None,
) -> List[GeneratedFile]:
    """Ask the LLM to propose a demo repo structure and file contents.

    The model is asked to return a JSON object so that we can parse it
    with ``json.loads`` in a robust way.
    """

    kb_context = kb.as_context(query=goal, top_k=10)

    system = ChatMessage(
        role="system",
        content=(
            "你是一名资深软件架构师和代码生成专家，现在需要为一个复杂代码生成任务"
            "构建一个最小可运行的 demo 代码仓库。"
        ),
    )

    ct_part = f"\n\n本次具体 Coding 题目是：{coding_task}\n" if coding_task else "\n"

    user_parts = [
        "任务目标如下:\n",
        goal,
        ct_part,
        "\n请根据上述目标以及你对多模态 Deep Research / 复杂代码生成的理解，",
        "设计一个与该 Coding 题目强相关的 Python 项目骨架。\n",
        "要求：\n",
        "1. 项目使用纯 Python，实现基础的模块化结构，但不需要真正训练大模型。\n",
        "2. 至少包含以下类型的文件：README.md、一个或多个模块文件 (如 agent.py / pipeline.py)、",
        "   以及一个可以直接运行的示例脚本 (如 run_demo.py)。\n",
        "3. README 中要简要描述该 Coding 题目、项目结构以及运行方式。\n",
        "4. 请只输出一个 JSON 字符串，不要添加任何解释性文字，不要使用代码块。\n",
        "   要求：\n",
        "   - 必须是合法 JSON：对象键和值都使用双引号。\n",
        "   - 文件内容中的换行请使用 \\n 进行转义，不要直接写多行字符串。\n",
        "   - 不要在 JSON 外再包一层代码块或文字说明。\n",
        "   JSON 格式示例：\n",
        "   {\n",
        '     \"project_name\": \"multimodal_deepresearch_demo\",\n',
        "     \"files\": [\n",
        '       {\"path\": \"README.md\", \"content\": \"# Title...\\n\"},\n',
        '       {\"path\": \"multimodal_deepresearch/__init__.py\", \"content\": \"...\"},\n',
        '       {\"path\": \"multimodal_deepresearch/agent.py\", \"content\": \"...\"},\n',
        '       {\"path\": \"examples/run_demo.py\", \"content\": \"...\"}\n',
        "     ]\n",
        "   }\n",
        "5. 请在 files 中显式体现 Deep Research 的 4 个核心组件：查询规划、信息获取、内存管理、答案生成。\n",
        "   例如，可以提供如下模块（可按任务需要调整命名，但职责要清晰对应）：\n",
        '   - {\"path\": \"deepresearch_core/query_planner.py\", \"content\": \"class QueryPlanner: ...\"}\n',
        '   - {\"path\": \"deepresearch_core/information_acquirer.py\", \"content\": \"class InformationAcquirer: ...\"}\n',
        '   - {\"path\": \"deepresearch_core/memory_manager.py\", \"content\": \"class MemoryManager: ...\"}\n',
        '   - {\"path\": \"deepresearch_core/answer_generator.py\", \"content\": \"class AnswerGenerator: ...\"}\n',
        "   如果当前 Coding 题目与代码审查 / 修复相关，请额外提供一组 code_review_* 模块，用于集成 Pylint/Flake8/Bandit 等工具的调用接口。\n",
    ]

    if kb_context:
        user_parts.append("\n如果有帮助，可以参考以下文档内容片段（无需在输出中重复粘贴）：\n")
        user_parts.append(kb_context)

    user = ChatMessage(role="user", content="".join(user_parts))

    raw = llm.chat([system, user])
    candidate = _extract_json_candidate(raw)

    files: List[GeneratedFile] = []
    parse_error: Exception | None = None

    data = None
    # Prefer JSON parsing (more robust for arbitrary file contents).
    try:
        data = json.loads(candidate)
    except Exception as exc_json:
        parse_error = exc_json
        # Fallback to Python 字典字面量，兼容旧 prompt。
        try:
            data = ast.literal_eval(candidate)
            parse_error = None
        except Exception as exc_ast:
            parse_error = exc_ast

    if isinstance(data, dict):
        file_entries = data.get("files")
        if isinstance(file_entries, list):
            for entry in file_entries:
                if not isinstance(entry, dict):
                    continue
                path = str(entry.get("path", "")).strip()
                content = str(entry.get("content", ""))
                if path:
                    files.append(GeneratedFile(path=path, content=content))

    if parse_error is not None and not files:
        # If parsing fails, keep the raw output so that the run is still reproducible.
        files.append(
            GeneratedFile(
                path="README.generated_raw_output.md",
                content=(
                    "LLM 输出未能按预期解析为项目文件列表（JSON / Python 字典），原始内容如下：\n\n"
                    f"{raw}\n\n解析错误: {parse_error}"
                ),
            )
        )

    if not files:
        # As a last resort, just store the raw text.
        files.append(GeneratedFile(path="README.generated_raw_output.md", content=raw))

    return files


def _ensure_core_demo_files(output_dir: Path, coding_task: str | None = None) -> None:
    """Ensure a small but runnable demo repo exists under ``output_dir``."""

    if coding_task:
        title = "Coding Task Demo"
        intro = (
            "This repo contains a minimal scaffold for the following coding task:\n\n"
            f"{coding_task}\n\n"
            "It uses a simple multi-agent, multimodal-style code generation pipeline, "
            "so you can focus on the task logic instead of boilerplate.\n\n"
        )
    else:
        title = "Multimodal Code Generation Demo"
        intro = (
            "A minimal demo project showing a simple multi-modal style code generation pipeline.\n\n"
        )

    # README
    readme_content = (
        f"# {title}\n"
        f"{intro}"
        "## Usage\n"
        "1. Ensure you have Python 3.10+ installed.\n"
        "2. From the project root, run:\n\n"
        "   python -m examples.run_demo\n\n"
        "## Project Structure\n"
        "- `multimodal_code_gen/`:\n"
        "  - `agent.py`         - High level agent that uses a data processor and model wrapper to produce code.\n"
        "  - `pipeline.py`      - Orchestrates multiple agents and merges their outputs.\n"
        "  - `data_processor.py`- Parse natural language requests into structured instructions.\n"
        "  - `model_wrapper.py` - Dummy model interface that turns instructions into Python code.\n"
        "  - `utils.py`         - Small helper functions.\n"
        "- `examples/`:\n"
        "  - `run_demo.py`      - End-to-end example entry.\n"
        "- `tests/`:\n"
        "  - `test_pipeline.py` - Simple unit test that checks the demo pipeline behaviour.\n"
    )

    readme = output_dir / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(readme_content, encoding="utf-8")

    # Package directory
    pkg_dir = output_dir / "multimodal_code_gen"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # __init__.py
    init_code = (
        "from .agent import CodeGenAgent\n"
        "from .pipeline import CodeGenPipeline\n\n"
        "__all__ = [\"CodeGenAgent\", \"CodeGenPipeline\"]\n"
    )
    (pkg_dir / "__init__.py").write_text(init_code, encoding="utf-8")

    # utils.py
    utils_code = (
        "import re\n\n\n"
        "def to_snake_case(text: str) -> str:\n"
        "    \"\"\"Convert free-form text into a safe snake_case identifier.\"\"\"\n"
        "    text = text.strip().lower()\n"
        "    text = re.sub(r\"[^a-z0-9]+\", \"_\", text)\n"
        "    text = re.sub(r\"_+\", \"_\", text).strip(\"_\")\n"
        "    return text or \"generated_function\"\n"
    )
    (pkg_dir / "utils.py").write_text(utils_code, encoding="utf-8")

    # data_processor.py
    data_processor_code = (
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class ParsedRequest:\n"
        "    # Structured representation of a code generation request.\n"
        "    kind: str\n"
        "    description: str\n\n\n"
        "class DataProcessor:\n"
        "    \"\"\"Very small text parser used in the demo pipeline.\"\"\"\n\n"
        "    def parse(self, prompt: str) -> ParsedRequest:\n"
        "        text = prompt.strip()\n"
        "        lower = text.lower()\n"
        "        desc = text\n\n"
        "        if \"function\" in lower and \"that\" in lower:\n"
        "            # Try to capture the description after the word 'that'.\n"
        "            try:\n"
        "                desc_part = text.split(\"that\", 1)[1].strip()\n"
        "                if desc_part:\n"
        "                    desc = desc_part\n"
        "            except Exception:\n"
        "                pass\n\n"
        "        return ParsedRequest(kind=\"python_function\", description=desc)\n"
    )
    (pkg_dir / "data_processor.py").write_text(data_processor_code, encoding="utf-8")

    # model_wrapper.py
    # NOTE: keep this implementation simple and robust – it is only a demo,
    # but tests expect the generated code string to contain the natural
    # language description (e.g. "adds two numbers"). We include it in a
    # comment line for simplicity.
    model_wrapper_source = '''from __future__ import annotations

from .utils import to_snake_case


class DummyModelWrapper:
    """Very small stand-in for a real model.
    It turns a natural language description into a Python function
    skeleton so that the demo pipeline has something concrete to run.
    """

    def __init__(self, language: str = "python") -> None:
        self.language = language

    def generate_function_code(self, description: str) -> str:
        name = to_snake_case(description[:40] or "generated_function")
        code_lines = [
            f"def {name}(items):",
            f"    # {description}",
            "    # Auto generated demo function, please replace with real logic.",
            "    # TODO: implement the real logic here",
            "    return items",
        ]
        return "\\n".join(code_lines) + "\\n"
'''

    (pkg_dir / "model_wrapper.py").write_text(model_wrapper_source, encoding="utf-8")

    # agent.py
    agent_code = (
        "from __future__ import annotations\n\n"
        "from .data_processor import DataProcessor\n"
        "from .model_wrapper import DummyModelWrapper\n\n\n"
        "class CodeGenAgent:\n"
        "    \"\"\"Simple code generation agent used in the demo pipeline.\"\"\"\n\n"
        "    def __init__(self, model_name: str) -> None:\n"
        "        self.model_name = model_name\n"
        "        self._processor = DataProcessor()\n"
        "        self._model = DummyModelWrapper()\n\n"
        "    def process_input(self, input_data: str) -> str:\n"
        "        parsed = self._processor.parse(input_data)\n"
        "        if parsed.kind == \"python_function\":\n"
        "            return self._model.generate_function_code(parsed.description)\n"
        "        return f\"# TODO: handle request: {parsed.description}\\n\"\n"
    )
    (pkg_dir / "agent.py").write_text(agent_code, encoding="utf-8")

    # pipeline.py
    pipeline_code = (
        "from __future__ import annotations\n\n"
        "from typing import List, Any\n\n\n"
        "class CodeGenPipeline:\n"
        "    \"\"\"Minimal multi-agent code generation pipeline.\"\"\"\n\n"
        "    def __init__(self) -> None:\n"
        "        self.agents: List[Any] = []\n\n"
        "    def add_agent(self, agent: Any) -> None:\n"
        "        self.agents.append(agent)\n\n"
        "    def run_pipeline(self, input_data: str) -> str:\n"
        "        \"\"\"Call all agents in sequence and join their outputs.\"\"\"\n\n"
        "        results = []\n"
        "        for agent in self.agents:\n"
        "            process = getattr(agent, \"process_input\", None)\n"
        "            if callable(process):\n"
        "                results.append(process(input_data))\n\n"
        "        return \"\\n\\n\".join(str(r) for r in results)\n"
    )
    (pkg_dir / "pipeline.py").write_text(pipeline_code, encoding="utf-8")

    # examples/run_demo.py
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    run_demo_code = (
        "from multimodal_code_gen import CodeGenPipeline, CodeGenAgent\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    # Create demo pipeline\n"
        "    pipeline = CodeGenPipeline()\n\n"
        "    # Add agents (simulating multi-modal processing)\n"
        "    text_agent = CodeGenAgent(\"text_model\")\n"
        "    pipeline.add_agent(text_agent)\n\n"
        "    image_agent = CodeGenAgent(\"image_model\")\n"
        "    pipeline.add_agent(image_agent)\n\n"
        "    # Input data\n"
        "    input_data = \"Generate a function that sorts a list of numbers\"\n\n"
        "    # Run pipeline\n"
        "    code_output = pipeline.run_pipeline(input_data)\n\n"
        "    print(\"Generated Code:\\n\")\n"
        "    print(code_output)\n"
    )
    (examples_dir / "run_demo.py").write_text(run_demo_code, encoding="utf-8")

    # tests/test_pipeline.py
    tests_dir = output_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    tests_code = (
        "import unittest\n\n"
        "from multimodal_code_gen import CodeGenAgent, CodeGenPipeline\n\n\n"
        "class TestCodeGenPipeline(unittest.TestCase):\n"
        "    def test_pipeline_generates_function_code(self) -> None:\n"
        "        pipeline = CodeGenPipeline()\n"
        "        pipeline.add_agent(CodeGenAgent(\"text_model\"))\n\n"
        "        prompt = \"Generate a function that adds two numbers\"\n"
        "        code = pipeline.run_pipeline(prompt)\n\n"
        "        self.assertIn(\"def \", code)\n"
        "        self.assertIn(\"adds two numbers\", code)\n\n\n"
        "if __name__ == \"__main__\":  # pragma: no cover - manual run only\n"
        "    unittest.main()\n"
    )
    (tests_dir / "test_pipeline.py").write_text(tests_code, encoding="utf-8")


def generate_demo_repo(
    llm: BaseLLM,
    kb: KnowledgeBase,
    goal: str,
    output_dir: Path,
    memory: MemoryManager | None = None,
    coding_task: str | None = None,
) -> None:
    """Generate a demo code repo for the current DeepCodeResearch session."""

    output_dir.mkdir(parents=True, exist_ok=True)

    files = _plan_repo_with_llm(llm=llm, goal=goal, kb=kb, coding_task=coding_task)

    for gf in files:
        file_path = output_dir / gf.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(gf.content, encoding="utf-8")

    # Overlay a stable, runnable reference implementation.
    _ensure_core_demo_files(output_dir, coding_task=coding_task)

    if memory is not None:
        memory.add_event(
            "repo_generated",
            {
                "output_dir": str(output_dir),
                "file_count": len(files),
                "files": [gf.path for gf in files],
                "coding_task": coding_task,
            },
        )
