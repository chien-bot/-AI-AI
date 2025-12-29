# DeepCodeResearch：面向复杂代码生成的多场景 Agent 系统

> 基于 MS-Agent 架构思想实现的参考系统，用于赛题 3「复杂代码生成 DeepCodeResearch」。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![OpenAtom Challenge](https://img.shields.io/badge/OpenAtom-Agent%20Challenge-orange.svg)](https://openatom.tech/agentchallenge)

本项目实现了一个可扩展的 Agent 运行时，支持：

- 多技术文档输入 + 轻量 RAG 检索（PDF / PPTX / DOCX / TXT / MD）
- 深度研究 → 需求归纳 → 架构设计 → 代码生成 → 测试 / 自我修复 → 文档撰写 的完整多步流程
- 多场景运行：`code_review` / `code_scratch` / `doc_research`
- 集成真实的 Python 代码质量工具：`flake8` / `pylint` / `bandit` / `mypy`
- 单元测试 + LLM 自我反思 + 自动修复闭环
- 企业化友好的 JSON / Markdown 运行报告与长期记忆

同一套运行时可以在不同 `goal` / `coding_task` 之下生成多个不同的 demo 仓库（REST API、日志分析工具、股票分析脚本等），证明系统具备通用的复杂代码生成能力，而不仅限于“代码审查 Agent”这一单一任务。

---

## 1. 目录结构

项目根目录为：`复杂代码生成DeepCodeResearch/`

```text
复杂代码生成DeepCodeResearch/
├─ main.py                     # 命令行入口；场景路由 & 顶层流水线
├─ code_review_agent.yml       # 项目级配置（静态工具开关、autofix、报告目录、LLM 配置等）
├─ requirements.txt            # 运行所需依赖
├─ deepcoderesearch/           # 核心 Agent 运行时
│  ├─ config.py                # AgentConfig / LLMConfig & 从 YAML 加载项目配置
│  ├─ llm.py                   # LLM 抽象：DummyLLM / OpenAILLM / build_llm
│  ├─ planner.py               # Planner：基于 LLM 生成 JSON 结构化计划
│  ├─ executor.py              # Executor：按步骤调用 RAG + 工具 + LLM，记录耗时与输出
│  ├─ memory.py                # MemoryManager：session-*.jsonl 长期记忆
│  ├─ tools.py                 # Tool / ToolRegistry 定义，兼容 OpenAI function calling
│  ├─ hooks.py                 # HookManager：pre_* / post_* / on_error 生命周期 Hook
│  ├─ autofix.py               # 自动修复逻辑（基于测试失败 / 静态分析 issues）
│  ├─ codegen.py               # 代码仓库生成：LLM 规划 + 保底 demo 实现
│  ├─ reflection.py            # Reflector：执行失败后的 LLM 自我反思
│  ├─ mcp_client.py            # MCP/HTTP 工具客户端 & OpenAPI schema 加载
│  └─ rag/
│     ├─ document_loaders.py   # PDF / PPTX / DOCX / TXT / MD 加载逻辑
│     └─ knowledge_base.py     # 轻量 RAG 知识库 + 可选向量检索
├─ plugins/                    # 插件化工具体系
│  ├─ __init__.py              # 自动发现并注册插件
│  ├─ default_tools.py         # echo / kb_search 工具 + 步骤级 Hook
│  ├─ code_runner.py           # run_python：执行临时代码片段（用于 unittest）
│  ├─ code_quality.py          # run_flake8 / run_pylint / run_bandit / run_mypy
│  └─ web_search.py            # web_search：通过 MCPClient 调用外部 Web 搜索
├─ references/                 # 赛题提供的 DeepResearch 相关文档（可选）
├─ outputs/                    # 每个任务生成的 demo 代码仓库（多任务证据）
└─ reports/                    # 运行报告：run-*.json + run-*.md
   └─ ...                      # 每次运行一份 JSON + 一份 Markdown
```

运行过程中还会自动创建：

- `memory/`：`session-*.jsonl`，记录 Planner / Executor / AutoFix 等行为轨迹
- `knowledge_base/`：如通过配置指定，将 RAG 索引持久化

---

## 2. 核心架构设计

本仓库遵循 MS-Agent 推荐的分层结构：Planner / Executor / Memory / Tools / RAG / Hooks，可视为官方 MS-Agent 的一个“精简定制版 runtime”。

### 2.1 LLM 层（`deepcoderesearch/llm.py`）

- `LLMConfig` 从环境变量或 `code_review_agent.yml` 读取 provider / model / api_key / base_url。
- 支持任意 OpenAI 协议兼容服务（如 OpenAI、ModelScope DeepSeek 等）。
- `build_llm()`：
  - 若配置了合法的 API Key，则构造 `OpenAILLM`，通过 `chat.completions` 调用远程模型。
  - 否则退化为 `DummyLLM`，只做占位输出，保证流程始终可跑。

### 2.2 知识库 & RAG（`rag/knowledge_base.py`）

- 支持文档格式：`.pdf` / `.pptx` / `.docx` / `.txt` / `.md`（见 `document_loaders.py`）。
- `KnowledgeBase.ingest_documents()` 将文档切分为 `DocumentChunk`，默认按词窗口聚合。
- 检索策略：
  - 如 `_build_openai_embedder()` 成功构建 embedding 客户端，使用向量余弦相似度检索。
  - 否则回退到基于词重叠的简化检索。
- `as_context(query)` 将 top-k 片段拼接为一段可直接喂给 LLM 的上下文字符串。
- 同时通过工具层暴露为 `kb_search`（见 `plugins/default_tools.py`），便于在 Executor 中做二次检索。

### 2.3 Planner（`deepcoderesearch/planner.py`）

- 输入：`goal` + 当前场景 `scenario` (`code_review` / `code_scratch` / `doc_research`)。
- 输出：`Plan` 对象，内部由多个 `PlanStep` 组成，字段包括：
  - `description`：自然语言描述
  - `step_type`：`research` / `design` / `code` / `test` / `fix` / `doc`
  - `tools`：建议使用的工具，如 `["kb_search", "web_search", "run_python"]`
  - `role`：子 Agent 角色，如 `ResearchAgent`、`CodeAgent` 等
- Prompt 层强制模型以 JSON 返回：
  - 支持顶层数组 `[{...}, ...]`，或 `{ "steps": [{...}] }` 两种格式。
  - `_extract_json_candidate()` 负责从 ```json ... ``` 代码块中剥离 JSON。
  - `_parse_plan_json()` 解析失败时回退到 `_parse_plan_text()`，保证鲁棒。

### 2.4 Executor（`deepcoderesearch/executor.py`）

- 对每个 `PlanStep` 做以下工作：
  1. 从 `KnowledgeBase` 查询相关文档片段；
  2. 根据 `step.tools` 调用 `kb_search` / `web_search` 等工具补充上下文；
  3. 构造包含 goal、步骤信息、工具上下文的 Prompt，调用 LLM 完成该步骤；
  4. 将输出写入 `MemoryManager`，记录：
     - `step_executed` / `step_error`
     - `step_type` / `role` / `tools` / `duration_seconds`
- 在出现异常时，调用 `Reflector.reflect(...)`，生成“可能原因 + 修复方向”的自我反思文本，并写入记忆。
- 同时通过 `HookManager` 触发 `pre_step` / `post_step` / `on_error`，默认插件会打印关键日志。

### 2.5 Tools & 插件体系（`plugins/` + `deepcoderesearch/tools.py`）

- `ToolRegistry` 提供统一的工具注册与查找接口，并支持导出为 OpenAI function 调用格式。
- `plugins/__init__.py` 动态加载 `plugins/` 目录中的所有 `*.py` 文件，并调用它们的 `register(tools, hooks)`。
- 内置插件：
  - `default_tools.py`：
    - `echo`：调试用 echo 工具；
    - `kb_search`：基于 `KnowledgeBase` 的文档检索；
    - 注册 `pre_step` / `post_step` Hook 打印执行日志。
  - `code_runner.py`：
    - `run_python`：在临时文件 `_agent_snippet.py` 中执行一段 Python 代码，用于跑 `unittest` 或演示 Code Agent 能力。
  - `code_quality.py`：
    - `run_flake8` / `run_pylint` / `run_bandit` / `run_mypy`；
    - 对外暴露统一的 `issues` schema（`file/line/column/tool/code/message`），供自动修复使用。
  - `web_search.py`：
    - `web_search`：通过 `MCPClient` 调用外部 HTTP 搜索服务；
    - 支持通过环境变量 `WEB_SEARCH_ENDPOINT` / `WEB_SEARCH_AUTH_HEADER` 配置。

### 2.6 Memory & Hooks（`deepcoderesearch/memory.py`, `deepcoderesearch/hooks.py`）

- `MemoryManager`：
  - 所有关键事件以 `MemoryEvent` 写入 `memory/session-*.jsonl`；
  - 便于赛题评审和日志回放。
- `HookManager`：
  - 支持 `pre_planning` / `post_planning` / `pre_step` / `post_step` / `pre_tool` / `post_tool` / `on_error` 等事件；
  - 插件可在不改动核心代码的前提下插入观测逻辑，契合 MS-Agent 的 “Hooks + Extensibility” 设计。

### 2.7 自动修复 & 自我反思（`deepcoderesearch/autofix.py`, `deepcoderesearch/reflection.py`）

- 基于单元测试失败的自动修复：`attempt_autofix(...)`：
  - 将当前 repo 关键 Python 文件截断后送入 LLM；
  - 提供 `unittest` 的 stdout/stderr；
  - 要求 LLM 返回一个 `{'changes': [{'path': ..., 'content': ...}, ...]}` 的 Python 字典字面量；
  - 使用 `ast.literal_eval` 解析并写回文件。
- 基于静态分析 issues 的自动修复：`attempt_autofix_from_issues(...)`：
  - 将统一 schema 的 issues 列表（含工具、文件、行号）提供给 LLM；
  - 走同样的 `changes` 机制进行修复。
- 所有修复行为都会写入 `MemoryManager`，并在运行报告中记录修改的文件列表。

---

## 3. 运行场景（scenario）

入口脚本 `main.py` 支持三种运行场景，通过 `--scenario` 选择：

### 3.1 code_review：代码审查与修复场景（默认）

适用于：设计并实现“Python 代码审查与自动修复 Agent 系统”这类任务。

流程概览：

1. **规划**  
   `Planner.create_plan(goal, scenario="code_review")` 生成包含 research/design/code/test/fix/doc 的多步计划。

2. **Human-in-the-loop 确认**  
   在终端打印完整计划，并询问：  
   > 是否继续执行这些步骤并生成代码仓库？(Y/n)

3. **执行**  
   `Executor.execute_plan(...)` 按步骤调用 RAG + 工具 + LLM，写入记忆，并统计每步耗时。

4. **代码仓库生成**  
   - 根据 `--coding-task` 生成任务 slug（如 `rest_api_fastapi_sqlite`）；
   - 调用 `generate_demo_repo(...)`：
     - 先让 LLM 返回一个包含 `project_name` + `files[]` 的 JSON；
     - 若 JSON 不合法，会安全降级：将原始输出写入 `README.generated_raw_output.md`；
     - 然后 `_ensure_core_demo_files(...)` 覆盖/补足一套始终可运行的 `multimodal_code_gen` demo（含 `tests/test_pipeline.py`）。

5. **静态分析 + 自动修复（可配置）**  
   - 依据 `AgentConfig.static_tools` 决定启用哪些工具；
   - 使用 `ThreadPoolExecutor` 并行运行 `run_flake8/run_pylint/run_bandit/run_mypy`；
   - 汇总为统一 issues 列表 + 每个工具的耗时与问题数量；
   - 若 `enable_static_autofix=True` 且存在问题，则调用 `attempt_autofix_from_issues(...)` 自动修复。

6. **测试 + 自我修复闭环**  
   - 通过 `run_python` 工具在 demo repo 中执行内联的 unittest 脚本；
   - 若失败且 `enable_test_autofix=True`：
     - 调用 `Reflector` 分析失败原因；
     - 调用 `attempt_autofix(...)` 根据测试输出修复代码；
     - 再次运行测试，并记录第二次结果。

7. **报告生成**  
   - JSON 报告：`reports/run-<session_id>.json`
   - Markdown 报告：`reports/run-<session_id>.md`
   - 报告中包含：
     - 每个步骤的描述 / 类型 / 角色 / 工具 / `duration_seconds`
     - 每个静态工具的耗时 / issue_count / error
     - 测试前后返回码与关键信息
     - 自动修复是否启用 / 是否尝试 / 修改了哪些文件
     - 当前所用 LLM 的 provider / model / base_url

### 3.2 code_scratch：通用复杂代码生成场景

适用于：从零实现某个应用或脚本，比如：

- “实现一个简单的图书管理 REST API（增删改查、分页、搜索）”
- “实现一个日志分析工具，从 log 文件中统计错误类型和频率”
- “实现一个股票数据分析脚本：拉取数据、计算指标、输出图表”

差异点：

- Planner 会以 `scenario="code_scratch"` 生成更偏向“项目实现”的步骤；
- 流程仍包含 research / design / code / test / doc；
- 生成 demo 代码仓库 + 单元测试，并运行测试；
- 默认不强制运行静态分析工具，仅保留 `static_analysis` 字段作为占位，展示可扩展性；
- 同样支持基于测试失败的自动修复闭环。

### 3.3 doc_research：纯文档研究场景

适用于：只做 Deep Research 和报告，不生成代码。

特性：

- 使用 `scenario="doc_research"` 生成偏研究/分析/文档的步骤；
- 在 Planner 结果上只保留 `research` / `design` / `doc` 类型的步骤；
- Human-in-the-loop 确认后，执行这些步骤：
  - 深度阅读参考文档和知识库；
  - 归纳需求、对比方案、输出架构与设计；
  - 形成结构化的文字报告。
- 不创建 demo 代码仓库，不运行静态分析与测试：
  - `output_dir` 字段记录为工作区目录；
  - `static_analysis.tools` / `tests` 字段为空占位；
- 仍然生成 JSON + Markdown 报告，方便在内部平台展示研究结论。

---

## 4. 报告与观测性设计

每次运行都会生成一份结构化 JSON 报告，以及对应的 Markdown 人类可读报告。

### 4.1 JSON 报告结构（节选）

```jsonc
{
  "goal": "...",
  "coding_task": "...",
  "session_id": "20251208-172813",
  "output_dir": "C:\\...\\outputs\\rest_api_fastapi_sqlite",
  "llm": {
    "provider": "modelscope",
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "base_url": "https://api-inference.modelscope.cn/v1"
  },
  "plan": {
    "step_count": 9,
    "steps": [
      {
        "index": 1,
        "description": "文档研究：...",
        "step_type": "research",
        "tools": ["kb_search", "web_search"],
        "role": "ResearchAgent",
        "duration_seconds": 47.18
      },
      ...
    ]
  },
  "static_analysis": {
    "issue_count": 0,
    "issues": [],
    "autofix_changed_files": [],
    "tools": [
      {
        "name": "run_flake8",
        "duration_seconds": 0.0375,
        "issue_count": 0,
        "error": null
      },
      ...
    ]
  },
  "tests": {
    "initial": { "returncode": 0, "stdout": "...", "stderr": "..." },
    "after_autofix": null,
    "autofix_changed_files": []
  },
  "autofix": {
    "static_issues": {
      "enabled": true,
      "attempted": false,
      "changed_files": []
    },
    "tests": {
      "enabled": true,
      "attempted": false,
      "changed_files": []
    }
  }
}
```

说明：

- **plan.steps[].duration_seconds**：每个 Agent 子步骤的执行耗时；
- **static_analysis.tools[]**：每个静态工具的耗时、问题数和错误信息；
- **autofix**：区分“静态问题修复”和“测试失败修复”的启用状态与实际修改文件。

### 4.2 Markdown 报告

`main._write_markdown_report(...)` 会根据 JSON 报告生成一份简洁 Markdown，包含：

- 基本元信息（目标、Coding Task、输出目录、LLM 信息）；
- 一个步骤耗时表格；
- 一个静态分析工具耗时与问题数表格；
- 测试结果片段；
- 自动修复开关与变更文件列表。

这份报告可以直接挂在企业内部平台或 CI 结果页，便于快速阅读。

---

## 5. 与赛题 3 要求的对齐情况

赛题 3 关注的能力与本系统的对应关系如下：

- **“先做自主研究，再设计和实现项目代码”**  
  - Planner / Executor pipeline 明确区分了 research → design → code → test → doc 等阶段；
  - doc_research 场景可以只做 Deep Research + 报告；
  - code_review / code_scratch 场景在研究之后实际生成 demo 代码仓库并运行测试。

- **多技术文档输入（多模态 RAG）**  
  - `references/` + `--docs` 支持 PDF / PPTX / DOCX / TXT / MD；
  - `KnowledgeBase` + `kb_search` 提供统一的检索接口；
  - 可选 OpenAI / ModelScope embedding 实现向量检索。

- **长短期记忆与超长上下文管理**  
  - `MemoryManager` 将关键事件写入 JSONL；
  - Executor 在每步前从知识库与工具中获取精简上下文，避免一次性塞入超长 prompt。

- **Code Agent 的自我反思能力（bug shooting）**  
  - 执行失败时通过 `Reflector` 做 LLM 级别的错误分析；
  - 静态分析与测试失败都会触发基于 LLM 的自动修复尝试。

- **Agent 架构与可扩展性**  
  - Planner / Executor / Memory / Tools / Hooks / RAG 模块清晰分离；
  - `plugins/` 支持在不修改核心代码的情况下新增工具与 Hook；
  - `web_search` + `MCPClient` 演示了通用远程工具协议的封装。

- **多任务覆盖能力**  
  - 同一 runtime 已成功在多种 `coding_task` 下运行，并生成对应 `outputs/` 仓库与 `reports/run-*.json/.md`：
    - 代码质量类：集成 flake8/pylint/bandit/mypy 的代码审查与自动修复 Agent；
    - 应用类：图书管理 REST API（FastAPI + SQLite）、日志分析工具；
    - 数据/分析类：股票数据分析脚本（拉取数据、计算指标、输出图表）。
  - 评审可通过查看 `outputs/*` 与 `reports/*` 直接看到多任务表现。

---

## 6. 安装与环境准备

### 6.1 Python 依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

- `openai` / `requests`（LLM + MCP 调用）
- `pypdf` / `python-docx` / `python-pptx`（多格式文档加载）
- `flake8` / `pylint` / `bandit` / `mypy`（静态分析工具）

### 6.2 LLM 配置

系统通过环境变量或 `code_review_agent.yml` 读取 LLM 配置，常用方式包括：

- OpenAI：
  - `OPENAI_API_KEY`
  - 可选：`OPENAI_MODEL`、`OPENAI_BASE_URL`
- ModelScope / DeepSeek（OpenAI 协议）：
  - `MODELSCOPE_API_KEY`
  - `MODELSCOPE_BASE_URL`
  - `MODELSCOPE_MODEL`

也可以在 `code_review_agent.yml` 中设置：

```yaml
llm:
  provider: modelscope
  model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  api_key: "<your_api_key>"
  base_url: "https://api-inference.modelscope.cn/v1"
```

如未配置任何 API Key，系统会自动退化为 `DummyLLM`，用于本地流程演示。

### 6.3 Web 搜索（可选）

如需启用 `web_search` 工具，请设置：

- `WEB_SEARCH_ENDPOINT`：HTTP 接口地址
- 可选 `WEB_SEARCH_AUTH_HEADER`：认证头，例如 `Bearer xxx`

---

## 7. 使用方式与示例命令

在项目根目录 `复杂代码生成DeepCodeResearch/` 下运行：

### 7.1 典型：代码审查与修复 Agent（code_review）

```powershell
python main.py ^
  --goal "设计一个面向Python项目的自动化代码审查与修复Agent系统" ^
  --coding-task "实现一个集成 flake8、pylint、bandit、mypy 的 Python 代码审查与自动修复 Agent 项目" ^
  --scenario code_review
```

执行完成后：

- 生成 demo repo：`outputs/flake8_pylint_bandit_mypy_python_agent/`
- 生成报告：`reports/run-*.json` / `.md`
- 关键执行日志写入：`memory/session-*.jsonl`

### 7.2 通用复杂代码生成（code_scratch）

例如：图书管理 REST API

```powershell
python main.py ^
  --goal "实现一个简单的图书管理 REST API 服务" ^
  --coding-task "实现一个支持增删改查、分页和关键字搜索的图书管理 REST API，使用 FastAPI 和 SQLite" ^
  --scenario code_scratch
```

此任务会生成一个包含 FastAPI 结构的 demo 仓库，并附带简单的单元测试。

再如：股票数据分析脚本

```powershell
python main.py ^
  --goal "实现一个股票数据分析脚本" ^
  --coding-task "从公开API拉取指定股票历史价格，计算简单移动平均和波动率，并生成图表和文字分析报告" ^
  --scenario code_scratch
```

### 7.3 纯文档研究（doc_research）

```powershell
python main.py ^
  --goal "系统性调研多模态 Deep Research Agent 架构" ^
  --coding-task "输出一份关于多模态 RAG 架构和优化策略的研究报告" ^
  --scenario doc_research
```

可选地，通过 `--docs` 显式指定技术文档：

```powershell
python main.py ^
  --goal "..." ^
  --coding-task "..." ^
  --scenario doc_research ^
  --docs .\references\deepresearch_paper.pdf .\references\design_notes.md
```

---

## 8. 配置项说明（`code_review_agent.yml`）

`code_review_agent.yml` 用于以“项目级配置”的方式统一管理行为：

```yaml
llm:
  provider: openai  # 或 modelscope
  model: gpt-4.1-mini

static_tools:
  flake8: true
  pylint: true
  bandit: true
  mypy: true

autofix:
  static_issues: true      # 静态分析后尝试自动修复
  test_failures: true      # 单元测试失败时尝试自动修复

reports:
  enabled: true
  dir: reports             # 相对于 workspace.dir 的路径

workspace:
  dir: .
  memory_dir: memory
  kb_dir: knowledge_base
```

如果该文件不存在或解析失败，系统会自动回退到合理的默认值，不影响运行。

---

## 9. 总结

本仓库提供了一套完整的、多场景的复杂代码生成 Agent 运行时，特点是：

- 架构上与官方 MS-Agent 思想对齐（Planner / Executor / Memory / RAG / Hooks / Tools）；
- 工程层面集成了真实的代码质量工具、自我修复闭环与报告系统；
- 同一 runtime 下可在多个复杂 `coding_task` 上复用，无需针对每个任务重写流水线。

评审只需：

1. 浏览本 README 了解系统设计；
2. 查看 `reports/run-*.json` / `.md` 与 `memory/session-*.jsonl` 验证执行过程；
3. 运行若干示例命令，观察在不同任务下生成的 `outputs/*` 仓库及其测试/静态分析结果。

即可全面评估本系统在赛题 3「复杂代码生成 DeepCodeResearch」中的表现。

---

## 10. 开源许可与致谢

### 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

```
Copyright 2025 DeepCodeResearch Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### OpenAtom 开源基金会

本项目参加 [OpenAtom Agent Challenge](https://openatom.tech/agentchallenge)，致力于推动开源 AI Agent 技术的发展与创新。

### 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进本项目。

