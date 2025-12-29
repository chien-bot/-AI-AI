from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os


@dataclass
class LLMConfig:
    """LLM 相关配置。

    默认使用 OpenAI，如果没有提供 API Key，则会退化为 DummyLLM。
    同时兼容 ModelScope 这类 OpenAI 协议兼容服务。
    """

    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    # 支持多种环境变量，按顺序优先级读取
    model: str = field(
        default_factory=lambda: (
            os.getenv("OPENAI_MODEL")
            or os.getenv("LLM_MODEL")
            or os.getenv("MODELSCOPE_MODEL")
            or "gpt-4.1-mini"
        )
    )
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("MODELSCOPE_API_KEY")
            ),
            base_url=(
                os.getenv("OPENAI_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or os.getenv("MODELSCOPE_BASE_URL")
            ),
        )


@dataclass
class StaticToolsConfig:
    """Configuration switches for static analysis tools.

    These map to the code quality tools exposed via plugins/code_quality.py.
    """

    flake8: bool = True
    pylint: bool = True
    bandit: bool = True
    mypy: bool = True


@dataclass
class AgentConfig:
    """Agent 全局配置。"""

    workspace_dir: Path = field(default_factory=lambda: Path.cwd())
    memory_dir: Path = field(default_factory=lambda: Path.cwd() / "memory")
    kb_dir: Path = field(default_factory=lambda: Path.cwd() / "knowledge_base")

    llm: LLMConfig = field(default_factory=LLMConfig.from_env)

    # 项目级开关与企业化配置
    static_tools: StaticToolsConfig = field(default_factory=StaticToolsConfig)
    # 是否在静态分析发现问题后，尝试自动修复代码
    enable_static_autofix: bool = True
    # 是否在单元测试失败时，触发基于测试输出来的自动修复
    enable_test_autofix: bool = True
    # 报告相关配置
    reports_enabled: bool = True
    report_dir: Optional[Path] = None

    def ensure_dirs(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    # ---- 静态工具与配置文件相关辅助方法 ----

    def get_static_tool_names(self) -> list[str]:
        """Return the list of static-analysis tool names to invoke as Agent tools."""

        names: list[str] = []
        if self.static_tools.flake8:
            names.append("run_flake8")
        if self.static_tools.pylint:
            names.append("run_pylint")
        if self.static_tools.bandit:
            names.append("run_bandit")
        if self.static_tools.mypy:
            names.append("run_mypy")
        return names

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "AgentConfig":
        """Load AgentConfig from a YAML file if present, otherwise use defaults.

        - 默认配置仍然依赖当前工作目录以及环境变量（LLM_CONFIG.from_env）
        - 如果当前目录存在 `code_review_agent.yml`，则将其中字段视为覆盖：
          - llm.*            覆盖 LLM provider/model/api_key/base_url
          - static_tools.*   控制是否启用 flake8/pylint/bandit/mypy
          - autofix.*        控制自动修复行为
          - reports.*        控制 JSON 报告生成与输出目录
          - workspace.*      覆盖工作目录/记忆目录/知识库存储目录
        """

        cfg = cls()

        if path is None:
            path = Path.cwd() / "code_review_agent.yml"

        if not path.exists():
            return cfg

        try:
            import yaml  # type: ignore[import]
        except Exception:
            # 如果未安装 PyYAML，则忽略文件，使用默认配置
            return cfg

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            # 配置文件解析失败时，保底返回默认配置，避免影响主流程
            return cfg

        if not isinstance(raw, dict):
            return cfg

        # ---- LLM 配置覆盖 ----
        llm_data = raw.get("llm")
        if isinstance(llm_data, dict):
            provider = llm_data.get("provider")
            model = llm_data.get("model")
            api_key = llm_data.get("api_key")
            base_url = llm_data.get("base_url")
            if provider is not None:
                cfg.llm.provider = str(provider)
            if model is not None:
                cfg.llm.model = str(model)
            if api_key is not None:
                cfg.llm.api_key = str(api_key)
            if base_url is not None:
                cfg.llm.base_url = str(base_url)

        # ---- 静态分析工具开关 ----
        static_data: Dict[str, Any] | None = raw.get("static_tools") or raw.get("tools")
        if isinstance(static_data, dict):
            for key in ("flake8", "pylint", "bandit", "mypy"):
                if key in static_data:
                    setattr(cfg.static_tools, key, bool(static_data[key]))

        # ---- 自动修复行为 ----
        autofix_data = raw.get("autofix")
        if isinstance(autofix_data, dict):
            if "static_issues" in autofix_data:
                cfg.enable_static_autofix = bool(autofix_data["static_issues"])
            if "test_failures" in autofix_data:
                cfg.enable_test_autofix = bool(autofix_data["test_failures"])

        # ---- 报告相关配置 ----
        reports_data = raw.get("reports")
        if isinstance(reports_data, dict):
            if "enabled" in reports_data:
                cfg.reports_enabled = bool(reports_data["enabled"])
            if "dir" in reports_data and reports_data["dir"]:
                # 报告目录允许为相对路径，相对于 workspace_dir
                cfg.report_dir = Path(str(reports_data["dir"]))

        # ---- 工作目录相关配置 ----
        workspace_data = raw.get("workspace")
        if isinstance(workspace_data, dict):
            if "dir" in workspace_data and workspace_data["dir"]:
                cfg.workspace_dir = Path(str(workspace_data["dir"]))
            if "memory_dir" in workspace_data and workspace_data["memory_dir"]:
                cfg.memory_dir = Path(str(workspace_data["memory_dir"]))
            if "kb_dir" in workspace_data and workspace_data["kb_dir"]:
                cfg.kb_dir = Path(str(workspace_data["kb_dir"]))

        # 简单提示，方便用户确认配置来源
        print(f"[Config] Loaded project config from {path}")
        return cfg
