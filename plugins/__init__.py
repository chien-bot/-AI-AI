from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Optional

from deepcoderesearch.tools import ToolRegistry
from deepcoderesearch.hooks import HookManager


def load_plugins(tools: ToolRegistry, hooks: HookManager, plugin_dir: Optional[Path] = None) -> None:
    """加载 `plugins/` 目录下的所有插件模块。

    约定：每个插件模块（*.py）如果定义了 `register(tools, hooks)` 函数，
    则在加载时自动调用，用于注册工具和 Hook。
    """

    if plugin_dir is None:
        plugin_dir = Path(__file__).parent

    for file in plugin_dir.glob("*.py"):
        if file.name == "__init__.py":
            continue
        module_name = f"plugins.{file.stem}"
        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - 插件加载失败不应终止主流程
            print(f"[Plugins] 加载插件 {module_name} 失败: {exc}")
            continue

        register = getattr(module, "register", None)
        if callable(register):
            try:
                register(tools, hooks)
                print(f"[Plugins] 已加载插件: {module_name}")
            except Exception as exc:  # pragma: no cover
                print(f"[Plugins] 插件 {module_name} 注册失败: {exc}")
