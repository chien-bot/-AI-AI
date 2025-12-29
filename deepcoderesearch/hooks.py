from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Any, List


HookFunc = Callable[..., None]


class HookManager:
    """简单的生命周期 Hook 系统。

    支持的典型事件：
    - pre_planning / post_planning
    - pre_step / post_step
    - pre_tool / post_tool
    - on_error
    插件可以向这些事件注册回调，实现观测/干预能力。
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, List[HookFunc]] = defaultdict(list)

    def add(self, event: str, func: HookFunc) -> None:
        self._hooks[event].append(func)

    def emit(self, event: str, **kwargs: Any) -> None:
        for func in self._hooks.get(event, []):
            try:
                func(**kwargs)
            except Exception as exc:  # pragma: no cover - Hook 不应中断主流程
                print(f"[Hook] 事件 {event} 执行失败: {exc}")
