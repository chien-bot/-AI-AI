from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import AgentConfig


@dataclass
class MemoryEvent:
    """单条记忆条目，用于记录 Agent 的行为轨迹。"""

    timestamp: str
    type: str
    content: Dict[str, Any]


class MemoryManager:
    """管理短期/长期记忆。

    - 短期记忆：存放于内存中，用于当前对话轮
    - 长期记忆：以 JSONL 写入磁盘，便于赛题评估与复现
    """

    def __init__(self, config: AgentConfig, session_id: Optional[str] = None) -> None:
        self._config = config
        self._config.ensure_dirs()

        if session_id is None:
            session_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.session_id = session_id

        self._short_term: List[MemoryEvent] = []
        self._file_path = self._config.memory_dir / f"session-{self.session_id}.jsonl"

    # -------- 公共 API --------
    def add_event(self, event_type: str, content: Dict[str, Any]) -> None:
        event = MemoryEvent(
            timestamp=datetime.utcnow().isoformat(),
            type=event_type,
            content=content,
        )
        self._short_term.append(event)
        self._append_to_file(event)

    def get_recent_events(self, limit: int = 20) -> List[MemoryEvent]:
        return self._short_term[-limit:]

    # -------- 内部实现 --------
    def _append_to_file(self, event: MemoryEvent) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("a", encoding="utf-8") as f:
            json.dump(asdict(event), f, ensure_ascii=False)
            f.write("\n")
