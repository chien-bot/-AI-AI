from __future__ import annotations

from .llm import BaseLLM, ChatMessage


class Reflector:
    """简单的自我反思模块，用于在执行出错后给出分析建议。"""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def reflect(self, goal: str, step: str, error: str, partial_output: str = "") -> str:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "你是一名代码调试专家，擅长分析 Agent 在执行任务时出现的错误，"
                    "给出可能原因与下一步改进建议。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"任务目标: {goal}\n"
                    f"当前步骤: {step}\n"
                    f"错误信息: {error}\n"
                    f"部分输出: {partial_output}\n\n"
                    "请用要点形式给出：1) 可能原因 2) 建议的修复方向。"
                ),
            ),
        ]
        return self._llm.chat(messages)
