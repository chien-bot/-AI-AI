from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

from .config import LLMConfig


@dataclass
class ChatMessage:
    role: str
    content: str


class BaseLLM:
    """LLM 抽象接口。"""

    def chat(self, messages: Iterable[ChatMessage], **kwargs: Any) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class DummyLLM(BaseLLM):
    """当未配置真实 LLM 时使用，用于流程调试。

    它不会调用外部网络，只会把输入摘要后返回占位结果。
    """

    def chat(self, messages: Iterable[ChatMessage], **kwargs: Any) -> str:
        last = list(messages)[-1] if messages else ChatMessage("user", "")
        return (
            "[DummyLLM 输出，仅用于调试]\n"
            f"收到的最后一条消息: {last.content[:200]}...\n"
            "请在接入真实 LLM 后替换为实际推理结果。"
        )


class OpenAILLM(BaseLLM):
    """简单的 OpenAI / ModelScope Chat LLM 封装。"""

    def __init__(self, config: LLMConfig) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失时提示
            raise RuntimeError(
                "未安装 openai 库，请先运行: pip install openai"
            ) from exc

        if not config.api_key:
            raise RuntimeError("未检测到 LLM API Key，无法调用远程模型接口。")

        client_kwargs: Dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self._client = OpenAI(**client_kwargs)
        self._model = config.model

    def chat(self, messages: Iterable[ChatMessage], **kwargs: Any) -> str:
        payload: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            **kwargs,
        )
        choice = resp.choices[0]
        return choice.message.content or ""


def build_llm(config: LLMConfig) -> BaseLLM:
    """根据配置返回合适的 LLM 实例。"""

    # ModelScope 也兼容 OpenAI 协议，因此在这里与 OpenAI 走同一实现
    if config.provider in {"openai", "modelscope"} and config.api_key:
        return OpenAILLM(config)

    # 默认退化为 DummyLLM，保证流程可跑
    return DummyLLM()
