"""Model client with tool calling. One OpenAI-compatible API covers OpenRouter, OpenAI, Groq and vLLM,
so you can swap models and compare cost/quality on the same task by changing LLM_MODEL."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model (may be invalid!)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def as_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}
                for t in self.tool_calls
            ]
        return msg


class OpenAICompatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2, http_client=None):
        from openai import OpenAI

        self.model = model or os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
        self._client = OpenAI(
            base_url=base_url or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key or os.environ.get("LLM_API_KEY", "not-needed"),
            http_client=http_client,
        )
        self.temperature = temperature

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in (msg.tool_calls or [])]
        usage = resp.usage
        return LLMResponse(
            content=msg.content or "",
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class ScriptedLLM:
    """Deterministic fake for tests. `script(messages, tools) -> LLMResponse` decides each turn."""

    def __init__(self, script: Callable[[list[dict], list[dict] | None], LLMResponse]):
        self.script = script
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls += 1
        return self.script(messages, tools)


_counter = 0


def call(name: str, **args) -> ToolCall:
    """Helper for scripts: build a ToolCall."""
    global _counter
    _counter += 1
    return ToolCall(id=f"call_{_counter}", name=name, arguments=json.dumps(args))
