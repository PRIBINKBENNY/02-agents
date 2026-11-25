"""Limits and permission checks that sit between the model and the world."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

Approval = Callable[[str, dict], bool]  # (tool_name, args) -> approved?


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Shared, thread-safe budget. Sub-agents draw from the same pool as the orchestrator."""

    max_steps: int = 40
    max_tokens: int = 200_000
    max_cost_usd: float = 0.50
    # $ per 1M tokens; defaults are roughly a small hosted model. Override per model.
    price_in: float = 0.15
    price_out: float = 0.60
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.steps += 1
            self.tokens += input_tokens + output_tokens
            self.cost_usd += (input_tokens * self.price_in + output_tokens * self.price_out) / 1_000_000

    def check(self) -> None:
        with self._lock:
            if self.steps >= self.max_steps:
                raise BudgetExceeded(f"max steps reached ({self.max_steps})")
            if self.tokens >= self.max_tokens:
                raise BudgetExceeded(f"token budget reached ({self.tokens}/{self.max_tokens})")
            if self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(f"cost budget reached (${self.cost_usd:.4f}/${self.max_cost_usd})")

    def snapshot(self) -> dict:
        return {"steps": self.steps, "tokens": self.tokens, "cost_usd": round(self.cost_usd, 6)}


def safe_path(workspace: Path, relative: str) -> Path:
    """Resolve a path inside the workspace; refuse anything that escapes it."""
    root = workspace.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f"blocked: '{relative}' is outside the agent workspace")
    return target


def approval_policy(mode: Literal["ask", "auto", "deny"]) -> Approval:
    if mode == "auto":
        return lambda tool, args: True
    if mode == "deny":
        return lambda tool, args: False

    lock = threading.Lock()  # parallel sub-agents must not interleave prompts

    def ask(tool: str, args: dict) -> bool:
        with lock:
            preview = {k: (v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v) for k, v in args.items()}
            print(f"\n⚠️  Agent wants to run risky tool '{tool}' with {preview}")
            return input("   approve? [y/N] ").strip().lower() in {"y", "yes"}

    return ask
