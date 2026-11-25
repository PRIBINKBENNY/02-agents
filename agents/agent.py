"""A single agent: plan -> act -> check, looping until the work passes its checks or the budget runs out."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .guardrails import Budget, BudgetExceeded
from .tools import TOOLS, FinishArgs, ToolContext, execute

Check = Callable[[Path], "str | None"]  # returns an error message, or None if the check passes

BASE_SYSTEM = """You are {name}, an autonomous agent that completes multi-step tasks on your own.

How you work:
1. PLAN: break the task into concrete steps.
2. ACT: use tools one step at a time. Read tool results carefully; if a tool errors, adapt.
3. CHECK: before calling `finish`, verify your work (re-read files you wrote, run code/tests in the sandbox).

Rules:
- All files live in your workspace. Paths are relative to it. You cannot touch anything outside it.
- Code only runs inside the sandbox, with no network access.
- Never invent facts or sources. If you could not find something, say so.
- Call `finish` only when the task is truly done; your result will be checked automatically.
{extra}"""

KEEP_FULL_TOOL_OUTPUTS = 6  # older tool outputs are shortened to keep the context small


@dataclass
class AgentResult:
    agent: str
    ok: bool
    answer: str = ""
    files: list[str] = field(default_factory=list)
    steps: int = 0
    reason: str = ""


class Agent:
    def __init__(self, name: str, llm, ctx: ToolContext, budget: Budget, tools: set[str] | None = None,
                 extra_instructions: str = "", checks: list[Check] | None = None,
                 max_steps: int | None = None, max_check_failures: int = 3):
        self.name = name
        self.llm = llm
        self.ctx = ctx
        self.ctx.agent = name
        self.budget = budget
        self.tools = set(tools or TOOLS) | {"finish"}
        self.extra = extra_instructions
        self.checks = checks or []
        self.max_steps = max_steps
        self.max_check_failures = max_check_failures
        self.messages: list[dict] = []

    # ------------------------------------------------------------------
    def run(self, task: str) -> AgentResult:
        t = self.ctx.tracer
        t.log(self.name, "start", task=task)
        self.messages = [
            {"role": "system", "content": BASE_SYSTEM.format(name=self.name, extra=self.extra)},
            {"role": "user", "content": task},
        ]
        steps = 0
        check_failures = 0
        idle_turns = 0
        schemas = [TOOLS[n].schema() for n in sorted(self.tools)]

        try:
            # ---- PLAN ----
            self.budget.check()
            plan = self.llm.complete(self.messages + [{"role": "user", "content":
                    "Write a short numbered plan for this task. Do not call tools yet."}])
            self.budget.charge(plan.input_tokens, plan.output_tokens)
            t.log(self.name, "plan", text=plan.content, tokens=plan.input_tokens + plan.output_tokens)
            self.messages += [
                {"role": "assistant", "content": f"Plan:\n{plan.content}"},
                {"role": "user", "content": "Good. Execute the plan with tools. Call `finish` when done and verified."},
            ]

            # ---- ACT / CHECK loop ----
            while True:
                self.budget.check()
                if self.max_steps is not None and steps >= self.max_steps:
                    raise BudgetExceeded(f"{self.name} hit its own step limit ({self.max_steps})")
                self._compact()
                resp = self.llm.complete(self.messages, schemas)
                self.budget.charge(resp.input_tokens, resp.output_tokens)
                self.messages.append(resp.as_message())

                if not resp.tool_calls:
                    idle_turns += 1
                    if idle_turns >= 3:
                        return self._done(False, steps, reason="stopped calling tools without finishing",
                                          answer=resp.content)
                    self.messages.append({"role": "user", "content":
                                          "Continue: call a tool, or call `finish` if the task is complete."})
                    continue
                idle_turns = 0

                for tc in resp.tool_calls:
                    steps += 1
                    args = _safe_json(tc.arguments)
                    t.log(self.name, "tool_call", step=steps, tool=tc.name, args=args,
                          tokens=resp.input_tokens + resp.output_tokens, budget=self.budget.snapshot())

                    if tc.name == "finish":
                        result = self._try_finish(tc, steps)
                        if result is not None:
                            return result
                        check_failures += 1
                        if check_failures >= self.max_check_failures:
                            return self._done(False, steps, reason=f"failed checks {check_failures} times")
                        continue

                    ok, out = execute(self.ctx, tc.name, tc.arguments, self.tools)
                    t.log(self.name, "tool_result", step=steps, tool=tc.name, ok=ok, output=out[:2000])
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:12000]})

        except BudgetExceeded as e:
            t.log(self.name, "budget_stop", reason=str(e), budget=self.budget.snapshot())
            return self._done(False, steps, reason=str(e))

    # ------------------------------------------------------------------
    def _try_finish(self, tc, steps: int) -> AgentResult | None:
        try:
            fa = FinishArgs.model_validate(_safe_json(tc.arguments))
        except Exception as e:
            self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Error: invalid finish args: {e}"})
            return None
        errors = []
        for f in fa.files:
            p = (self.ctx.workspace / f)
            if not p.is_file() or p.stat().st_size == 0:
                errors.append(f"file '{f}' is missing or empty")
        for check in self.checks:
            try:
                err = check(self.ctx.workspace)
            except Exception as e:
                err = f"check crashed: {e}"
            if err:
                errors.append(err)
        if errors:
            msg = "Checks failed:\n- " + "\n- ".join(errors) + "\nFix these, then call finish again."
            self.ctx.tracer.log(self.name, "check", ok=False, text=msg)
            self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": msg})
            return None
        self.ctx.tracer.log(self.name, "check", ok=True, text="all checks passed")
        self.ctx.tracer.log(self.name, "finish", text=fa.answer, files=fa.files, budget=self.budget.snapshot())
        return self._done(True, steps, answer=fa.answer, files=fa.files)

    def _done(self, ok: bool, steps: int, reason: str = "", answer: str = "", files=None) -> AgentResult:
        return AgentResult(self.name, ok, answer, files or [], steps, reason)

    def _compact(self) -> None:
        """Shorten old tool outputs so long runs don't blow up the context window."""
        tool_idx = [i for i, m in enumerate(self.messages) if m.get("role") == "tool"]
        for i in tool_idx[:-KEEP_FULL_TOOL_OUTPUTS]:
            c = self.messages[i]["content"]
            if len(c) > 600:
                self.messages[i]["content"] = c[:500] + f"\n...[{len(c) - 500} chars trimmed from an old step]"


def _safe_json(raw: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"_raw": raw}
    except json.JSONDecodeError:
        return {"_raw": raw}
