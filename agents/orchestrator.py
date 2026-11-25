"""Orchestrator: splits a big task into sub-tasks, runs sub-agents in parallel, then a writer agent
merges their work and the final result is checked."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .agent import Agent, AgentResult, Check
from .guardrails import Approval, Budget, BudgetExceeded, approval_policy
from .sandbox import get_sandbox
from .tools import ToolContext
from .tracing import Tracer

ROLE_TOOLS: dict[str, set[str]] = {
    "researcher": {"web_search", "fetch_url", "read_file", "write_file", "list_files"},
    "reader": {"read_file", "list_files", "fetch_url", "write_file"},
    "coder": {"read_file", "write_file", "list_files", "run_python"},
    "writer": {"read_file", "write_file", "list_files"},
}

PLANNER_PROMPT = """You are the orchestrator of a team of agents. Split the task into 2-4 independent sub-tasks
that can run IN PARALLEL. Each sub-task is done by one sub-agent with a role:
- researcher: searches the web and fetches pages
- reader: reads files in the workspace or given URLs
- coder: writes and runs Python in a sandbox
- writer: drafts text from files in the workspace

Each sub-agent writes its findings to its own markdown file under notes/.
Then a final writer merges everything into `final_file`.

Reply with ONLY this JSON (no prose, no code fences):
{"subtasks": [{"id": "s1", "role": "researcher", "instructions": "...", "output_file": "notes/s1.md"}],
 "final_file": "report.md"}"""


class SubTask(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,20}$")
    role: Literal["researcher", "reader", "coder", "writer"]
    instructions: str = Field(min_length=10)
    output_file: str

    @field_validator("output_file")
    @classmethod
    def in_notes(cls, v: str) -> str:
        if ".." in v or v.startswith("/"):
            raise ValueError("output_file must be a relative path inside the workspace")
        return v


class Plan(BaseModel):
    subtasks: list[SubTask] = Field(min_length=1, max_length=6)
    final_file: str = "report.md"


@dataclass
class RunResult:
    ok: bool
    task: str
    plan: Plan | None
    sub_results: list[AgentResult] = field(default_factory=list)
    final: AgentResult | None = None
    budget: dict = field(default_factory=dict)
    seconds: float = 0.0
    log_file: str = ""
    reason: str = ""


class Orchestrator:
    def __init__(self, llm, workspace: str | Path = "workspace", budget: Budget | None = None,
                 approve: Approval | None = None, sandbox: str = "auto", max_parallel: int = 3,
                 sub_agent_steps: int = 15, log_dir: str | Path = "runs", echo: bool = True):
        self.llm = llm
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.budget = budget or Budget()
        self.approve = approve or approval_policy("ask")
        self.sandbox = get_sandbox(self.workspace, sandbox)
        self.max_parallel = max_parallel
        self.sub_agent_steps = sub_agent_steps
        self.tracer = Tracer(Path(log_dir) / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl", echo=echo)

    def _ctx(self) -> ToolContext:
        return ToolContext(self.workspace, self.sandbox, self.approve, self.tracer)

    # ------------------------------------------------------------------
    def plan(self, task: str) -> Plan:
        messages = [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": task}]
        last_err = ""
        for _ in range(3):
            self.budget.check()
            resp = self.llm.complete(messages)
            self.budget.charge(resp.input_tokens, resp.output_tokens)
            try:
                plan = Plan.model_validate(_extract_json(resp.content))
                self.tracer.log("orchestrator", "plan", text=plan.model_dump_json(), budget=self.budget.snapshot())
                return plan
            except (ValueError, ValidationError) as e:
                last_err = str(e)
                messages += [{"role": "assistant", "content": resp.content},
                             {"role": "user", "content": f"That plan is invalid: {e}. Reply with valid JSON only."}]
        raise ValueError(f"planner failed to produce a valid plan: {last_err}")

    def run(self, task: str, checks: list[Check] | None = None) -> RunResult:
        t0 = time.perf_counter()
        self.tracer.log("orchestrator", "start", task=task, sandbox=getattr(self.sandbox, "name", None))
        try:
            plan = self.plan(task)
        except (ValueError, BudgetExceeded) as e:
            self.tracer.log("orchestrator", "error", reason=str(e))
            return RunResult(False, task, None, budget=self.budget.snapshot(), reason=str(e),
                             seconds=time.perf_counter() - t0, log_file=str(self.tracer.path))

        # ---- sub-agents in parallel ----
        def run_sub(i_st):
            i, st = i_st
            agent = Agent(f"sub-agent {i}", self.llm, self._ctx(), self.budget, ROLE_TOOLS[st.role],
                          extra_instructions=f"Your role: {st.role}. Save your findings to `{st.output_file}` "
                                             f"(markdown, with source URLs where relevant), then call finish "
                                             f"with files=['{st.output_file}'].",
                          max_steps=self.sub_agent_steps)
            return agent.run(f"Overall goal: {task}\n\nYour sub-task: {st.instructions}")

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            sub_results = list(pool.map(run_sub, enumerate(plan.subtasks, start=1)))

        # ---- writer merges ----
        notes = [st.output_file for st in plan.subtasks]
        writer = Agent("writer", self.llm, self._ctx(), self.budget, ROLE_TOOLS["writer"],
                       extra_instructions=f"Read the sub-agent notes {notes} and write the final result to "
                                          f"`{plan.final_file}`. Keep source links. If a note is missing, say "
                                          f"what is missing rather than inventing it. Call finish with "
                                          f"files=['{plan.final_file}'].",
                       checks=checks or [], max_steps=self.sub_agent_steps)
        status = "\n".join(f"- {r.agent}: {'done' if r.ok else 'FAILED (' + r.reason + ')'}" for r in sub_results)
        final = writer.run(f"Goal: {task}\n\nSub-agent status:\n{status}")

        ok = final.ok
        self.tracer.log("orchestrator", "finish", text=f"ok={ok}", budget=self.budget.snapshot())
        return RunResult(ok, task, plan, sub_results, final, self.budget.snapshot(),
                         round(time.perf_counter() - t0, 2), str(self.tracer.path),
                         reason="" if ok else final.reason)


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))
