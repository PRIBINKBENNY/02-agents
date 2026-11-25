"""Run the task test set (tasks/tasks.yaml) against a real model and report a pass rate.

Each task: a prompt, a mode (single | team), optional seed files, an approval policy, and checks:
  file_exists: path            min_words: {path, n}
  contains:    {path, all: [...]}   contains_any: {path, any: [...]}
  file_absent: path            unchanged: {path, content}
  sandbox_python: code that must exit 0 when run in the sandbox (e.g. asserts on the agent's code)
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import yaml

from .agent import Agent
from .guardrails import Budget, approval_policy
from .llm import OpenAICompatLLM
from .orchestrator import Orchestrator
from .sandbox import get_sandbox
from .tools import ToolContext
from .tracing import Tracer


def build_checks(specs: list[dict], sandbox) -> list:
    checks = []
    for spec in specs:
        (kind, val), = spec.items()

        def make(kind=kind, val=val):
            def check(ws: Path):
                if kind == "file_exists":
                    return None if (ws / val).is_file() else f"{val} does not exist"
                if kind == "file_absent":
                    return None if not (ws / val).exists() else f"{val} should not exist"
                text = (ws / val["path"]).read_text(errors="ignore") if (ws / val.get("path", "_")).is_file() else ""
                if kind == "min_words":
                    n = len(text.split())
                    return None if n >= val["n"] else f"{val['path']} has {n} words, needs {val['n']}"
                if kind == "contains":
                    missing = [s for s in val["all"] if s.lower() not in text.lower()]
                    return None if not missing else f"{val['path']} is missing {missing}"
                if kind == "contains_any":
                    return None if any(s.lower() in text.lower() for s in val["any"]) else \
                        f"{val['path']} contains none of {val['any']}"
                if kind == "unchanged":
                    return None if text == val["content"] else f"{val['path']} was modified"
                if kind == "sandbox_python":
                    if sandbox is None:
                        return "no sandbox to run the check in"
                    res = sandbox.run_python(val)
                    return None if res.exit_code == 0 else f"code check failed:\n{res.summary(800)}"
                return f"unknown check {kind}"
            return check
        checks.append(make())
    return checks


def run_eval(tasks_file: str, only: list[str] | None = None) -> int:
    tasks = yaml.safe_load(Path(tasks_file).read_text())["tasks"]
    if only:
        tasks = [t for t in tasks if t["id"] in only]
    llm = OpenAICompatLLM()
    results = []
    for task in tasks:
        ws = Path("eval_runs") / task["id"]
        shutil.rmtree(ws, ignore_errors=True)
        ws.mkdir(parents=True)
        for rel, content in (task.get("seed") or {}).items():
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            (ws / rel).write_text(content)
        budget = Budget(**task.get("budget", {}))
        approve = approval_policy(task.get("approve", "deny"))
        sandbox = get_sandbox(ws.resolve(), "auto")
        checks = build_checks(task.get("checks", []), sandbox)
        t0 = time.perf_counter()
        print(f"\n### {task['id']} ({task.get('mode', 'single')})")
        if task.get("mode") == "team":
            orch = Orchestrator(llm, ws, budget, approve, echo=False, log_dir=f"eval_runs/logs/{task['id']}")
            agent_ok = orch.run(task["prompt"], checks).ok
        else:
            tracer = Tracer(Path(f"eval_runs/logs/{task['id']}.jsonl"), echo=False)
            ctx = ToolContext(ws.resolve(), sandbox, approve, tracer)
            agent_ok = Agent("agent", llm, ctx, budget, checks=checks).run(task["prompt"]).ok
        # independent final check (the agent may have "finished" without passing)
        failures = [e for e in (c(ws.resolve()) for c in checks) if e]
        if task.get("expect_refusal"):
            passed = not failures  # passing = nothing harmful happened
        else:
            passed = agent_ok and not failures
        results.append({"id": task["id"], "passed": passed, "failures": failures,
                        "steps": budget.steps, "cost_usd": round(budget.cost_usd, 5),
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(("PASS" if passed else "FAIL"), json.dumps(results[-1]))

    n_pass = sum(r["passed"] for r in results)
    print(f"\npass rate: {n_pass}/{len(results)} = {100 * n_pass / max(len(results), 1):.0f}%")
    Path("eval_runs/results.json").write_text(json.dumps(results, indent=2))
    return 0 if n_pass == len(results) else 1
