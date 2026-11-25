"""agent run "research and write a report on X"      (team: orchestrator + parallel sub-agents)
agent run --single "write and test a fizzbuzz"     (one agent)
agent eval tasks/tasks.yaml                        (task test set -> pass rate)"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import Agent
from .guardrails import Budget, approval_policy
from .llm import OpenAICompatLLM
from .orchestrator import Orchestrator
from .sandbox import get_sandbox
from .tools import ToolContext
from .tracing import Tracer


def _load_dotenv() -> None:
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.split("#")[0].strip().strip('"'))


def main(argv=None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--single", action="store_true", help="one agent instead of orchestrator + sub-agents")
    r.add_argument("--workspace", default="workspace")
    r.add_argument("--approve", choices=["ask", "auto", "deny"], default="ask")
    r.add_argument("--sandbox", choices=["auto", "docker", "e2b", "local"], default="auto")
    r.add_argument("--max-steps", type=int, default=60)
    r.add_argument("--max-tokens", type=int, default=400_000)
    r.add_argument("--max-cost", type=float, default=0.50)
    r.add_argument("--parallel", type=int, default=3)
    e = sub.add_parser("eval")
    e.add_argument("tasks", nargs="?", default="tasks/tasks.yaml")
    e.add_argument("--only", nargs="*")
    args = p.parse_args(argv)

    if args.cmd == "eval":
        from .evaluate import run_eval

        return run_eval(args.tasks, args.only)

    llm = OpenAICompatLLM()
    budget = Budget(max_steps=args.max_steps, max_tokens=args.max_tokens, max_cost_usd=args.max_cost)
    approve = approval_policy(args.approve)

    if args.single:
        ws = Path(args.workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        tracer = Tracer(Path("runs") / "single.jsonl")
        ctx = ToolContext(ws, get_sandbox(ws, args.sandbox), approve, tracer)
        res = Agent("agent", llm, ctx, budget).run(args.task)
        print(json.dumps({"ok": res.ok, "answer": res.answer, "files": res.files, "steps": res.steps,
                          "reason": res.reason, "budget": budget.snapshot()}, indent=2))
        return 0 if res.ok else 1

    orch = Orchestrator(llm, args.workspace, budget, approve, args.sandbox, args.parallel)
    res = orch.run(args.task)
    print("\n" + "=" * 60)
    print(f"ok: {res.ok}   time: {res.seconds}s   budget used: {res.budget}")
    for s in res.sub_results:
        print(f"  {s.agent}: {'ok' if s.ok else 'failed - ' + s.reason} ({s.steps} steps)")
    if res.final:
        print(f"  writer: {'ok' if res.final.ok else 'failed - ' + res.final.reason} -> {res.final.files}")
    print(f"log: {res.log_file}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
