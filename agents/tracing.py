"""Every step, tool call, input, output, token count and cost goes to a JSONL log."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class Tracer:
    def __init__(self, path: Path, echo: bool = True):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()
        self.events: list[dict] = []

    def log(self, agent: str, kind: str, **data) -> None:
        event = {"ts": round(time.time(), 3), "agent": agent, "kind": kind, **data}
        with self._lock:
            self.events.append(event)
            with self.path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")
            if self.echo:
                self._print(event)

    @staticmethod
    def _print(e: dict) -> None:
        tag = f"[{e['agent']}]"
        k = e["kind"]
        if k == "tool_call":
            args = json.dumps(e.get("args", {}))
            print(f"{tag} step {e.get('step')}: {e['tool']}({args[:120]})")
        elif k == "tool_result":
            status = "ok" if e.get("ok") else "ERROR"
            print(f"{tag}   -> {status}: {str(e.get('output', ''))[:140]!r}")
        elif k in {"blocked", "denied"}:
            print(f"{tag}   ⛔ {k}: {e.get('reason')}")
        elif k in {"plan", "check", "finish", "budget_stop", "start", "error"}:
            msg = e.get("text") or e.get("reason") or e.get("task") or ""
            print(f"{tag} {k}: {str(msg)[:200]}")
