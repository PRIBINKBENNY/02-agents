"""Tools the agent can call. Every call is validated against a Pydantic schema before it runs,
so a malformed call fails loudly (and the error goes back to the model) instead of doing something odd."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from .guardrails import Approval, safe_path
from .tracing import Tracer

Risk = Literal["safe", "risky", "exec"]


@dataclass
class ToolContext:
    workspace: Path
    sandbox: object | None
    approve: Approval
    tracer: Tracer
    agent: str = "agent"


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[ToolContext, BaseModel], str]
    risk: Risk = "safe"

    def schema(self) -> dict:
        params = self.args.model_json_schema()
        params.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": params}}


class ToolError(Exception):
    pass


# ---------------- argument schemas ----------------
class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)


class FetchArgs(BaseModel):
    url: str = Field(pattern=r"^https?://")
    max_chars: int = Field(default=8000, ge=500, le=30000)


class PathArgs(BaseModel):
    path: str = Field(default=".", max_length=300)


class WriteArgs(BaseModel):
    path: str = Field(max_length=300)
    content: str = Field(max_length=200_000)
    append: bool = False


class CodeArgs(BaseModel):
    code: str = Field(min_length=1, max_length=50_000)


class FinishArgs(BaseModel):
    answer: str = Field(min_length=1, description="final answer or summary of what was done")
    files: list[str] = Field(default_factory=list, description="files in the workspace that hold the result")


# ---------------- implementations ----------------
def _search(ctx: ToolContext, a: SearchArgs) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ToolError("web search unavailable: pip install ddgs") from e
    results = list(DDGS().text(a.query, max_results=a.max_results))
    if not results:
        return "no results"
    return "\n\n".join(f"{i}. {r.get('title')}\n   {r.get('href')}\n   {r.get('body')}" for i, r in enumerate(results, 1))


_TAGS = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def _fetch(ctx: ToolContext, a: FetchArgs) -> str:
    import httpx

    r = httpx.get(a.url, timeout=20, follow_redirects=True, headers={"User-Agent": "agents-demo/0.1"})
    r.raise_for_status()
    text = r.text
    if "html" in r.headers.get("content-type", ""):
        text = html.unescape(_TAGS.sub(" ", text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[: a.max_chars] + (" ...[truncated]" if len(text) > a.max_chars else "")


def _list(ctx: ToolContext, a: PathArgs) -> str:
    root = safe_path(ctx.workspace, a.path)
    if not root.exists():
        raise ToolError(f"{a.path} does not exist")
    items = sorted(p.relative_to(ctx.workspace).as_posix() + ("/" if p.is_dir() else "")
                   for p in root.rglob("*") if not p.name.startswith(".agent"))
    return "\n".join(items[:500]) or "(empty)"


def _read(ctx: ToolContext, a: PathArgs) -> str:
    p = safe_path(ctx.workspace, a.path)
    if not p.is_file():
        raise ToolError(f"{a.path} is not a file")
    text = p.read_text(errors="ignore")
    return text[:30000] + (" ...[truncated]" if len(text) > 30000 else "")


def _write(ctx: ToolContext, a: WriteArgs) -> str:
    p = safe_path(ctx.workspace, a.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if a.append else "w") as f:
        f.write(a.content)
    return f"wrote {len(a.content)} chars to {a.path}"


def _delete(ctx: ToolContext, a: PathArgs) -> str:
    p = safe_path(ctx.workspace, a.path)
    if p == ctx.workspace.resolve():
        raise PermissionError("blocked: refusing to delete the workspace root")
    if not p.is_file():
        raise ToolError(f"{a.path} is not a file")
    p.unlink()
    return f"deleted {a.path}"


def _run(ctx: ToolContext, a: CodeArgs) -> str:
    if ctx.sandbox is None:
        raise PermissionError("blocked: no sandbox available (start Docker or set E2B_API_KEY); code was not run")
    return ctx.sandbox.run_python(a.code).summary()


def _finish(ctx: ToolContext, a: FinishArgs) -> str:
    return a.answer


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("web_search", "Search the web. Returns titles, URLs and snippets.", SearchArgs, _search),
    Tool("fetch_url", "Download a web page and return its text.", FetchArgs, _fetch),
    Tool("list_files", "List files in the workspace.", PathArgs, _list),
    Tool("read_file", "Read a text file from the workspace.", PathArgs, _read),
    Tool("write_file", "Write (or append to) a text file in the workspace.", WriteArgs, _write),
    Tool("delete_file", "Delete a file from the workspace. Needs human approval.", PathArgs, _delete, risk="risky"),
    Tool("run_python", "Run Python code in an isolated sandbox (no network). The workspace is the working "
                       "directory. Print what you want to see.", CodeArgs, _run, risk="exec"),
    Tool("finish", "Call when the task is complete and checked. Give the final answer and result files.",
         FinishArgs, _finish),
]}


def execute(ctx: ToolContext, name: str, raw_args: str, allowed: set[str]) -> tuple[bool, str]:
    """Validate and run one tool call. Returns (ok, output). Never raises: errors go back to the model."""
    if name not in allowed or name not in TOOLS:
        ctx.tracer.log(ctx.agent, "blocked", tool=name, reason="tool not allowed for this agent")
        return False, f"Error: tool '{name}' is not available. Allowed: {sorted(allowed)}"
    tool = TOOLS[name]
    try:
        args = tool.args.model_validate(json.loads(raw_args or "{}"))
    except (json.JSONDecodeError, ValidationError) as e:
        return False, f"Error: invalid arguments for {name}: {e}. Fix them and try again."

    needs_approval = tool.risk == "risky" or (
        tool.risk == "exec" and getattr(ctx.sandbox, "name", "") == "local-unsafe"
    )
    if needs_approval and not ctx.approve(name, args.model_dump()):
        ctx.tracer.log(ctx.agent, "denied", tool=name, reason="human declined")
        return False, "Error: a human declined this action. Find another way or finish without it."

    try:
        return True, tool.fn(ctx, args)
    except PermissionError as e:
        ctx.tracer.log(ctx.agent, "blocked", tool=name, reason=str(e))
        return False, f"Error: {e}"
    except Exception as e:  # tool failures are information for the model, not crashes
        return False, f"Error: {type(e).__name__}: {e}"
