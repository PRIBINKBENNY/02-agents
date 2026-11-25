"""Offline tests with a scripted model: no API keys, no network, deterministic."""
import json
import re
import time
from pathlib import Path

import pytest

from agents import Agent, Budget, Orchestrator, approval_policy
from agents.llm import LLMResponse, ScriptedLLM, call
from agents.sandbox import LocalSandbox
from agents.tools import ToolContext
from agents.tracing import Tracer


def turn(messages):
    return sum(1 for m in messages if m["role"] == "assistant")


def last_tool_output(messages):
    return next((m["content"] for m in reversed(messages) if m["role"] == "tool"), "")


def make_agent(tmp_path, script, approve="deny", sandbox=None, budget=None, **kw):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(tmp_path / "log.jsonl", echo=False)
    ctx = ToolContext(ws, sandbox, approval_policy(approve), tracer)
    agent = Agent("agent", ScriptedLLM(script), ctx, budget or Budget(), **kw)
    return agent, ws, tracer


def events(tracer, kind):
    return [e for e in tracer.events if e["kind"] == kind]


# ---------------------------------------------------------------- long horizon
def test_runs_more_than_10_steps_without_help(tmp_path):
    def script(messages, tools):
        if tools is None:
            return LLMResponse("1. write 12 section files\n2. merge\n3. verify\n4. finish")
        t = turn(messages) - 1  # first assistant turn is the plan
        if t <= 12:
            return LLMResponse(tool_calls=[call("write_file", path=f"parts/{t:02}.md", content=f"section {t}\n")])
        if t == 13:
            return LLMResponse(tool_calls=[call("list_files", path="parts")])
        if t == 14:
            return LLMResponse(tool_calls=[call("write_file", path="report.md",
                                                content="".join(f"section {i}\n" for i in range(1, 13)))])
        if t == 15:
            return LLMResponse(tool_calls=[call("read_file", path="report.md")])
        return LLMResponse(tool_calls=[call("finish", answer="report written", files=["report.md"])])

    agent, ws, tracer = make_agent(tmp_path, script)
    res = agent.run("write a 12-section report")
    assert res.ok, res.reason
    assert res.steps >= 10
    assert (ws / "report.md").read_text().count("section") == 12
    # every step is logged with tool, input, output and tokens
    calls, results = events(tracer, "tool_call"), events(tracer, "tool_result")
    assert len(calls) == res.steps and len(results) == res.steps - 1  # finish has no tool_result
    assert all({"tool", "args", "tokens", "budget"} <= set(e) for e in calls)
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == len(tracer.events) and json.loads(lines[0])["kind"] == "start"


# ---------------------------------------------------------------- guardrails
def test_writing_outside_workspace_is_blocked(tmp_path):
    def script(messages, tools):
        if tools is None:
            return LLMResponse("plan")
        if turn(messages) == 1:
            return LLMResponse(tool_calls=[call("write_file", path="../../evil.txt", content="x")])
        assert "outside the agent workspace" in last_tool_output(messages)
        return LLMResponse(tool_calls=[call("finish", answer="could not write there")])

    agent, ws, tracer = make_agent(tmp_path, script)
    assert agent.run("write evil").ok
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()
    assert events(tracer, "blocked")


def test_risky_tool_needs_human_approval(tmp_path):
    def script(messages, tools):
        if tools is None:
            return LLMResponse("plan")
        if turn(messages) == 1:
            return LLMResponse(tool_calls=[call("delete_file", path="keep.txt")])
        assert "declined" in last_tool_output(messages)
        return LLMResponse(tool_calls=[call("finish", answer="left the file alone")])

    agent, ws, tracer = make_agent(tmp_path, script, approve="deny")
    (ws / "keep.txt").write_text("important")
    assert agent.run("delete keep.txt").ok
    assert (ws / "keep.txt").read_text() == "important"
    assert events(tracer, "denied")

    agent2, ws2, _ = make_agent(tmp_path / "b", script=lambda m, t: (
        LLMResponse("plan") if t is None else
        LLMResponse(tool_calls=[call("delete_file", path="keep.txt")]) if turn(m) == 1 else
        LLMResponse(tool_calls=[call("finish", answer="deleted")])), approve="auto")
    (ws2 / "keep.txt").write_text("x")
    assert agent2.run("delete").ok and not (ws2 / "keep.txt").exists()


def test_invalid_tool_args_are_rejected_and_fed_back(tmp_path):
    seen = {}

    def script(messages, tools):
        if tools is None:
            return LLMResponse("plan")
        if turn(messages) == 1:
            return LLMResponse(tool_calls=[call("fetch_url", url="file:///etc/passwd")])
        if turn(messages) == 2:
            seen["err"] = last_tool_output(messages)
            return LLMResponse(tool_calls=[call("not_a_tool")])
        seen["err2"] = last_tool_output(messages)
        return LLMResponse(tool_calls=[call("finish", answer="done")])

    agent, _, _ = make_agent(tmp_path, script)
    assert agent.run("x").ok
    assert "invalid arguments" in seen["err"] and "not available" in seen["err2"]


def test_budget_stops_runaway_loop(tmp_path):
    def script(messages, tools):
        if tools is None:
            return LLMResponse("plan", input_tokens=10, output_tokens=5)
        return LLMResponse(tool_calls=[call("list_files")], input_tokens=1000, output_tokens=200)

    agent, _, tracer = make_agent(tmp_path, script, budget=Budget(max_steps=8))
    res = agent.run("loop forever")
    assert not res.ok and "max steps" in res.reason
    assert events(tracer, "budget_stop")

    agent, _, _ = make_agent(tmp_path / "c", script, budget=Budget(max_cost_usd=0.001, price_in=1.0, price_out=1.0))
    res = agent.run("loop forever")
    assert not res.ok and "cost budget" in res.reason


def test_failed_checks_send_agent_back_to_work(tmp_path):
    def script(messages, tools):
        if tools is None:
            return LLMResponse("plan")
        t = turn(messages)
        if t == 1:
            return LLMResponse(tool_calls=[call("finish", answer="done", files=["out.txt"])])  # lies
        if t == 2:
            assert "Checks failed" in last_tool_output(messages)
            return LLMResponse(tool_calls=[call("write_file", path="out.txt", content="42")])
        return LLMResponse(tool_calls=[call("finish", answer="done", files=["out.txt"])])

    agent, ws, tracer = make_agent(tmp_path, script,
                                   checks=[lambda w: None if (w / "out.txt").exists() and
                                           (w / "out.txt").read_text() == "42" else "out.txt must contain 42"])
    res = agent.run("write 42")
    assert res.ok and [e["ok"] for e in events(tracer, "check")] == [False, True]


def test_code_never_runs_without_a_sandbox(tmp_path):
    def script(messages, tools):
        if tools is None:
            return LLMResponse("plan")
        if turn(messages) == 1:
            return LLMResponse(tool_calls=[call("run_python", code="open('pwned','w').write('x')")])
        assert "no sandbox" in last_tool_output(messages)
        return LLMResponse(tool_calls=[call("finish", answer="no sandbox")])

    agent, ws, _ = make_agent(tmp_path, script, sandbox=None)
    assert agent.run("run code").ok and not (ws / "pwned").exists()


def test_local_sandbox_runs_code_with_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ALLOW_LOCAL_EXEC", "1")
    out = {}

    def script(messages, tools):
        if tools is None:
            return LLMResponse("plan")
        if turn(messages) == 1:
            return LLMResponse(tool_calls=[call("run_python", code="print(sum(range(10)))")])
        out["r"] = last_tool_output(messages)
        return LLMResponse(tool_calls=[call("finish", answer=out["r"])])

    ws = tmp_path / "ws"
    ws.mkdir()
    agent, _, _ = make_agent(tmp_path, script, approve="auto", sandbox=LocalSandbox(ws))
    assert agent.run("sum").ok and "45" in out["r"] and "exit_code=0" in out["r"]


# ---------------------------------------------------------------- orchestrator
def test_orchestrator_runs_sub_agents_in_parallel(tmp_path):
    SLEEP = 0.4

    def script(messages, tools):
        system = messages[0]["content"]
        if "orchestrator of a team" in system:
            return LLMResponse(json.dumps({"subtasks": [
                {"id": f"s{i}", "role": "researcher", "instructions": f"research angle number {i}",
                 "output_file": f"notes/s{i}.md"} for i in (1, 2, 3)], "final_file": "report.md"}))
        name = re.search(r"You are ([\w -]+?), an", system).group(1)
        if tools is None:
            return LLMResponse(f"plan for {name}")
        t = turn(messages)
        if name.startswith("sub-agent"):
            i = name.split()[-1]
            if t == 1:
                time.sleep(SLEEP)  # pretend to be a slow model / web search
                return LLMResponse(tool_calls=[call("write_file", path=f"notes/s{i}.md", content=f"finding {i} https://example.com/{i}")])
            return LLMResponse(tool_calls=[call("finish", answer="ok", files=[f"notes/s{i}.md"])])
        # writer
        if t <= 3:
            return LLMResponse(tool_calls=[call("read_file", path=f"notes/s{t}.md")])
        if t == 4:
            body = "\n".join(m["content"] for m in messages if m["role"] == "tool")
            return LLMResponse(tool_calls=[call("write_file", path="report.md", content="# Report\n" + body)])
        return LLMResponse(tool_calls=[call("finish", answer="report done", files=["report.md"])])

    orch = Orchestrator(ScriptedLLM(script), tmp_path / "ws", Budget(), approval_policy("deny"),
                        sandbox="auto", log_dir=tmp_path / "runs", echo=False)
    t0 = time.perf_counter()
    res = orch.run("research and write a report",
                   checks=[lambda w: None if "finding 3" in (w / "report.md").read_text() else "missing finding 3"])
    elapsed = time.perf_counter() - t0
    assert res.ok, res.reason
    assert len(res.sub_results) == 3 and all(r.ok for r in res.sub_results)
    assert elapsed < SLEEP * 2.5, f"sub-agents did not run in parallel ({elapsed:.2f}s)"
    report = (tmp_path / "ws" / "report.md").read_text()
    assert all(f"finding {i}" in report for i in (1, 2, 3))
    agents_in_log = {json.loads(l)["agent"] for l in Path(res.log_file).read_text().splitlines()}
    assert {"orchestrator", "sub-agent 1", "sub-agent 2", "sub-agent 3", "writer"} <= agents_in_log


def test_orchestrator_retries_invalid_plan(tmp_path):
    attempts = []

    def script(messages, tools):
        if "orchestrator of a team" in messages[0]["content"]:
            attempts.append(1)
            if len(attempts) == 1:
                return LLMResponse('{"subtasks": [{"id": "bad id!", "role": "wizard"}]}')
            return LLMResponse('{"subtasks": [{"id": "s1", "role": "writer", "instructions": "write a haiku file",'
                               ' "output_file": "notes/s1.md"}], "final_file": "out.md"}')
        if tools is None:
            return LLMResponse("plan")
        t = turn(messages)
        name = re.search(r"You are ([\w -]+?), an", messages[0]["content"]).group(1)
        target = "notes/s1.md" if name.startswith("sub") else "out.md"
        if t == 1:
            return LLMResponse(tool_calls=[call("write_file", path=target, content="text")])
        return LLMResponse(tool_calls=[call("finish", answer="ok", files=[target])])

    orch = Orchestrator(ScriptedLLM(script), tmp_path / "ws", Budget(), approval_policy("deny"),
                        log_dir=tmp_path / "runs", echo=False)
    res = orch.run("make out.md")
    assert res.ok and len(attempts) == 2 and res.plan.subtasks[0].role == "writer"


def test_eval_checks(tmp_path):
    from agents.evaluate import build_checks

    (tmp_path / "a.md").write_text("Meera owns the roadmap, launch March 3")
    checks = build_checks([{"file_exists": "a.md"}, {"contains": {"path": "a.md", "all": ["meera", "march 3"]}},
                           {"min_words": {"path": "a.md", "n": 50}}, {"file_absent": "zzz"}], sandbox=None)
    results = [c(tmp_path) for c in checks]
    assert results[0] is None and results[1] is None and results[3] is None
    assert "needs 50" in results[2]


def test_task_set_is_valid():
    import yaml

    tasks = yaml.safe_load((Path(__file__).parents[1] / "tasks" / "tasks.yaml").read_text())["tasks"]
    assert len(tasks) >= 5
    assert {t["id"] for t in tasks} >= {"research-report", "code-with-tests", "refuse-unsafe"}


def test_openai_compatible_client_parses_tool_calls():
    """The real client against a mocked HTTP endpoint (same wire format as OpenRouter / vLLM)."""
    import httpx

    from agents.llm import OpenAICompatLLM
    from agents.tools import TOOLS

    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "write_file", "arguments": '{"path": "a.md", "content": "hi"}'}}]}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}})

    llm = OpenAICompatLLM(model="m", base_url="http://test/v1", api_key="k",
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    resp = llm.complete([{"role": "user", "content": "hi"}], [TOOLS["write_file"].schema()])
    assert resp.tool_calls[0].name == "write_file" and json.loads(resp.tool_calls[0].arguments)["path"] == "a.md"
    assert (resp.input_tokens, resp.output_tokens) == (120, 30)
    assert seen["body"]["tools"][0]["function"]["parameters"]["required"] == ["path", "content"]
