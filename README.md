# 02 · Agents for long-horizon tasks

An agent that finishes a multi-step task **on its own**, with **sub-agents running in parallel** inside a **sandbox**, behind **guardrails**, with **every step logged**.

```
$ agent run "research and write a report"
loop 01  plan → act → check
├─ sub-agent 1  ██████  searching
├─ sub-agent 2  ██████  reading
└─ sub-agent 3  ██████  writing
sandbox  blocked an unsafe action ✓
tests    all tasks passed ✓
```

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env            # OpenRouter key (or point at your vLLM server from 03-deploy)
docker info                     # Docker running = code runs in a throwaway container (or set E2B_API_KEY)

agent run "Compare Chroma, Qdrant and pgvector for a small RAG app and write report.md"
agent run --single "Write fizzbuzz.py plus tests and run them until they pass"
agent eval                      # task test set -> pass rate
pytest                          # 13 offline tests, no keys needed
```

## How it works

```
                 ┌───────────── Orchestrator ─────────────┐
 task ──► plan (JSON, Pydantic-validated, retried if invalid)
                 │                │                │
           sub-agent 1      sub-agent 2      sub-agent 3      ← ThreadPool, shared budget
           researcher       reader           coder
           notes/s1.md      notes/s2.md      notes/s3.md
                 └────────────────┼────────────────┘
                               writer  ──► report.md ──► checks ✓
```

Every agent runs the same loop (`agents/agent.py`):

1. **Plan.** A numbered plan before any tool call.
2. **Act.** One tool call at a time. Tool errors go back to the model as information, not crashes.
3. **Check.** `finish` triggers automatic checks: listed files exist and aren't empty, plus any task-specific checks. A failed check sends the agent back to work, up to 3 times.

Long runs stay inside the context window: old tool outputs are trimmed and the last 6 are kept in full.

## Guardrails

| risk | what stops it |
|---|---|
| model sends malformed tool args | every call is validated against a **Pydantic** schema; the error goes back to the model |
| model calls a tool it shouldn't | per-role **tool allowlist** (a writer can't run code, a coder can't browse) |
| path escape (`../../etc/passwd`) | all file tools resolve inside the **workspace**, so escapes get blocked and logged |
| destructive action (`delete_file`) | marked **risky**, so a **human approves** (`--approve ask/auto/deny`) |
| running agent-written code | only inside a **sandbox**: E2B microVM, or Docker with `--network none`, 512 MB RAM, 1 CPU, read-only root. With no sandbox, code doesn't run |
| runaway loop / bill | shared **budget**: max steps, max tokens, max $ (thread-safe across sub-agents) |
| silent failure | **JSONL trace** of every step: agent, tool, input, output, tokens, running cost |

## Task test set

`tasks/tasks.yaml` holds 5 tasks with independent checks, run fresh in their own workspace by `agent eval`:

| task | mode | checked by |
|---|---|---|
| research-report | team | report exists, ≥300 words, cites URLs |
| code-with-tests | single | sandbox imports `fizzbuzz` and asserts outputs |
| data-analysis | single | `answer.txt` has `total=823`, `best=May` |
| summarise-notes | single | summary names owners and dates from 3 notes |
| refuse-unsafe | single | `keep.txt` untouched, nothing written outside the workspace |

Swap `LLM_MODEL` and rerun `agent eval` to compare models on pass rate, steps and $ per task (results go to `eval_runs/results.json`).

The offline `pytest` suite proves the "done when" criteria with a scripted model:
- **10+ steps without help:** `test_runs_more_than_10_steps_without_help` (16 steps, all logged)
- **sandboxed and logged:** `test_code_never_runs_without_a_sandbox`, `test_writing_outside_workspace_is_blocked`
- **parallel sub-agents:** `test_orchestrator_runs_sub_agents_in_parallel` (3 agents, wall time < 2.5× one agent)
- plus budget stops, approval denial, bad args, failed-check retry, plan retry, and the real client's wire format

## Toolkit choices

- **Framework.** Hand-written loop on purpose (that's the skill: loop engineering). Pydantic schemas for tool inputs. Pydantic AI, smolagents or LangGraph are good next steps once you know what they abstract.
- **Models.** OpenRouter: one API, many models, so cost and quality are easy to compare.
- **Sandbox.** E2B (isolated microVM) → Docker (throwaway container). Local execution is opt-in for CI only.
- **Guardrails.** Allowlisted tools, Pydantic validation, step/token/$ caps, human approval for risky actions, JSONL logs.

## Layout

```
agents/agent.py         plan → act → check loop, context trimming, finish checks
agents/orchestrator.py  planner + parallel sub-agents + writer
agents/tools.py         Pydantic-validated tools, risk levels, approval gate
agents/sandbox.py       E2B / Docker / opt-in local
agents/guardrails.py    budget, workspace path jail, approval policies
agents/tracing.py       JSONL step log
agents/evaluate.py      task-set runner
tasks/tasks.yaml        the test set
```

## What I learned

loop engineering · sub-agents · sandboxing · guardrails · monitoring · testing

Start here: Hugging Face AI Agents Course (free), huggingface.co/learn/agents-course
