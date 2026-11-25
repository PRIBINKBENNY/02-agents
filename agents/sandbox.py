"""Where agent-written code runs. Never on your machine directly.

- E2BSandbox:    isolated cloud microVM (set E2B_API_KEY, `pip install e2b-code-interpreter`)
- DockerSandbox: throwaway container, no network, capped memory/CPU, only the workspace mounted
- LocalSandbox:  plain subprocess with a timeout. NOT isolated. Opt-in only (AGENT_ALLOW_LOCAL_EXEC=1),
                 meant for CI and tests.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    sandbox: str

    def summary(self, limit: int = 4000) -> str:
        out = f"exit_code={self.exit_code}\n--- stdout ---\n{self.stdout[-limit:]}"
        if self.stderr.strip():
            out += f"\n--- stderr ---\n{self.stderr[-limit:]}"
        return out


class SandboxUnavailable(RuntimeError):
    pass


class DockerSandbox:
    name = "docker"

    def __init__(self, workspace: Path, image: str = "python:3.12-slim", timeout: int = 60):
        if not shutil.which("docker"):
            raise SandboxUnavailable("docker not installed")
        if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            raise SandboxUnavailable("docker daemon not running")
        self.workspace, self.image, self.timeout = workspace.resolve(), image, timeout

    def run_python(self, code: str) -> ExecResult:
        script = self.workspace / ".agent_run.py"
        script.write_text(code)
        cmd = [
            "docker", "run", "--rm", "--network", "none", "--memory", "512m", "--cpus", "1",
            "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp",
            "-v", f"{self.workspace}:/work", "-w", "/work", self.image, "python", ".agent_run.py",
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            return ExecResult(p.stdout, p.stderr, p.returncode, self.name)
        except subprocess.TimeoutExpired:
            return ExecResult("", f"timed out after {self.timeout}s", 124, self.name)
        finally:
            script.unlink(missing_ok=True)


class E2BSandbox:
    name = "e2b"

    def __init__(self, workspace: Path, timeout: int = 60):
        if not os.environ.get("E2B_API_KEY"):
            raise SandboxUnavailable("E2B_API_KEY not set")
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as e:
            raise SandboxUnavailable("pip install e2b-code-interpreter") from e
        self._sbx = Sandbox(timeout=timeout * 10)
        self.workspace = workspace

    def run_python(self, code: str) -> ExecResult:
        # sync workspace files up so code can read what the agent wrote
        for f in self.workspace.rglob("*"):
            if f.is_file() and f.stat().st_size < 5_000_000:
                self._sbx.files.write(f"/home/user/{f.relative_to(self.workspace)}", f.read_bytes())
        ex = self._sbx.run_code(code)
        stdout = "".join(ex.logs.stdout)
        stderr = "".join(ex.logs.stderr) + (f"\n{ex.error.name}: {ex.error.value}" if ex.error else "")
        return ExecResult(stdout, stderr, 1 if ex.error else 0, self.name)


class LocalSandbox:
    name = "local-unsafe"

    def __init__(self, workspace: Path, timeout: int = 30):
        if os.environ.get("AGENT_ALLOW_LOCAL_EXEC") != "1":
            raise SandboxUnavailable("local execution disabled (set AGENT_ALLOW_LOCAL_EXEC=1 to allow)")
        self.workspace, self.timeout = workspace.resolve(), timeout

    def run_python(self, code: str) -> ExecResult:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.workspace,
                               capture_output=True, text=True, timeout=self.timeout,
                               env={"PATH": os.environ.get("PATH", ""), "HOME": str(self.workspace)})
            return ExecResult(p.stdout, p.stderr, p.returncode, self.name)
        except subprocess.TimeoutExpired:
            return ExecResult("", f"timed out after {self.timeout}s", 124, self.name)


def get_sandbox(workspace: Path, prefer: str = "auto"):
    """Pick the strongest isolation available: e2b > docker > (opt-in) local."""
    order = {"auto": ["e2b", "docker", "local"], "e2b": ["e2b"], "docker": ["docker"], "local": ["local"]}[prefer]
    errors = []
    for kind in order:
        try:
            return {"e2b": E2BSandbox, "docker": DockerSandbox, "local": LocalSandbox}[kind](workspace)
        except SandboxUnavailable as e:
            errors.append(f"{kind}: {e}")
    return None if prefer == "auto" else _raise(errors)


def _raise(errors):
    raise SandboxUnavailable("; ".join(errors))
