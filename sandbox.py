"""The sandbox: the only place model-written commands run, and the only directory agents can touch."""

import os
import subprocess
import uuid
from pathlib import Path

IMAGE = "python:3.12-slim"
MAX_OUTPUT = 8_000


class Sandbox:
    """One throwaway container per run, with the workspace directory as its only writable mount."""

    def __init__(self, workspace: Path, image: str = IMAGE):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.name = f"agent-sandbox-{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> "Sandbox":
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
        subprocess.run([
            "docker", "run", "--detach", "--rm", "--name", self.name,
            "--network", "none",                    # no exfiltration, no downloads
            "--read-only", "--tmpfs", "/tmp:rw,size=64m",  # root filesystem is immutable
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", user,                         # never root, even inside the container
            "--memory", "512m", "--cpus", "1", "--pids-limit", "128",  # fork bombs and memory hogs die here
            "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--volume", f"{self.workspace}:/workspace", "--workdir", "/workspace",
            self.image, "sleep", "infinity",
        ], check=True, capture_output=True, text=True)
        return self

    def __exit__(self, *exc) -> None:
        subprocess.run(["docker", "rm", "--force", self.name], capture_output=True)

    def run(self, command: str, timeout: int = 60) -> str:
        """Run a shell command inside the container. `timeout` kills it in the container, not just the client."""
        try:
            done = subprocess.run(
                ["docker", "exec", self.name, "timeout", "--signal=KILL", str(timeout), "sh", "-c", command],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout + 15)
        except subprocess.TimeoutExpired:
            return f"exit code 137\n(no response from the sandbox after {timeout + 15}s)"
        output = (done.stdout + done.stderr).strip()
        if len(output) > MAX_OUTPUT:  # keep head and tail: errors are usually at the end
            output = f"{output[:MAX_OUTPUT // 2]}\n... [{len(output) - MAX_OUTPUT} characters cut] ...\n{output[-MAX_OUTPUT // 2:]}"
        note = " (killed: time limit)" if done.returncode == 137 else ""
        return f"exit code {done.returncode}{note}\n{output}"

    def resolve(self, path: str) -> Path:
        """Map a model-supplied path into the workspace, refusing anything that escapes it."""
        target = (self.workspace / path).resolve()  # resolve() also follows symlinks planted from inside
        if not target.is_relative_to(self.workspace):
            raise PermissionError(f"{path!r} is outside the workspace")
        return target

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return f"wrote {len(content)} characters to {path}"

    def read_file(self, path: str) -> str:
        return self.resolve(path).read_text(encoding="utf-8")[:MAX_OUTPUT * 4]

    def list_files(self) -> str:
        files = sorted(p.relative_to(self.workspace).as_posix() for p in self.workspace.rglob("*") if p.is_file())
        return "\n".join(files) or "(workspace is empty)"
