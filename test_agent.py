"""The loop, the approval gate and the sandbox, tested without calling the API.

The Docker tests run against a real container and are skipped when no daemon is available.
"""

import itertools
import subprocess
from types import SimpleNamespace

import pytest

import agent
from agent import AGENTS, Runtime
from sandbox import Sandbox

ids = itertools.count()
NO_USAGE = SimpleNamespace(input_tokens=0, output_tokens=0)


def say(text):
    return SimpleNamespace(stop_reason="end_turn", usage=NO_USAGE, content=[SimpleNamespace(type="text", text=text)])


def call(name, **args):
    block = SimpleNamespace(type="tool_use", id=f"toolu_{next(ids)}", name=name, input=args)
    return SimpleNamespace(stop_reason="tool_use", usage=NO_USAGE, content=[block])


class FakeClient:
    """Plays back scripted model turns and records every request."""

    def __init__(self, *turns):
        self.turns, self.requests = iter(turns), []
        self.beta = SimpleNamespace(messages=self)

    def create(self, **request):
        self.requests.append(request)
        return next(self.turns)


class FakeSandbox(Sandbox):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.commands = []

    def run(self, command, timeout=60):
        self.commands.append(command)
        return "exit code 0\nok"


def last_result(client):
    """The newest tool_result in the last conversation (the message list keeps growing after each request)."""
    results = [m["content"] for m in client.requests[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    return results[-1][0]


def test_orchestrator_delegates_and_gets_the_report_back(tmp_path):
    client = FakeClient(
        call("delegate", agent="coder", task="write hello.py and run it"),  # orchestrator
        call("write_file", path="hello.py", content="print('hi')"),         # coder
        call("run_command", command="python hello.py"),                     # coder
        say("hello.py works"),                                              # coder
        say("Built and verified hello.py"),                                 # orchestrator
    )
    sandbox = FakeSandbox(tmp_path)

    report = Runtime(client, sandbox).run("orchestrator", "say hi")

    assert report == "Built and verified hello.py"
    assert (tmp_path / "hello.py").read_text() == "print('hi')" and sandbox.commands == ["python hello.py"]
    assert last_result(client)["content"] == "hello.py works"  # the coder's report is the delegate tool's result
    assert client.requests[1]["messages"][0] == {"role": "user", "content": "write hello.py and run it"}
    assert [t["name"] for t in client.requests[1]["tools"]] == list(AGENTS["coder"].tools)


def test_human_can_reject_with_a_reason(tmp_path):
    client = FakeClient(call("run_command", command="rm -rf ."), say("ok, I will not"))
    sandbox = FakeSandbox(tmp_path)

    Runtime(client, sandbox, human=lambda prompt: "n never delete the workspace").run("coder", "clean up")

    assert sandbox.commands == []
    result = last_result(client)
    assert result["is_error"] and "never delete the workspace" in result["content"]


def test_human_can_rewrite_a_command(tmp_path):
    client = FakeClient(call("run_command", command="rm -rf build"), say("done"))
    sandbox = FakeSandbox(tmp_path)

    Runtime(client, sandbox, human=lambda prompt: "e rm -r build/tmp").run("coder", "clean up")

    assert sandbox.commands == ["rm -r build/tmp"]
    assert "replaced your command with: rm -r build/tmp" in last_result(client)["content"]


@pytest.mark.parametrize("command,asks", [
    ("python -m unittest", False), ("ls -la", False),
    ("rm -rf .", True), ("pip install requests", True), ("curl http://example.com", True),
    ("python x.py; rm -rf .", True), ("python -c 'import os' && rm x", True), ("cat x > y", True),
    ("ls $(rm x)", True), ("", True),
])
def test_only_routine_commands_skip_approval(command, asks):
    assert agent.needs_approval("run_command", {"command": command}) is asks


def test_files_cannot_escape_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    client = FakeClient(call("write_file", path="../escaped.txt", content="x"), say("ok"))

    Runtime(client, FakeSandbox(workspace)).run("coder", "try to escape")

    assert not (tmp_path / "escaped.txt").exists()
    assert last_result(client)["is_error"] and "outside the workspace" in last_result(client)["content"]


def test_reviewer_cannot_write_even_if_the_model_tries(tmp_path):
    client = FakeClient(call("write_file", path="x.py", content="x"), say("ok"))

    Runtime(client, FakeSandbox(tmp_path)).run("reviewer", "review")

    assert not (tmp_path / "x.py").exists() and last_result(client)["is_error"]


def test_loops_are_bounded_per_agent_and_per_run(tmp_path, monkeypatch):
    forever = (call("list_files") for _ in itertools.count())
    assert "turn limit" in Runtime(FakeClient(*itertools.islice(forever, 50)), FakeSandbox(tmp_path)).run("coder", "x")

    monkeypatch.setattr(agent, "MAX_LLM_CALLS", 3)
    client = FakeClient(*itertools.islice(forever, 50))
    assert "budget is spent" in Runtime(client, FakeSandbox(tmp_path)).run("coder", "x")
    assert len(client.requests) == 3


def test_a_truncated_turn_never_runs_its_tool_call(tmp_path):
    cut_off = call("run_command", command="python half-written")
    cut_off.stop_reason = "max_tokens"
    sandbox = FakeSandbox(tmp_path)

    assert Runtime(FakeClient(cut_off), sandbox).run("coder", "x").startswith("[stopped: max_tokens]")
    assert sandbox.commands == []


docker = pytest.mark.skipif(
    subprocess.run(["docker", "info"], capture_output=True).returncode != 0, reason="Docker daemon not running")


@docker
def test_real_sandbox_is_isolated(tmp_path):
    with Sandbox(tmp_path) as sandbox:
        sandbox.write_file("probe.py", "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=5)")
        assert "exit code 0" not in sandbox.run("python probe.py")              # no network
        assert "Read-only file system" in sandbox.run("touch /usr/local/x")     # immutable root filesystem
        assert sandbox.run("id -u").split()[-1] != "0"                          # not root
        assert "exit code 0" in sandbox.run("echo hi > out.txt")                # the workspace is writable...
        assert (tmp_path / "out.txt").read_text().strip() == "hi"               # ...and shared with the host
        assert "exit code 137" in sandbox.run("sleep 30", timeout=2)            # runaway commands are killed
    listing = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True)
    assert sandbox.name not in listing.stdout                                   # nothing left behind
