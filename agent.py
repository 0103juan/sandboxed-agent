"""An orchestrator that delegates to specialist agents, pauses for a human, and only acts inside a sandbox.

    python agent.py "Build a CLI that converts CSV to JSON, with tests"

Every agent is the same bounded loop (Runtime.run) with a different prompt and tool set.
Delegation is just a tool whose implementation is that loop again, with a fresh context.
"""

import argparse
import json
import re
import shlex
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anthropic

from sandbox import Sandbox

MODEL = "claude-sonnet-5-5"
MAX_TURNS = 15  # per agent invocation
MAX_LLM_CALLS = 60  # for the whole run, across every agent
USD_PER_MTOK = {"input_tokens": 2.00, "output_tokens": 10.00}  # claude-sonnet-5-5; thinking bills as output
AUTO_APPROVED = {"python", "python3", "ls", "cat", "head", "tail", "wc", "grep", "pwd"}
SHELL_SYNTAX = set(";&|<>`$(){}\n\\")


def _tool(name: str, description: str, **properties: dict) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": properties,
                             "required": list(properties), "additionalProperties": False}}


def _text(description: str) -> dict:
    return {"type": "string", "description": description}


TOOLS = {
    "list_files": _tool("list_files", "List every file in the workspace. Call this to see what already exists."),
    "read_file": _tool("read_file", "Read a text file from the workspace.",
                       path=_text("Path relative to the workspace root")),
    "write_file": _tool("write_file", "Create or overwrite a text file in the workspace.",
                        path=_text("Path relative to the workspace root"), content=_text("Full file contents")),
    "run_command": _tool(
        "run_command",
        "Run a shell command in the sandbox, with the workspace as the working directory. Returns the exit "
        "code and combined output. Commands other than plain python/ls/cat/grep calls wait for human approval.",
        command=_text("The command line to run")),
    "delegate": _tool(
        "delegate",
        "Hand a task to a specialist agent and get its final report back. The agent starts with no knowledge "
        "of this conversation, so the task must be self-contained. The human approves each delegation.",
        agent={"type": "string", "enum": ["coder", "reviewer"]},
        task=_text("What to do, which files are involved, and what counts as done")),
    "ask_human": _tool(
        "ask_human",
        "Ask the human a question. Use this only when the request is ambiguous in a way that changes what "
        "gets built; otherwise make a reasonable choice and say which one you made.",
        question=_text("The question, with the options you are choosing between")),
}

SANDBOX_FACTS = ("The workspace is a sandbox with Python 3.12, no network access and no package installs: "
                 "use the standard library only, and unittest for tests.")


@dataclass(frozen=True)
class Agent:
    name: str
    tools: tuple[str, ...]
    system: str


AGENTS = {agent.name: agent for agent in [
    Agent("orchestrator", ("delegate", "ask_human", "list_files", "read_file"), f"""\
You lead a small software team working in a shared workspace. {SANDBOX_FACTS}

Break the user's request into steps and delegate them: `coder` writes and runs code, `reviewer` \
independently verifies it. Each delegated agent starts from a blank context, so write tasks that stand \
on their own. Have the reviewer check the coder's work before you finish, and if the reviewer reports \
defects, send the coder back with those specific findings. If the human rejects a delegation, their \
reason is the new requirement: revise the plan to fit it.

Finish with a short report: what was built, where it is, and how it was verified."""),
    Agent("coder", ("list_files", "read_file", "write_file", "run_command"), f"""\
You are a software engineer. {SANDBOX_FACTS}

Write files, run them, and iterate until they work. Some commands wait for human approval; if one is \
denied, the reason tells you what to do differently, so adapt instead of retrying it.

Finish with a short report: the files you created, how you ran them, and the output showing they work."""),
    Agent("reviewer", ("list_files", "read_file", "run_command"), f"""\
You are a code reviewer. You can read files and run commands, but you do not change the code. {SANDBOX_FACTS}

Verify the work described in your task: read the code, run it and its tests yourself, and try the edge \
cases the author may have missed. Report either APPROVED, or a list of specific defects, each with the \
command you ran and what it printed."""),
]}


def needs_approval(tool: str, args: dict) -> bool:
    """Delegations are plan checkpoints; commands are checked unless they are obviously routine."""
    if tool == "delegate":
        return True
    if tool != "run_command":
        return False
    command = args["command"]
    if SHELL_SYNTAX & set(command):
        return True
    try:
        return shlex.split(command)[0] not in AUTO_APPROVED
    except (ValueError, IndexError):
        return True


class Denied(Exception):
    pass


class Runtime:
    """Shared state for one run: the sandbox, the human, the call budget and the trace."""

    def __init__(self, client, sandbox: Sandbox, human: Callable[[str], str] | None = None,
                 trace: Path | None = None):
        self.client, self.sandbox, self.human, self.trace = client, sandbox, human, trace
        self.calls_left = MAX_LLM_CALLS
        self.usage = Counter()

    def log(self, agent: str, event: str, **data) -> None:
        print(f"[{agent}] {event}: {json.dumps(data, ensure_ascii=False)[:300]}")
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with self.trace.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "agent": agent, "event": event, **data}) + "\n")

    def run(self, agent_name: str, task: str) -> str:
        agent = AGENTS[agent_name]
        messages = [{"role": "user", "content": task}]
        for _ in range(MAX_TURNS):
            if self.calls_left <= 0:
                return "[stopped: the run's LLM call budget is spent]"
            self.calls_left -= 1
            response = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                output_config={"effort": "medium"},
                # a safety decline is re-run server-side on Anthropic's recommended fallback model
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=agent.system,
                tools=[TOOLS[name] for name in agent.tools],
                messages=messages,
            )
            self.usage.update(calls=1, input_tokens=response.usage.input_tokens,
                              output_tokens=response.usage.output_tokens)
            # Append the whole content (thinking and tool_use blocks included) and never edit it afterwards.
            messages.append({"role": "assistant", "content": response.content})
            text = "".join(block.text for block in response.content if block.type == "text").strip()
            if response.stop_reason in ("refusal", "max_tokens"):
                # A cut-off turn can end in a half-written tool call. Never execute it.
                return f"[stopped: {response.stop_reason}] {text}"
            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                self.log(agent.name, "done", report=text)
                return text
            # All results for one turn go back together, in a single user message.
            messages.append({"role": "user", "content": [self.execute(agent, call) for call in calls]})
        return "[stopped: turn limit reached before the task was finished]"

    def execute(self, agent: Agent, call) -> dict:
        args = dict(call.input)
        self.log(agent.name, call.name, args=args)
        try:
            if call.name not in agent.tools:  # privilege separation is enforced here, not by the prompt
                raise Denied(f"{agent.name} is not allowed to use {call.name}")
            note = self.gate(agent, call.name, args)
            result, failed = note + self.dispatch(call.name, args), False
        except Denied as denied:
            result, failed = str(denied), True
        except Exception as error:  # any tool failure goes back to the model, which can usually recover
            result, failed = f"{type(error).__name__}: {error}", True
        self.log(agent.name, "result", tool=call.name, failed=failed, output=result[:500])
        return {"type": "tool_result", "tool_use_id": call.id, "content": result, "is_error": failed}

    def gate(self, agent: Agent, tool: str, args: dict) -> str:
        """Human-in-the-loop checkpoint: approve, reject with a reason, or rewrite the command."""
        if self.human is None or not needs_approval(tool, args):
            return ""
        reply = self.human(f"\n>>> WAITING FOR YOU: [{agent.name}] wants to {tool}: "
                           f"{json.dumps(args, ensure_ascii=False)}\n"
                           ">>> approve? [y]es / n <reason> / e <replacement command>: ").strip()
        self.log(agent.name, "human", tool=tool, reply=reply)
        if reply.lower() in ("", "y", "yes"):
            return ""
        if reply.lower().startswith("e ") and tool == "run_command":
            args["command"] = reply[2:].strip()
            return f"(The human replaced your command with: {args['command']})\n"
        reason = re.sub(r"^no?\b\s*", "", reply, flags=re.IGNORECASE)  # "n too risky" -> "too risky"
        raise Denied(f"The human rejected this action. Reason: {reason or 'none given'}")

    def dispatch(self, tool: str, args: dict) -> str:
        match tool:
            case "list_files":
                return self.sandbox.list_files()
            case "read_file":
                return self.sandbox.read_file(args["path"])
            case "write_file":
                return self.sandbox.write_file(args["path"], args["content"])
            case "run_command":
                return self.sandbox.run(args["command"])
            case "delegate":
                return self.run(args["agent"], args["task"])
            case "ask_human":
                if self.human is None:
                    return "No human is available in this run. Make a reasonable choice and state it."
                return self.human(f"\n>>> WAITING FOR YOU, a question: {args['question']}\n> ")
        raise ValueError(f"unknown tool {tool}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # piped output on Windows defaults to cp1252, which has no "≈"
    parser =argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--yes", action="store_true", help="unattended: approve everything, answer no questions")
    options = parser.parse_args()

    trace = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with Sandbox(options.workspace) as sandbox:
        runtime = Runtime(anthropic.Anthropic(), sandbox, human=None if options.yes else input, trace=trace)
        report = runtime.run("orchestrator", options.task)
        cost = sum(runtime.usage[kind] * usd / 1e6 for kind, usd in USD_PER_MTOK.items())
        runtime.log("run", "usage", **runtime.usage, usd=round(cost, 4))
    print(f"\n{report}\n\nworkspace: {sandbox.workspace}\ntrace: {trace}")


if __name__ == "__main__":
    main()
