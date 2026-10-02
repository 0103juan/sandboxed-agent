# sandboxed-agent

[![CI](https://github.com/0103juan/sandboxed-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/0103juan/sandboxed-agent/actions/workflows/ci.yml)

A multi-agent coding system in about 300 lines of Python: an orchestrator that delegates to specialist agents, stops for a human at the decisions that matter, and can only act inside a locked-down Docker container.

It is written directly against the Anthropic Messages API, with no agent framework, so that the three things this project is about are visible in the code: the loop, the approval gate, and the isolation boundary.

```
                      ┌──────── human ────────┐
                      │ approve / reject with │
                      │ a reason / rewrite    │
                      └───────────▲───────────┘
                                  │ gate
user task ─▶ orchestrator ── delegate ──▶ coder ────▶ write_file, run_command ─┐
                  ▲            │                                               ▼
                  │            └────────▶ reviewer ─▶ read_file, run_command ─▶ Docker sandbox
                  └── final reports ◀──────────────────────────────────────────┘
```

## How it works

**One loop, three agents.** `Runtime.run` in `agent.py` is the whole agent loop: call the model, execute the tool calls it asked for, send every result back in one message, repeat. The orchestrator, the coder and the reviewer are that same loop with a different system prompt and tool list. `delegate` is an ordinary tool whose implementation calls the loop again with a fresh context, and the sub-agent's final report becomes the tool result. Sub-agents have no `delegate` tool, so the depth is bounded by construction.

**Loops that end.** Each agent gets at most 15 turns, and the whole run shares a budget of 60 model calls, so a coder and a reviewer that keep disagreeing cannot run forever. A turn cut off by `max_tokens` or declined by the model is never executed, because it may end in a half-written tool call. Tool failures go back to the model as error results so it can change approach.

**Human in the loop.** Two kinds of action wait for a person:

- every delegation, which makes the human a checkpoint on the plan before any work starts;
- any shell command that is not obviously routine. Plain `python`, `ls`, `cat` and `grep` calls run straight away; anything with pipes, redirects, substitutions or another executable asks first.

The human can approve, reject with a reason, or replace the command. A rejection is returned to the agent as a failed tool call carrying the reason, so the agent replans around it. The orchestrator can also call `ask_human` when the request is ambiguous.

**Isolation.** Model-written commands only ever run through `Sandbox.run`:

| Flag | What it prevents |
|---|---|
| `--network none` | Downloading code, exfiltrating data |
| `--read-only` with a small `/tmp` tmpfs | Modifying the container's own filesystem |
| `--cap-drop ALL`, `no-new-privileges`, non-root `--user` | Privilege escalation |
| `--memory 512m --cpus 1 --pids-limit 128` | Fork bombs, memory exhaustion |
| `timeout --signal=KILL` inside the container | Commands that never return |
| a single bind mount | Reaching anything on the host except the workspace |

File tools are confined separately: every path is resolved, symlinks included, and rejected if it lands outside the workspace. Tool permissions are checked in code as well as in the prompt, so a reviewer that tries to call `write_file` is refused even if the model attempts it.

## Run it

Requires Docker, [uv](https://docs.astral.sh/uv/) and `ANTHROPIC_API_KEY`.

```bash
uv sync
uv run pytest        # 18 tests; the loop tests need no API key, the sandbox test uses real Docker

uv run python agent.py "Build a CLI that converts CSV to JSON, with unit tests"
uv run python agent.py --yes "..."     # unattended: approves everything, answers no questions
```

Generated files land in `./workspace`. Every tool call, human decision and result is appended to `runs/<timestamp>.jsonl`.

## A real session

The task above, run interactively on 1 October 2026 with `claude-sonnet-5-5`. Taken from the run's trace, abridged:

```
orchestrator  delegate -> coder       "create csv2json.py and test_csv2json.py ..."        human: y
coder         write_file x3, then     python -m unittest -v | tail; two smoke commands     human: y
coder         done                    35 tests pass
orchestrator  delegate -> reviewer    "independently review ..."                           human: y
reviewer      read_file x2, then      python -m unittest -v | tail                         human: y
reviewer      run_command             printf 'a\n1e999\n...' | python csv2json.py --infer-types   human: y
reviewer      done                    Verdict: not approved
orchestrator  delegate -> coder       "fix defects ... and add tests"                      human: y
coder         done                    42 tests pass
orchestrator  done
```

| | |
|---|---|
| Model calls | 15 of the 60 allowed |
| Human decisions | 7, all approvals: 3 delegations and 4 shell commands with pipes |
| Tokens | 66,923 input, 14,668 output |
| Cost | $0.28 |
| Wall time | 3 min 9 s, including the time I took to answer |

What the reviewer was for. The coder's first version passed its own 35 tests. The reviewer, which can read and run but not write, reproduced three defects those tests missed, each with the command that shows it:

- `1e999` under `--infer-types` came out as `Infinity`, which is not valid JSON;
- Arabic-Indic digits (`١٢`) were silently converted to the number 12;
- a field of more than 4,300 digits crashed the CLI with a traceback instead of exiting 1.

The coder fixed those and three minor issues and added a test for each. I ran the result myself afterwards: 42 tests pass and `1e999` stays a string.

What the run did not do: the orchestrator ended without sending the fixes back to the reviewer, and said so in its final report ("the fixed version has only the coder's own test run behind it"). Nothing in the loop forces a second review.

Rejection, from an earlier run of the same task that I stopped before the review. I answered the first delegation with `n also add an --ident option`. The rejection went back to the orchestrator as a failed tool call, and instead of guessing it called `ask_human`:

```
My plan already includes `--indent N` (JSON indentation). Did you mean `--ident` as a typo for that,
or do you want a separate `--ident` option? ... (a) it's a typo, keep just --indent; (b) accept
`--ident` as an alias of --indent; (c) something else
```

I answered `a`, and it delegated again with the original plan.

## What the tests prove

`test_agent.py` drives the real loop with a scripted model, so the control flow is tested deterministically: delegation and report hand-back, rejection with a reason, command rewriting, which commands skip approval, path escapes, tool permissions, both loop bounds, and truncated turns. One test starts a real container and checks that the network is unreachable, the root filesystem is read-only, the process is not root, runaway commands are killed, and the container is removed afterwards.

## Design decisions

- **A hand-written loop instead of an SDK tool runner or a graph framework.** The loop is the subject of the project, and owning it keeps the gate, the budgets and the delegation in one readable place.
- **Server-side refusal fallback** is enabled, so a request the model declines is retried on Anthropic's recommended fallback model.
- **The message history is append-only.** Responses are stored whole, thinking blocks included, and never edited.

## Limits, stated plainly

- A Docker container is not a VM. For code from an adversary rather than from a cooperative model, use gVisor or a microVM.
- File tools run on the host, confined by path checks, not inside the container.
- The reviewer is read-only at the tool level. It could still write through a shell redirect, which is why redirects require approval; under `--yes` that protection is gone.
- Delegations run one at a time, and a run cannot be paused and resumed later.
- The loop and the sandbox are tested. Agent quality is not measured: one real task, described above, is an example and not an evaluation.
- A review that ends in "not approved" does not force a second review of the fix; the orchestrator decides.

## Layout

```
agent.py       agents, tools, the loop, the approval gate, the CLI
sandbox.py     the Docker container and the workspace path confinement
test_agent.py  loop tests with a scripted model, plus a real-Docker isolation test
```
