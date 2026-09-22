#!/usr/bin/env python3
"""Which coding agent plays, and how to read what it did.

Verbatim from arc-code rig/agents.py: which CLI is at the controls, and how to
read its event stream, has nothing to do with what is being played.

The harness does not care who is at the controls: something reads `CLAUDE.md`,
runs `./act`, and writes files. But the two CLIs that can do that disagree on
everything else — how to be run headlessly, what their event streams look like,
and whether they will tell you what a session cost. This is the whole of that
difference, so the rest of the harness can stay ignorant of it.

Both must be able to answer three questions about a stream: how many turns and
tool calls happened, what the session cost, and — for the audit — every command
the agent ran and every file it wrote.

One question is this benchmark's own. Here the observation is an image, and looking
at one is a legitimate channel — it is the channel `play_craftax` gives a person —
so `images()` counts how many entered a session's context rather than fencing them
off. It is a measurement, not a finding, and on a run this long it is the one that
says whether the frames are being parsed or eyeballed.

Where a CLI's stream does not answer all four, the adapter says where the rest is
kept rather than reporting zero: `harvest` reads back what a session did that its own
stdout left out, and the launcher folds it into the stream before anything reads it.
That is the difference between a measurement of nothing and nothing measured, and on
the frame count it is also the difference between an audit that saw everything and
one that only appeared to.
"""

import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

# Claude Code reports its own spend; Codex reports tokens and leaves the
# arithmetic to the caller, so a price is needed to compare the two.
#
# Dollars per million tokens, and the one thing here that is not measured.
# OpenAI's own pricing page renders client-side and could not be read, so these
# come from two independent trackers that agree: openrouter.ai and
# aipricing.guru, July 2026. Cache writes cost more than fresh input and are
# most of the bill on a short session, so they are priced separately rather
# than folded in.
#
# Not modelled: requests over 272K input tokens bill at 2x input and 1.5x
# output for the whole request. These usage figures are per session, not per
# request, so that cannot be applied honestly — which makes the cost of a long
# game a floor rather than a figure.
PRICES: dict[str, dict[str, float]] = {
    "gpt-5.6-sol": {"input": 5.00, "cached": 0.50, "write": 6.25, "output": 30.00},
}

# An event the launcher appended to a stream after the session that produced it had
# ended, read back out of the CLI's own records rather than off its stdout. Its own
# type, so that a stream stays honest about where each line came from and nothing
# downstream mistakes one for something the CLI streamed. See `Codex.harvest`.
HARVESTED = "harvested.item"


class Agent(Protocol):
    name: str
    model: str
    # Where the CLI keeps its sessions and its credential, and the variable that
    # moves the whole tree somewhere this launch owns. Both CLIs default to a
    # directory under the operator's home, which is where the operator's own notes
    # on this harness live — see `session_config` in run.py.
    CONFIG_ENV: str
    CONFIG_DIR: str
    CREDENTIAL: str
    # Whether a session of this CLI is run inside the launcher's filesystem fence.
    # An adapter's own answer rather than a run's, because it is a fact about how
    # this CLI behaves when it is left alone with a disk — see `Codex` below, and
    # `fenced_argv` in run.py for what the fence grants and why the two arms differ.
    FENCED: bool

    def argv(self, task: str, model: str, ws: Path) -> list[str]: ...
    def absorb(self, report: Any, event: dict) -> None: ...
    def ran(self, event: dict) -> list[str]: ...
    def results(self, event: dict) -> list[str]: ...
    def web(self, event: dict) -> int: ...
    def images(self, event: dict) -> int: ...
    def freshness(self, path: Path) -> float: ...
    def harvest(self, home: Path, since: float) -> list[dict]: ...


class Claude:
    """`claude -p`, streaming its own structured events."""

    name = "claude"
    model = "claude-opus-5"
    CONFIG_ENV = "CLAUDE_CONFIG_DIR"
    CONFIG_DIR = ".claude"
    CREDENTIAL = ".credentials.json"
    # Unfenced, and the published 30k pass is why. Thirteen sessions had the whole
    # disk and the audit came back empty: this CLI did not once reach for the
    # package or for this harness. Fencing it now would change what a finished
    # result means rather than protect an unfinished one.
    FENCED = False
    # The agent acts through ./act and reasons with a shell and files, so these
    # are pre-approved for an unattended session.
    #
    # This is approval, not availability: the session still registers every other
    # tool, and a call to one would be recorded and then denied for want of anyone
    # to approve it. Do not read this list as a claim that the web is out of
    # reach — the counter in audit.py is what establishes that.
    ALLOWED = "Bash,Read,Write,Edit,Grep,Glob"
    # arc-code stopped at approval, because the CLI it ran on registered little
    # else. This one registers reaching the web, spawning agents, orchestrating
    # workflows, loading skills and scheduling work, so the six above are held to
    # by denying the rest outright rather than by there being nobody to approve
    # them. Skill is on the list because skills are read from the project the
    # workspace sits in, which would hand the agent this repository's context.
    # The registry is not fixed: it differs by CLI version and by environment, and
    # a session run without the parent's variables registered five tools that one
    # run with them did not. So this list is a union of everything seen outside the
    # six, and run.py checks what a session *actually* registered rather than
    # trusting that this list is complete.
    DENIED = ",".join(
        (
            "WebSearch,WebFetch",  # the web, directly
            "Task,Workflow,SendMessage,ListAgents,RemoteTrigger",  # other agents
            "TaskCreate,TaskGet,TaskList,TaskUpdate,TaskOutput,TaskStop",
            "Monitor",  # runs background commands and opens WebSockets
            "Skill,ToolSearch",  # skills are read from the project this sits in
            "CronCreate,CronDelete,CronList,ScheduleWakeup",  # work outliving the run
            "EnterWorktree,ExitWorktree",  # the workspace sits inside a git repository
            "PushNotification,SendUserFile,Artifact,DesignSync",  # reaching outward
            "AskUserQuestion,EnterPlanMode,ExitPlanMode,EndConversation",  # nobody there
            "NotebookEdit,ReportFindings",  # not in the six, and not needed
            "WaitForMcpServers",  # waits on tool servers a session has none of
        )
    )
    # Empty means the CLI's own default, which for Opus 5 is `high` — that is
    # what every pass so far ran at, and leaving it unset keeps them reproducible.
    # `max` is session-only, which is exactly right when it is passed per session.
    LEVELS = ("low", "medium", "high", "xhigh", "max")
    EFFORT = os.environ.get("ARCSEC_EFFORT", "")

    def argv(self, task: str, model: str, ws: Path) -> list[str]:
        argv = [
            "claude",
            "-p",
            task,
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--allowedTools",
            self.ALLOWED,
            "--disallowedTools",
            self.DENIED,
        ]
        if self.EFFORT:
            if self.EFFORT not in self.LEVELS:
                raise SystemExit(f"claude: effort {self.EFFORT!r} is not one of {self.LEVELS}")
            argv += ["--effort", self.EFFORT]
        return argv

    def absorb(self, report: Any, event: dict) -> None:
        kind = event.get("type")
        if kind == "assistant":
            message = event.get("message", {})
            # One API response arrives as several `assistant` events — a thinking
            # block, a text block, a tool call — all carrying the same `message.id`.
            # Counted per event, a turn stopped meaning a response: the published
            # 30,000-action run reported 7,239 turns for 3,774 responses, 1.92 events
            # to a response. `tool_calls` below is immune, because a `tool_use` block
            # is counted once wherever the split puts it — which is why the two
            # disagreed with the readout for years without anything looking wrong.
            #
            # An id-less event is counted on its own: there is nothing to group it by,
            # and grouping them all under one absent id would be the opposite error.
            ident = message.get("id")
            if not ident:
                report.turns += 1
            elif ident != report.last_message:
                report.turns += 1
                report.last_message = ident
            said = message.get("content", [])
            report.tool_calls += sum(1 for b in said if b.get("type") == "tool_use")
        elif kind == "system" and event.get("subtype") == "compact_boundary":
            report.compactions += 1
        elif kind == "result":
            usage = event.get("usage", {})
            report.cost_usd = event.get("total_cost_usd", 0.0)
            report.input_tokens = usage.get("input_tokens", 0)
            report.output_tokens = usage.get("output_tokens", 0)
            report.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            report.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

    def ran(self, event: dict) -> list[str]:
        if event.get("type") != "assistant":
            return []
        said = []
        for block in event.get("message", {}).get("content", []):
            kind = block.get("type")
            # A server tool — web_search, web_fetch — runs on the provider's
            # infrastructure rather than in the sandbox, so no network rule can
            # see it. It arrives as its own block type, which this used to skip.
            if kind == "server_tool_use":
                said.append(f"{block.get('name', '')} {block.get('input', {})}")
                continue
            if kind != "tool_use":
                continue
            got = block.get("input", {})
            said.append(str(got.get("command", "")))
            said.append(str(got.get("content", "")) + str(got.get("new_string", "")))
            # Read, Grep and Glob take a path rather than a command, and what must
            # not be reached here is a file. arc-code could leave these out: the
            # answer it guarded was on the internet, so only a client could fetch
            # it. Leaving them out here made `Read <answer key>` invisible.
            said.append(
                " ".join(
                    str(got.get(key, ""))
                    for key in ("file_path", "path", "pattern", "glob", "notebook_path")
                )
            )
        return said

    def web(self, event: dict) -> int:
        """How many web requests the provider says this session made.

        Reported by the API in the result event's usage, not by the agent and not
        by the CLI, so nothing inside the sandbox can shade it. It is the only
        check here that covers a tool executing on the provider's own machines,
        where the fence has no view at all.
        """
        if event.get("type") != "result":
            return 0
        counts = (event.get("usage") or {}).get("server_tool_use") or {}
        return sum(int(v) for k, v in counts.items() if k.endswith("_requests"))

    def results(self, event: dict) -> list[str]:
        if event.get("type") != "user":
            return []
        out = []
        for block in event.get("message", {}).get("content", []):
            if block.get("type") != "tool_result":
                continue
            body = block.get("content", "")
            if isinstance(body, list):
                body = " ".join(str(p.get("text", "")) for p in body if isinstance(p, dict))
            out.append(str(body))
        return out

    def images(self, event: dict) -> int:
        """How many frames this session read by eye rather than through a script.

        A tool result carries an image block when the agent has read a PNG, which
        is exactly the human subjects\' channel and is not misconduct. It is worth
        counting anyway: a session reading hundreds of frames into context is
        spending on transcription the room it will want for a plan, and this is
        where that shows up in a report.
        """
        if event.get("type") != "user":
            return 0
        seen = 0
        for block in event.get("message", {}).get("content", []):
            if block.get("type") != "tool_result":
                continue
            body = block.get("content", "")
            if isinstance(body, list):
                seen += sum(
                    1 for part in body
                    if isinstance(part, dict) and part.get("type") == "image"
                )
        return seen

    def freshness(self, path: Path) -> float:
        """When this stored login's access token runs out, as POSIX seconds.

        Only ever compared against another copy of the same credential, so what it
        means matters less than that later is better and unreadable is worst. Zero
        for anything missing or malformed, which makes those the same answer: this
        file is not worth keeping.
        """
        try:
            oauth = json.loads(path.read_text()).get("claudeAiOauth") or {}
            return float(oauth.get("expiresAt") or 0) / 1000
        except (OSError, ValueError, TypeError, AttributeError):
            return 0.0

    def harvest(self, home: Path, since: float) -> list[dict]:
        """Nothing: this CLI's stream is the whole of what it did."""
        return []

    def no_commands(self, turns: int, tool_calls: int) -> str:
        """Nothing: this CLI has no silent failure this shape would catch."""
        return ""


class Codex:
    """`codex exec --json`, fenced by configuration and read from two sources.

    Three things about this CLI that the Claude adapter never has to think about.

    **Its own sandbox cannot start here.** `workspace-write` and `read-only` run
    every command through bubblewrap, which needs an unprivileged user namespace,
    and this machine sets `kernel.apparmor_restrict_unprivileged_userns=1`. So each
    one dies with `bwrap: setting up uid map: Permission denied` — and the model does
    not stop when that happens. Asked to write a file and read it back, it reported
    the contents it expected; no file existed. A session of this harness would play
    no actions, score zero, and look from the launcher exactly like one that read the
    workspace and gave up. So the CLI's own sandbox is off, as it is on the Claude
    arm — and what stands in its place is where the two arms part: see `FENCED`
    below, which is this adapter's answer to what a session of it does when it is
    left alone with a disk. `no_commands` below is the residual check, because a
    shell that cannot start is worth telling apart from a session that chose not to
    use one.

    **Its tool surface is a configuration, not a flag.** There is no denylist to
    pass. What a session can reach is decided by features that ship on, a browser and
    other agents among them, so `DISABLED` is the analogue of `Claude.DENIED` and
    `tools.web_search` is turned off by name.

    **Half of what it does is not in its stream.** `--json` carries messages,
    commands and file changes and nothing else. A `view_image` call — the frame
    channel this benchmark is measured on — produces no event at all: a session can
    open a hundred observations and the stream will show none of them. The CLI's own
    thread history records each one with its path, so `harvest` reads it back at the
    end of a session and the launcher folds it into the stream. Without that both the
    frame count and the audit are blind to every image a session opened, and the
    audit's claim to have seen everything would be false rather than merely partial.
    """

    name = "codex"
    model = "gpt-5.6-sol"
    CONFIG_ENV = "CODEX_HOME"
    CONFIG_DIR = ".codex"
    CREDENTIAL = "auth.json"
    # Fenced, and a smoke run is why. Given thirty actions it played eight and spent
    # the rest reading: `tools/route.py`, which is a scripted player that walks the
    # early tech tree, then the package's own constants. The audit caught it, which
    # is the audit working — but a finding at the end of a fifteen-hour run is a
    # score that measured reading. So this arm runs inside `fenced_argv`, and
    # `Report.fenced` records that it did.
    FENCED = True
    # The transcript the JSON stream leaves out, under CONFIG_ENV. Versioned by the
    # CLI: a newer one would land beside this rather than replace it, which `harvest`
    # reports rather than silently reading nothing.
    HISTORY = "thread_history_1.sqlite"
    # The analogue of Claude.DENIED. Every one of these is stable and on by default,
    # and every one is a way out of a workspace: a browser, the desktop, agents of
    # its own, plugins, and skills read from wherever the CLI finds them. Disabling
    # is not the same as denying — an unknown name here is ignored rather than
    # refused — so, as on the Claude side, what a session actually reached is
    # established by the audit and not by this list.
    DISABLED = (
        "browser_use", "browser_use_external", "browser_use_full_cdp_access",
        "computer_use", "image_generation", "apps", "in_app_browser",
        "multi_agent", "plugins", "remote_plugin", "skill_search", "hooks",
        "tool_suggest",
    )
    # The brief is about 8KB and the CLI's own default cap is smaller than this by
    # enough to matter. Named rather than defaulted because a truncated brief is a
    # silent difference between the two arms — the Codex session would be playing
    # with less of the prompt and nothing would say so.
    PROJECT_DOC_MAX = 131072
    # How hard it thinks per turn. Set once for a whole batch through the
    # environment, because it belongs to the run rather than to a game, and the
    # upper levels cost enough to be a deliberate choice rather than a default.
    # `max` and `ultra` sit above `xhigh` for this model, so the published Codex
    # pass was not run at its ceiling. This used to rewrite a requested `max`
    # down to `xhigh`, which silently capped a batch that asked to think harder.
    LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    EFFORT = os.environ.get("ARCSEC_EFFORT", "high")

    def argv(self, task: str, model: str, ws: Path) -> list[str]:
        if self.EFFORT not in self.LEVELS:
            raise SystemExit(f"codex: effort {self.EFFORT!r} is not one of {self.LEVELS}")
        argv = [
            "codex",
            "exec",
            "--json",
            # Without this the operator's ~/.codex/config.toml decides the model, the
            # effort and which plugins load — on this machine it sets all three, and
            # marks the directory this harness sits in as trusted.
            "--ignore-user-config",
            "-m",
            model,
            "-c",
            f"model_reasoning_effort={self.EFFORT}",
            "-c",
            "approval_policy=never",
            "-c",
            "tools.web_search=false",
            # The frame channel, and the one thing here turned on rather than off. It
            # is what `play_craftax` gives a person and what the Claude arm has, so a
            # run without it would be measuring a different agent on a different game.
            "-c",
            "tools.view_image=true",
            # Both arms are handed a workspace that differs in nothing, down to the
            # filename. This CLI reads AGENTS.md; the fallback makes ours its project
            # doc without a second copy of the brief to drift from the first.
            "-c",
            'project_doc_fallback_filenames=["CLAUDE.md"]',
            "-c",
            f"project_doc_max_bytes={self.PROJECT_DOC_MAX}",
            # See the class docstring: its own sandbox cannot start on this machine,
            # and a session whose every command fails does not say so.
            "-s",
            "danger-full-access",
            "--skip-git-repo-check",
            "-C",
            str(ws),
        ]
        for feature in self.DISABLED:
            argv += ["--disable", feature]
        # Last, and positional: anything after it would be read as part of the prompt.
        argv.append(task)
        return argv

    def absorb(self, report: Any, event: dict) -> None:
        kind = event.get("type")
        item = event.get("item", {})
        if kind == "item.completed" and item.get("type") == "agent_message":
            # Codex counts one turn per prompt, so a whole session is one turn.
            # Its messages are the closer analogue of Claude's assistant turns.
            report.turns += 1
        elif kind == "item.completed" and item.get("type") == "command_execution":
            report.tool_calls += 1
        elif kind == HARVESTED and item.get("type") == "imageView":
            # Looking at a frame is a tool call on the Claude side, where it is a
            # Read of a PNG, so it is one here too — otherwise the two arms' tool
            # counts would be measuring different sets of things.
            report.tool_calls += 1
        elif kind == "turn.completed":
            usage = event.get("usage", {})
            report.input_tokens = usage.get("input_tokens", 0)
            # Reasoning tokens are billed as output and are most of the spend on
            # a hard game, so they belong in the output count, not beside it.
            report.output_tokens = usage.get("output_tokens", 0) + usage.get(
                "reasoning_output_tokens", 0
            )
            report.cache_read_tokens = usage.get("cached_input_tokens", 0)
            report.cache_creation_tokens = usage.get("cache_write_input_tokens", 0)
            report.cost_usd = cost(report)

    def ran(self, event: dict) -> list[str]:
        item = event.get("item", {})
        if event.get("type") == HARVESTED:
            # A path, not a command, and the reason harvesting exists at all: the
            # audit is a claim about everything a session reached, and `view_image`
            # reaches a file. Before this, opening the answer key as a picture was
            # the one way to read it that left no trace anywhere.
            if item.get("type") == "imageView":
                return [str(item.get("path", ""))]
            return []
        if event.get("type") != "item.completed":
            return []
        if item.get("type") == "command_execution":
            return [str(item.get("command", ""))]
        if item.get("type") in ("file_change", "patch_apply"):
            return [str(item)]
        return []

    def results(self, event: dict) -> list[str]:
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "command_execution":
            return [str(item.get("aggregated_output", ""))]
        return []

    def images(self, event: dict) -> int:
        """How many frames this session read by eye rather than through a script.

        Not from the stream, which carries no such event, but from the thread history
        `harvest` folded into it — so this counts only what a launcher harvested. A
        stream read without that step reports no frames, which is the truth about
        that file rather than a claim about the session.
        """
        if event.get("type") != HARVESTED:
            return 0
        return int(event.get("item", {}).get("type") == "imageView")

    def freshness(self, path: Path) -> float:
        """When this credential was last refreshed, as POSIX seconds.

        Claude's answer to the same question is an expiry and this one is an age,
        which does not matter: the only thing either is used for is choosing between
        two copies of one credential, where later is better and unreadable is worst.

        The CLI writes nanoseconds and `fromisoformat` takes at most microseconds, so
        the fraction is truncated rather than parsed.
        """
        try:
            stamp = json.loads(path.read_text())["last_refresh"]
            stamp = re.sub(r"(\.\d{6})\d+", r"\1", str(stamp).replace("Z", "+00:00"))
            return datetime.fromisoformat(stamp).timestamp()
        except (OSError, ValueError, TypeError, KeyError):
            return 0.0

    def harvest(self, home: Path, since: float) -> list[dict]:
        """The items this session's stream left out, newer than `since`.

        `since` is a millisecond timestamp and not a row count, because a run's
        sessions share one history: every session after the first would otherwise
        re-harvest the ones before it, and every frame would be counted again.

        Returns events shaped like the stream's own so that the launcher can append
        them to it and everything downstream — this adapter, the audit, the readout —
        keeps reading one file. Failure is empty: a history that cannot be opened
        costs a run its frame count, and that is worth reporting through the count
        itself rather than by killing a session that has already been played.
        """
        history = home / self.HISTORY
        if not history.exists():
            return []
        try:
            # Read-only, and over a copy of nothing: the CLI may still hold a write
            # lock on the WAL when a chained session starts, and a harvest must never
            # be the thing that stops a run.
            db = sqlite3.connect(f"file:{history}?mode=ro&immutable=0", uri=True)
            try:
                rows = db.execute(
                    "select created_at_ms, item_json from thread_items "
                    "where created_at_ms > ? order by created_at_ms",
                    (int(since),),
                ).fetchall()
            finally:
                db.close()
        except sqlite3.Error:
            return []
        out = []
        for at, blob in rows:
            try:
                item = json.loads(blob)
            except (ValueError, TypeError):
                continue
            if item.get("type") != "imageView":
                continue
            out.append({"type": HARVESTED, "at": int(at), "item": item})
        return out

    def no_commands(self, turns: int, tool_calls: int) -> str:
        """Why a session that played nothing played nothing, if this adapter knows.

        One failure mode is worth naming because it is silent and is not the agent's
        doing: when the shell cannot start, this CLI answers the prompt anyway, out
        of what it expected the commands to produce. What that leaves behind is a
        session that spoke and never ran anything — a shape a session that gave up
        does not have, since giving up here takes reading the workspace first.
        """
        if turns and not tool_calls:
            return ("it spoke and ran nothing at all — the shell could not start, "
                    "and the model answered without it")
        return ""

    def web(self, event: dict) -> int:
        """Codex reports no such counter, so this says nothing rather than zero.

        For Codex the commands are the evidence, and they were enough: the four
        sessions that went looking for solutions were caught by the patterns in
        audit.py, from the commands they ran.
        """
        return 0


def cost(report: Any) -> float:
    """What a Codex session cost, if the price of its model is known.

    Zero means it is not, which is the honest answer — an invented price would
    quietly become the number in a comparison table.

    `input_tokens` is the total and the cached and written counts are parts of
    it, so fresh input is what is left over. Priced apart because they differ by
    more than tenfold, and most of a session's input is cache traffic.
    """
    price = PRICES.get(getattr(report, "model", ""))
    if not price:
        return 0.0
    cached, written = report.cache_read_tokens, report.cache_creation_tokens
    fresh = max(0, report.input_tokens - cached - written)
    return (
        fresh * price["input"]
        + cached * price["cached"]
        + written * price["write"]
        + report.output_tokens * price["output"]
    ) / 1_000_000


AGENTS: dict[str, Agent] = {"claude": Claude(), "codex": Codex()}
