#!/usr/bin/env python3
"""Launch coding-agent sessions to play Craftax.

Adapted from cc_humanrl's run.py, which took it from cc_autumn and arc-code. One
world is one long session with its own workspace: the prompt, the log it writes, the
observations it is given, and the notes it keeps. Worlds are independent, so they run
concurrently.

A run longer than one agent session is played by several of them: ``--stint`` says
how many actions each may play, and when a stint is spent that session is over and
the launcher starts the next one from exactly where it stopped. ``--continue`` does
the same thing to a launch that has already finished, rebuilding its world from the
recorded history and carrying it on under a larger ``--budget``.

Each session's stream is recorded verbatim to ``agent_stream.jsonl`` — it is both the
trace to debug from and the source of the cost figures.

Two things differ from the humanRL port:

* **The brief is generated, not copied.** A run gives out whichever of the package's
  three observation channels ``--obs`` asked for, and the first paragraph of
  ``GAME.md`` says which. Written by hand it would drift from what the log actually
  carries the first time somebody ran with a different channel.
* **A workspace is still named after nothing**, but for a weaker reason. In cc_humanrl
  the six games were one game with five appearances, so a name would have been the
  answer key. Here two sessions never share a world — but a sibling's working-out
  would still make one run evidence about two, and the launch's own record has to
  stay out of reach either way.

There is still no broker, so the package is reachable from the agent's machine and
only the audit and the interpreter split stand there. And a world cannot be
*continued* after a session dies, because it is a JAX state in a process rather than a
session on a server. ``--replay`` starts the world again in the same workspace, which
keeps the notes and rotates the log — the agent's memory survives, its world does not.
"""

import argparse
import asyncio
import json
import os
import secrets
import shutil
import sys
import time
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parent
# rig/ is not a package, so that a sandbox can overwrite any of it between builds.
sys.path.insert(0, str(REPO / "rig"))

from agents import AGENTS, HARVESTED  # noqa: E402
from audit import Audit, audit_session  # noqa: E402

from act import DEFAULT_BUDGET, RESULT  # noqa: E402
from craftax_game import CHANNELS, DEFAULT_CHANNELS, VARIANTS  # noqa: E402

# Outside the repository on purpose. A workspace under it puts this harness's source,
# its git history, the interpreter that can import the package and every sibling
# session one `..` away.
#
# Named for neither game, and shared with the NetHack harness, because a workspace's
# own path is the one string a session cannot avoid reading: it is in `pwd`, in every
# absolute path it writes and in the note every failed tool call appends. A root
# called `craftax-runs` told each of the first five runs what it was playing before
# it had played anything.
RUNS = Path(os.environ.get("RUNS_ROOT") or Path.home() / "agent-runs")
# The interpreter the workspace's `python` shim points at: numpy and Pillow, and
# deliberately not the package. Built by tools/make_agent_venv.sh.
AGENT_PYTHON = REPO / ".agent-venv" / "bin" / "python"
# The filesystem fence, applied to a session's own process before its CLI starts.
# See `fenced_argv` below and rig/fence.py for what it is and why only one arm has
# one.
FENCE = REPO / "rig" / "fence.py"

# The longest a chain will wait between sessions that played nothing. An hour is
# the shape of the thing being waited out — an exhausted rate window — and waiting
# longer than one before checking again buys nothing. (From cc_nle, df61c08.)
MAX_IDLE_WAIT = 3600

# No I, O, 0 or 1: a label is read off a directory listing and typed back by hand.
LABEL_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
LABEL_LEN = 5
RIG = ".rig"
LABELS = "labels.json"

# How each channel is described in the brief. One sentence each, naming where the
# files go and nothing about what is in them.
SAID = {
    "pixels": "an image, written to `frames/` as a PNG",
    "text": "a description, written to `text/` as a text file",
    "symbolic": "a row of numbers, written to `symbolic/` as a .npy",
}

TASK = (
    "Read CLAUDE.md, then play this game with ./act until `./act status` reports "
    "the run is over. Work autonomously and do not stop to ask questions."
)
REPLAY_TASK = (
    "A previous session played this world and its workspace is yours: read CLAUDE.md, "
    "then recover what it established from notes.md and the logs-attempt*.txt it left "
    "before acting. The world has been started again from the beginning, so what it "
    "learned still applies. Continue with ./act until `./act status` reports the run "
    "is over. Work autonomously and do not stop to ask questions."
)
# The difference from REPLAY_TASK is the whole of why this exists: there, the world
# went back to the beginning and only the notes carried over. Here the run itself
# carries over — same life, same inventory, same position — and the session before
# this one stopped mid-thought because its stint ran out, not because it was done.
# So the handover is stronger and the instruction has to be too: `logs.txt` is one
# file for the whole run and is far too large to read, `./act status` and `./act
# board` are where you are, and notes.md is what the last session managed to say.
CONTINUE_TASK = (
    "This run is already part-played and you are the session that continues it. Read "
    "CLAUDE.md, then notes.md — a previous session wrote it for you and it is the "
    "only thing it could hand you. `logs.txt` holds every action of the run so far "
    "and is too big to read: its opening lines document its format, `./act status` "
    "and `./act board` say where you are now, and anything else you want from it you "
    "get with a script. The world is exactly where the last session left it — its "
    "life, what it was carrying, where it stood. Any helper scripts in the workspace "
    "are yours. Continue with ./act until `./act status` reports this session is "
    "over, and before it does, leave notes.md fit for whoever comes next. Work "
    "autonomously and do not stop to ask questions."
)
STREAM_LIMIT = 1 << 22  # single stream-json lines can be large
# Telemetry an adapter *assigns* rather than adds, because the CLI reports one
# cumulative figure per session: right within a session, and across a chain it would
# keep only the last session's. The launcher banks each session's contribution and
# puts the running total back, which leaves both adapters' semantics alone. Before
# this, a four-session run that cost $258 reported $4.64 — its last session.
BANKED = ("cost_usd", "input_tokens", "output_tokens",
          "cache_read_tokens", "cache_creation_tokens")
# The same accounting one level up. A run can outlive a launch as well as a session —
# 3,000 actions, continued to 10,000, continued again to 20,000 — and everything the
# report takes from the record (`actions_used`, the lives, every score) is already the
# whole run's. These are the fields the launcher counts for itself, so left alone they
# described a 20,000-action run with the last launch's bill: $169 of the $532 it had
# actually cost. A continued run picks them up from the report it is continuing.
CARRIED = ("seconds", "turns", "tool_calls", "compactions", "sessions", *BANKED)


class Report(BaseModel):
    # `label` is what the workspace is called and the only one of these the session
    # could have seen. `variant` and `seed` come from the launch's own record.
    label: str
    variant: str = "craftax"
    seed: int = 0
    obs: list[str] = list(DEFAULT_CHANNELS)
    workspace: str
    agent: str = "claude"
    model: str = ""
    exit_code: int = 0
    seconds: float = 0.0
    audit: Audit = Audit()
    # One API response, however many events it arrived as. See `Claude.absorb`.
    turns: int = 0
    tool_calls: int = 0
    # Which response the last event belonged to, so that the several events one
    # response arrives as are counted once. Working state rather than a finding, so
    # it is kept off the report: an adapter needs somewhere to remember this, and it
    # cannot be the adapter, which is one object shared by every world in a launch.
    last_message: str | None = Field(default=None, exclude=True)
    # Whether the sessions ran inside the filesystem fence — see `fenced_argv`. Only
    # one arm has one, so this is the field that keeps the two arms' scores from
    # being read as if they were measured the same way.
    fenced: bool = False
    # How many agent sessions played this run. More than one means it was stinted:
    # every counter above is the sum over all of them, and every score below is the
    # run's, because the run is the thing they were all playing.
    sessions: int = 1
    compactions: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    # -- the score, in the units the benchmark itself reports ----------------- #
    budget: int = 0
    actions_used: int = 0
    max_score: int = 0
    # Achievement-weighted return: the best single life, the union over the run, and
    # every life in order — the last being the thing no RL baseline has an analogue
    # for, since the agent carries notes across deaths where a policy carries weights.
    best_episode: int = 0
    best_episode_pct: float = 0.0
    # Mean over the lives the world ended. This is the one that sits beside
    # PPO-RNN@1B's 15.3%, and `episodes_completed` is its n — often 1 or 2, which is
    # why the matrix is several seeds rather than one long run.
    episodes_completed: int = 0
    mean_episode: float = 0.0
    mean_episode_pct: float = 0.0
    score: int = 0
    score_pct: float = 0.0
    episode_scores: list[int] = []
    achievements: list[str] = []
    lives: int = 0
    deaths: int = 0
    unique_cells: int = 0
    max_level: int = 0

    # Registered despite not being approved. Registration is not use — the audit's
    # provider-reported counter is what says whether anything was reached — but a
    # tool that turns up here is one the denylist missed.
    unapproved_tools: list[str] = []

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def child_env(api_key: str | None) -> dict[str, str]:
    """A clean environment for a spawned agent.

    The parent's CLAUDE_* variables identify *this* session; inheriting them would
    make the child a continuation of it rather than a session of its own, and would
    bill it accordingly. (arc-code, run.py)

    CODEX_* for the same reason and one more: CODEX_HOME is how a session is given a
    configuration directory of its own, and a value inherited from whoever launched
    the run would put every session back in the operator's.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CLAUDE_CODE_", "CLAUDECODE", "CLAUDE_", "CODEX_"))
    }
    env.pop("ANT_API_KEY", None)
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


def brief(channels: list[str]) -> str:
    """GAME.md with its first paragraph filled in from what this run gives out.

    Generated rather than written, so a run with a different `--obs` cannot end up
    with a brief describing a channel it does not have. The wording says where the
    files go and nothing about what is in them.
    """
    said = [SAID[c] for c in CHANNELS if c in channels]
    if len(said) == 1:
        body = f"Its state is {said[0]} every time you act."
    else:
        body = (
            "Its state is written down every time you act, in more than one form: "
            + ", ".join(said[:-1])
            + f", and {said[-1]}."
        )
    return (REPO / "GAME.md").read_text().replace(
        "{observations}", f"{body} That is the whole of what you are told."
    )


BIN = ".bin"


def bin_dir(root: Path) -> Path:
    """Neutral names for the interpreters and the actuator.

    A shim is a file in the workspace, and the first thing a curious session does
    with a command it has been handed is read it. Written directly, `act` says
    `exec "<...>/cc_craftax/craftax-code/.env-venv/bin/python" "<...>/craftax-code/act.py"`,
    which names the game twice before the session has played anything — and the run
    that prompted this read it at its nineteenth command. Through these the shim says
    `<launch>/.bin/env-python <launch>/.bin/actuator.py`.

    Launching the daemon through them closes the same leak in `ps`: `/proc` is
    readable inside the fence, so a session that lists processes would otherwise find
    the harness's path on the actuator's own command line.

    A fence one step deep, not a wall. Reading *these* — `cat .bin/env-python`, or
    `ls -l` on the directory — gets you the harness's path, and a session that reads
    the actuator's source finds the game named in its first lines. Both are
    *reaching* rather than noticing, and the audit grades them. What this closes is
    the accident: the session that learns what it is playing from a path it printed
    for some other reason.

    The interpreters are wrappers rather than symlinks, and that is not a style
    choice. A venv's `python` finds its own site-packages from the directory it was
    invoked through, so a symlink to one lands outside the venv and imports nothing.
    The actuator is a symlink, because Python resolves a script's symlink before
    deciding what is on `sys.path`.
    """
    where = root / BIN
    where.mkdir(parents=True, exist_ok=True)
    for name, target in (("env-python", Path(sys.executable)), ("python", AGENT_PYTHON)):
        (where / name).write_text(f'#!/bin/sh\nexec "{target}" "$@"\n')
        (where / name).chmod(0o755)
    link = where / "actuator.py"
    if not (link.is_symlink() and link.resolve() == (REPO / "act.py").resolve()):
        link.unlink(missing_ok=True)
        link.symlink_to(REPO / "act.py")
    return where


def make_workspace(root: Path, label: str, channels: list[str]) -> Path:
    """A workspace holds the prompt, a way to act, a way to look, and nothing else.

    The brief says what this environment is; PROMPT.md says how to play and names no
    environment. A new benchmark means a new brief and actuator — the prompt travels.

    Two shims. `act` runs the actuator under this project's interpreter directly
    rather than through `uv run`, because the agent invokes it hundreds of times.
    `python` is not a convenience: the observation is a PNG, and the system
    interpreter on this machine has neither numpy nor Pillow, so without it the run
    is unplayable. It points at an interpreter that cannot import the package —
    handing over the harness's own would hand over the world generator.

    Both shims run through the launch's `.bin`, so that reading one says nothing
    about what is being played. See `bin_dir`.
    """
    if not AGENT_PYTHON.exists():
        raise SystemExit(
            f"run: no agent interpreter at {AGENT_PYTHON} — the observations cannot "
            "be opened without one. Build it with tools/make_agent_venv.sh"
        )
    ws = root / label
    ws.mkdir(parents=True)
    (ws / "CLAUDE.md").write_text(brief(channels) + "\n" + (REPO / "PROMPT.md").read_text())
    neutral = bin_dir(root)
    for name, interpreter, script in (
        ("act", neutral / "env-python", f' "{neutral / "actuator.py"}"'),
        ("python", neutral / "python", ""),
    ):
        shim = ws / name
        shim.write_text(f'#!/bin/sh\nexec "{interpreter}"{script} "$@"\n')
        shim.chmod(0o755)
    return ws


def refresh_brief(ws: Path) -> None:
    """Rewrite a kept workspace's brief from the current GAME.md and PROMPT.md.

    A run that keeps its workspace keeps the CLAUDE.md whatever version of the
    harness started it wrote, and a brief that contradicts the log is worse than a
    stale one: a session told "the budget is the whole of what you have" beside a
    preamble saying it has a stint of it has to decide which to believe.

    The channels come from the record and not from the arguments, for the same
    reason the world does — they are part of what this run is. Nothing else in the
    workspace is touched: notes.md and the helper scripts are the session's, and
    CLAUDE.md is the harness's.
    """
    channels = list(json.loads((ws / "state.json").read_text())["obs"])
    (ws / "CLAUDE.md").write_text(brief(channels) + "\n" + (REPO / "PROMPT.md").read_text())


def credential_life(path: Path) -> tuple[float, float]:
    """When a stored login's access and refresh tokens run out, as POSIX seconds.

    ``(0, 0)`` for anything missing, unreadable or malformed, which makes all three
    the same answer: this credential is not worth keeping.
    """
    try:
        oauth = json.loads(path.read_text()).get("claudeAiOauth") or {}
        return (float(oauth.get("expiresAt") or 0) / 1000,
                float(oauth.get("refreshTokenExpiresAt") or 0) / 1000)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0, 0.0


def session_config(root: Path, label: str, agent=None) -> Path:
    """A config directory of this session's own, seeded with the CLI's credential.

    The CLI keeps its sessions and its memory of a project under its config
    directory, which by default is the operator's ``~/.claude`` — or ``~/.codex``,
    which holds the same kind of thing under different names: a prompt history, a
    memories store and a log of every session the operator has run. A session is told
    about its memory directory, so listing the parent is a short walk to a sibling
    project's notes; pointing the whole tree somewhere else removes that walk.

    Which directory, which credential file and which of two copies is the fresher are
    the CLI's business and live in its adapter. Everything below is the same for both.

    Mitigation and not prevention: an absolute path still reaches the real one, and
    until a sandbox exists only the audit stands there.

    The credential is a symlink to the operator's until a session refreshes a token,
    at which point the CLI writes the file anew and the symlink becomes a copy. Which
    of the two to keep is decided by which lasts longer, and both directions of that
    have been wrong here before.

    Leaving the copy alone was wrong: it expires where the operator's does not, so a
    launch resumed the next day failed to authenticate in every session that had ever
    refreshed. Re-linking unconditionally is wrong the other way, and only shows up on
    a run long enough to need several sessions: an access token lasts about eight
    hours and nothing refreshes the operator's file while a chain is running, so from
    the second session on the copy is the *newer* of the two and relinking hands the
    next session a token that has already expired.

    So: keep whichever expires later. A chained run then sustains itself — session one
    seeds from the operator's file, refreshes when it needs to, and every session
    after inherits the refreshed copy — for as long as the refresh token lives, which
    is about a month rather than about eight hours.
    """
    agent = agent or AGENTS["claude"]
    config = root / ".sessions" / label / agent.CONFIG_DIR
    config.mkdir(parents=True, exist_ok=True)
    credential = Path.home() / agent.CONFIG_DIR / agent.CREDENTIAL
    link = config / agent.CREDENTIAL
    if not credential.exists():
        return config
    if link.is_symlink() and link.resolve() == credential.resolve():
        return config
    if agent.freshness(link) > agent.freshness(credential):
        return config
    link.unlink(missing_ok=True)
    link.symlink_to(credential)
    return config


def site_packages(venv: Path) -> Path:
    """Where a virtual environment keeps what was installed into it."""
    for found in sorted(venv.glob("lib/python*/site-packages")):
        return found
    raise SystemExit(f"run: {venv} has no site-packages — is it built?")


def fenced_argv(agent, argv: list[str], ws: Path, home: Path) -> list[str]:
    """`argv`, wrapped in the only filesystem the session will be able to see.

    The fence itself is rig/fence.py; this is the allowlist, and the allowlist is
    the whole design. What it exists to close is a single reach: the package's
    source carries every achievement by name and this harness carries a scripted
    player that walks the tech tree, so a session that greps either one is reading
    the answer instead of playing. That is not hypothetical — see the module
    docstring in rig/fence.py for the session that did it.

    The Codex arm is fenced by default and the Claude arm is not, which is an
    asymmetry and is recorded as one: `Report.fenced` says which side of it a run was
    on. The Claude arm's published 30k pass had the same reach and never used it, so
    fencing *that* run now would change a result rather than protect one; fencing this
    arm makes its score a claim about playing rather than about reading. A new Claude
    run can be fenced with `--fence` — the CLI runs inside it unchanged — and a model
    with no record of staying in bounds is the case for it. Every run's side is
    disclosed; none is inferred from another's.

    Three kinds of grant, and the difference between them matters:

    * **read** — the system, the two interpreters, the CLI's own binary, and the
      credential, which is granted by name because the copy in `home` is a symlink
      to it and Landlock checks where a symlink lands.
    * **write** — the workspace, this session's config directory, and the temporary
      directories every CLI assumes.
    * **list** — the harness's own directory and the interpreter's `site-packages`,
      whose *names* are readable and whose contents are not. Python lists the
      directory a script sits in to find the module beside it, so `./act` cannot run
      without this; with it, everything in this repository except the two files
      `./act` is made of stays shut, `tools/route.py` among them.

    The package is closed by never being named: rules only ever widen a hierarchy,
    so `site-packages` is passed as its contents minus the one entry that is the
    answer key.
    """
    binary = shutil.which(argv[0])
    if not binary:
        raise SystemExit(f"run: {argv[0]} is not on PATH — there is nothing to fence")
    # Not the binary's directory but the release it was installed as: this CLI ships
    # tools of its own — `rg` among them — in a sibling of its `bin`, and a session
    # that loses them has lost a capability the unfenced arm keeps. The fence is
    # meant to take away the answer, not the tools.
    shipped = Path(binary).resolve().parent
    shipped = shipped.parent if shipped.name == "bin" else shipped
    site = site_packages(REPO / ".env-venv")
    credential = Path.home() / agent.CONFIG_DIR / agent.CREDENTIAL
    read = [
        Path("/usr"),
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
        # /etc/resolv.conf is a symlink into here on this machine, and a session
        # that cannot resolve a name cannot reach its own API.
        Path("/run/systemd/resolve"),
        Path(sys.base_prefix),  # the real interpreter behind both venvs, and its stdlib
        shipped,  # the CLI's own binary and the tools it ships beside it
        # The neutral names the workspace's shims are written in. Granted as well as
        # `act.py` below, not instead of it: `actuator.py` is a symlink and Landlock
        # checks where a symlink lands, not what it is called.
        ws.parent / BIN,
        REPO / "act.py",  # the two files the workspace's `act` shim runs
        REPO / "craftax_game.py",
        REPO / ".env-venv" / "bin",
        REPO / ".env-venv" / "pyvenv.cfg",
        AGENT_PYTHON.parents[1],  # the agent's own interpreter, which has no package
        *(p for p in site.iterdir() if not p.name.startswith("craftax")),
        *([credential] if credential.exists() else []),
    ]
    write = [ws, home, Path("/tmp"), Path("/var/tmp"), Path("/dev")]
    listed = [REPO, site]
    flags: list[str] = []
    for flag, paths in (("--ro", read), ("--rw", write), ("--ls", listed)):
        flags += [arg for path in paths for arg in (flag, str(path))]
    return [sys.executable, str(FENCE), *flags, "--", *argv]


async def act(ws: Path, *args: str) -> str:
    """Run one actuator command in a workspace, as the launcher rather than as the
    session. Through the launch's `.bin` like the session's own shim, because this is
    what starts the daemon and `/proc` is readable inside the fence: launched by its
    real path, the actuator would name the game on its own command line."""
    neutral = bin_dir(ws.parent)
    proc = await asyncio.create_subprocess_exec(
        str(neutral / "env-python"),
        str(neutral / "actuator.py"),
        *args,
        cwd=ws,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = out.decode()
    if proc.returncode != 0:
        raise RuntimeError(f"act {' '.join(args)} failed in {ws}:\n{text}")
    return text


# --------------------------------------------------------------------------- #
# Which world is which workspace
# --------------------------------------------------------------------------- #


def write_arm(root: Path, label: str, report: "Report") -> Path:
    """Who is playing this world, written before the first session starts.

    The report says it too, but a report is written when a launch *ends*, and a page
    built while the run plays has to tell four arms apart before any of them has
    ended — by the model, which two runs of one CLI differ in, and by the fence,
    which a Claude run can now be on either side of. Rewritten on every launch,
    because a `--continue` could in principle change either.
    """
    out = rig_dir(root) / "arms"
    out.mkdir(exist_ok=True)
    path = out / f"{label}.json"
    path.write_text(json.dumps({
        "agent": report.agent, "model": report.model, "fenced": report.fenced,
        "effort": getattr(AGENTS[report.agent], "EFFORT", ""),
    }, indent=2))
    return path


def rig_dir(root: Path) -> Path:
    """Where the launch keeps what no session may read.

    A dotted directory rather than the launch root, because the root is one `..`
    from every workspace and a listing of it would otherwise carry the mapping, the
    summary and every other session's score.
    """
    found = root / RIG
    found.mkdir(parents=True, exist_ok=True)
    return found


def write_report(root: Path, report: "Report") -> Path:
    """The session's verdict, kept out of the workspace it is about.

    It carries the achievement set by name, which is the tech tree — and a replay is
    *told* to read what its predecessor left behind, so this would be waiting for it.

    A previous verdict is rotated rather than replaced, for the same reason the log
    is: a replay must not overwrite the result of the session it is replaying.
    """
    out = rig_dir(root) / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"{report.label}.json"
    if path.exists():
        n = 1 + len(list(out.glob(f"{report.label}-attempt*.json")))
        path.rename(out / f"{report.label}-attempt{n}.json")
    path.write_text(report.model_dump_json(indent=2))
    return path


def carried(root: Path, label: str, resuming: bool) -> dict[str, float]:
    """What this run had already spent before this launch.

    Read from the report this launch will rotate, and only when the run is being
    continued: `--replay` starts the world over, and what it cost to play a world
    that no longer exists is not part of what the new one costs.
    """
    path = rig_dir(root) / "reports" / f"{label}.json"
    if not resuming or not path.exists():
        # Nothing was played before this launch, so every one of these is nought —
        # including `sessions`, whose blank below is one. The two cases differ: there
        # being no prior run is not the same as a prior run not having said.
        return dict.fromkeys(CARRIED, 0)
    was = json.loads(path.read_text())
    # A field a report does not carry is a field that report is silent about, not a
    # field it says is zero, and the two differ for exactly one of these. A report
    # exists because a run was played, so a missing `sessions` is one session and not
    # none — and read as none, the session that started the run vanished from every
    # total after it. That is what happened here: the 3,000-action pilot was written
    # before this field existed, so the 30,000-action run it grew into reported 12
    # sessions against the 13 its own stream records.
    #
    # The blanks are the report's own defaults rather than a list kept alongside, so
    # a field added later cannot acquire a second, wrong, default here.
    blanks = {field: Report.model_fields[field].default for field in CARRIED}
    return {
        field: blanks[field] if (value := was.get(field)) is None else value
        for field in CARRIED
    }


def assign_labels(root: Path, plan: list[tuple[str, int]]) -> dict[str, tuple[str, int]]:
    """An opaque name for each (variant, seed), recorded outside every workspace.

    Drawn at random per launch rather than derived from the spec: a name that said
    which seed it was would let a session recognise a world it had played before, and
    a launch that ran the same world twice would have two workspaces announcing it.
    """
    path = rig_dir(root) / LABELS
    known = {
        label: (row["variant"], row["seed"])
        for label, row in json.loads(path.read_text()).items()
    } if path.exists() else {}
    have = {spec: label for label, spec in known.items()}
    for spec in plan:
        if spec in have:
            continue
        while (label := "".join(secrets.choice(LABEL_CHARS) for _ in range(LABEL_LEN))) in known:
            pass
        known[label], have[spec] = spec, label
    path.write_text(
        json.dumps({k: {"variant": v, "seed": s} for k, (v, s) in known.items()}, indent=2)
    )
    return known


# --------------------------------------------------------------------------- #
# Playing
# --------------------------------------------------------------------------- #


def played(ws: Path) -> int:
    """Actions this run has spent, from the record the actuator keeps in the open."""
    state = ws / "state.json"
    if not state.exists():
        return 0
    return int(json.loads(state.read_text())["actions_used"])


async def one_session(
    argv: list[str],
    ws: Path,
    env: dict[str, str],
    report: "Report",
    agent,
) -> tuple[str, int]:
    """Run one agent session to its end, folding its stream into the run's telemetry.

    Appends to `agent_stream.jsonl` rather than truncating it, because the sessions
    of a stinted run are one run: the audit reads the whole file and the totals in
    the report are the run's, not the last session's.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=ws,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,  # codex reads stdin if it is a pipe
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_LIMIT,
    )
    # Line-buffered: a run that is killed rather than finished still leaves a
    # readable transcript, and the file can be watched while it plays.
    with (ws / "agent_stream.jsonl").open("a", buffering=1) as stream:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace")
            stream.write(line)
            absorb(agent, report, line)
    stderr = (await proc.stderr.read()).decode(errors="replace")
    await proc.wait()
    return stderr, proc.returncode or 0


def harvested_since(ws: Path) -> float:
    """The newest harvested item already folded into this run's stream.

    Read back out of the stream rather than kept in memory, because a run outlives
    the process that started it: a `--continue` picks up a workspace whose stream
    already holds every frame the earlier launch harvested, and a watermark starting
    at zero would append all of them a second time and count them twice.
    """
    path = ws / "agent_stream.jsonl"
    if not path.exists():
        return 0.0
    best = 0.0
    # A line at a time, and the cheap test first. This file runs to hundreds of
    # megabytes on a long run and all but a handful of its lines are not this, so
    # reading it whole to find them would cost more memory than the run it resumes.
    with path.open(errors="replace") as stream:
        for line in stream:
            if HARVESTED not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == HARVESTED:
                best = max(best, float(event.get("at", 0)))
    return best


def harvest(agent, home: Path, ws: Path, report: "Report", since: float) -> float:
    """Fold what the CLI recorded but did not stream into the session's stream.

    Appended after the session that produced it has ended, because that is when the
    CLI's own record is complete and no longer being written to. The events carry
    their own type, so the stream stays honest about which of its lines came off a
    stdout and which were read back out of a database afterwards.
    """
    events = agent.harvest(home, since)
    if not events:
        return since
    with (ws / "agent_stream.jsonl").open("a", buffering=1) as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
            agent.absorb(report, event)
    return max(float(event.get("at", since)) for event in events)


async def play(
    label: str,
    spec: tuple[str, int],
    root: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    siblings: list[str],
    limit: asyncio.Semaphore,
) -> Report:
    variant, seed = spec
    budget = args.budget or DEFAULT_BUDGET[variant]
    # Outside the workspace, and somewhere the launcher can find again: it holds the
    # environment's log and the score it writes, and the score is the list of
    # achievements unlocked, by name.
    env_dir = root / ".envs" / label
    async with limit:
        if args.carry_on:
            ws, opening = root / label, "resume"
            refresh_brief(ws)
            print(f"[{label}] carrying the recorded run on to {budget} actions",
                  flush=True)
        elif args.replay:
            ws, opening = root / label, "again"
            refresh_brief(ws)  # before `act init --again`, which clears state.json
            print(f"[{label}] replaying with the last session's notes", flush=True)
        else:
            ws, opening = make_workspace(root, label, args.obs), "fresh"

        # On a kept workspace the channels are the record's, not the arguments' —
        # `act --resume` ignores `--obs` for the same reason, and a report that
        # disagreed with the run would be the only place the two differed.
        channels = (list(json.loads((ws / "state.json").read_text())["obs"])
                    if opening == "resume" else list(args.obs))
        # Seeded here rather than added at the end because the adapters accumulate
        # into the report as they read the stream: they have to be counting from the
        # run's total, not from this launch's zero.
        was = carried(root, label, opening == "resume")
        agent = AGENTS[args.agent]
        report = Report(label=label, variant=variant, seed=seed, obs=channels,
                        workspace=str(ws), agent=args.agent, model=args.model,
                        budget=budget, turns=int(was["turns"]),
                        tool_calls=int(was["tool_calls"]),
                        fenced=((agent.FENCED or getattr(args, "fence", False))
                                and not getattr(args, "no_fence", False)),
                        compactions=int(was["compactions"]))

        def briefed(n: int) -> str:
            """What session `n` is told it is walking into.

            Only a first session ever meets a world at action zero, and only the
            first session of a `--replay` meets one that was started over. Every
            other session in a launch is continuing a run that is already going.
            """
            if n > 1 or opening == "resume":
                return CONTINUE_TASK
            return REPLAY_TASK if opening == "again" else TASK

        def opened(n: int) -> list[str]:
            """How the environment is started for session `n` of this run."""
            out = [
                "init",
                "--variant", variant,
                "--seed", str(seed),
                "--budget", str(budget),
                "--stint", str(args.stint),
                "--obs", *args.obs,
                "--env-dir", str(env_dir),
            ]
            if args.fresh_world:
                out.append("--fresh-world")
            # Only the first session of a launch decides how the world begins. Every
            # session after it resumes, whatever the first one did — including a
            # fresh run, whose second session picks up the state its first left.
            if n > 1 or opening == "resume":
                out.append("--resume")
            elif opening == "again":
                out.append("--again")
            return out

        write_arm(root, label, report)
        if args.dry_run:
            await act(ws, *opened(1))
            await act(ws, "stop")
            shown = " ".join(agent.argv(briefed(1), args.model, ws)[:6])
            print(f"[{label}] ready, not played: {shown} ...", flush=True)
            return report

        started = time.monotonic()
        stderr, idle = "", 0
        try:
            # One run, played by as many sessions as its stint divides it into. The
            # environment is stopped and rebuilt between them rather than held open,
            # which costs a replay of the history each time (about 3ms an action) and
            # buys two things worth more than that: a session cannot mint itself more
            # actions, because the only thing that grants a stint is a launcher
            # starting the daemon; and every handover is a clean process, which over
            # twenty hours matters more than the minutes it costs.
            banked = {field: was[field] for field in BANKED}
            watermark = harvested_since(ws)
            for n in count(1):
                before = played(ws)
                turns_before, calls_before = report.turns, report.tool_calls
                # Zeroed so each session contributes exactly its own figures: one
                # that is killed before its result event contributes nothing rather
                # than the running total a second time.
                for field in BANKED:
                    setattr(report, field, 0)
                await act(ws, *opened(n))
                argv = agent.argv(briefed(n), args.model, ws)
                # Re-seeded per session and not per run: the CLI replaces the
                # symlink with a copy of whatever the credential was when it last
                # refreshed a token, and a chain of sessions can outlive that by
                # many hours (F13).
                home = session_config(root, label, agent)
                session_env = env | {agent.CONFIG_ENV: str(home)}
                # After the config directory exists, because the fence has to grant
                # it: a session cannot be given a home it may not write to.
                if report.fenced:
                    argv = fenced_argv(agent, argv, ws, home)
                stderr, report.exit_code = await one_session(
                    argv, ws, session_env, report, agent)
                # Before anything reads the stream, and inside the loop rather than
                # after it: a chain's sessions share one record, and each has to be
                # taken while it is the newest thing in it.
                watermark = harvest(agent, home, ws, report, watermark)
                report.sessions = int(was["sessions"]) + n
                for field in BANKED:
                    banked[field] += getattr(report, field)
                    setattr(report, field, banked[field])
                if report.exit_code:
                    # Not fatal to the run. The actions it played are recorded and
                    # the next session rebuilds the world from them; what is lost is
                    # whatever it had not written down.
                    print(f"[{label}] session {n} exited {report.exit_code}: "
                          f"{' '.join(stderr.split())[-200:]}", flush=True)
                # Why a session played nothing, when the adapter can tell from the
                # shape of its stream. Here rather than in the retry branch below,
                # which a run without a stint never reaches: a first smoke run is
                # exactly where this diagnosis is worth the most, and exactly the
                # run that would otherwise be told only that nothing was played.
                if played(ws) <= before:
                    why = getattr(agent, "no_commands", lambda *_: "")(
                        report.turns - turns_before, report.tool_calls - calls_before)
                    if why:
                        print(f"[{label}] session {n} {why}", flush=True)

                state = json.loads((ws / "state.json").read_text())
                if state["terminal"]:
                    break
                if not args.stint:
                    # Without a stint a run is one session, which is what every run
                    # was before stints existed. A session that stops with budget
                    # left has decided it is finished, and starting another would be
                    # the launcher overruling it rather than continuing it.
                    break
                if state["actions_used"] <= before:
                    # It played nothing at all: an API error at startup, an expired
                    # credential, a session that read the workspace and gave up.
                    # Costing almost nothing, a retry or two is worth it on a run
                    # meant to go unattended for hours — but a chain that cannot make
                    # progress must stop rather than spend the night failing.
                    idle += 1
                    if idle > args.retries:
                        print(f"[{label}] session {n} played no actions, and neither "
                              f"did the {args.retries} before it — stopping the chain "
                              f"at {state['actions_used']}/{budget}", flush=True)
                        break
                    # The retries were written for a failure that is over by the time
                    # the next session starts. The one that actually ends a long run
                    # is not: a rate limit is a window, and retrying inside it three
                    # times in as many seconds spends the whole allowance of patience
                    # in under a minute. So the wait doubles, and a chain that means
                    # to play for a day can afford to sit out an hour of one. cc_nle
                    # learned this first; a Codex plan's quota is a week-long window,
                    # which is the same shape with a longer `--retries`.
                    wait = min(args.idle_wait * 2 ** (idle - 1), MAX_IDLE_WAIT)
                    print(f"[{label}] session {n} played no actions — trying again "
                          f"in {wait / 60:.0f} min ({idle} of {args.retries})",
                          flush=True)
                    await asyncio.sleep(wait)
                else:
                    idle = 0
                if n >= args.max_sessions:
                    print(f"[{label}] {n} sessions is the limit — stopping at "
                          f"{state['actions_used']}/{budget}", flush=True)
                    break
                print(f"[{label}] session {n} handed over at "
                      f"{state['actions_used']}/{budget} actions", flush=True)
        finally:
            # The environment is a daemon holding a JAX process; a batch of six would
            # otherwise leave six behind, each with its compiled step function.
            #
            # Not allowed to raise. The first real pilot ended with a daemon that had
            # already died, `act stop` failed, the exception left this block, and the
            # report — read and written below — was never produced at all. A run that
            # played 2757 actions reported nothing because its shutdown was untidy.
            report.seconds = was["seconds"] + time.monotonic() - started
            try:
                await act(ws, "stop")
            except (RuntimeError, OSError) as exc:
                print(f"[{label}] the environment did not shut down cleanly: "
                      f"{' '.join(str(exc).split())[:200]}", flush=True)

        if stderr.strip():
            (ws / "agent_stderr.log").write_text(stderr)
        # From the actuator's private record rather than state.json, which
        # deliberately carries neither the achievements nor the score. Written after
        # every action, so it exists unless the session never played one.
        result = env_dir / RESULT
        if not result.exists():
            print(f"[{label}] no actions were played — nothing to score", flush=True)
            report.exit_code = report.exit_code or 1
            write_report(root, report)
            return report
        record = json.loads(result.read_text())
        for field in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                      "episodes_completed", "mean_episode", "mean_episode_pct",
                      "score", "score_pct", "episode_scores", "achievements",
                      "lives", "deaths", "unique_cells", "max_level"):
            setattr(report, field, record[field])
        report.audit = audit_session(
            ws / "agent_stream.jsonl", agent, own=label, siblings=siblings,
            root=str(root), repo=str(REPO),
        )
        write_report(root, report)

        flag = "" if report.ok else f" EXIT {report.exit_code}"
        print(
            f"[{label}] {report.variant} seed {report.seed}: "
            f"{report.mean_episode_pct:.1f}% mean of {report.episodes_completed}, "
            f"{report.best_episode_pct:.1f}% best life, {report.score_pct:.1f}% union, "
            f"{report.actions_used}/{report.budget} actions, {report.deaths} deaths, "
            f"{report.turns} turns over {report.sessions} session(s), "
            f"${report.cost_usd:.2f}, {report.seconds / 60:.1f} min{flag}",
            flush=True,
        )
        if not report.ok:
            print(f"[{label}] stderr tail: {stderr.strip()[-500:]}", flush=True)
        if report.unapproved_tools:
            print(
                f"[{label}] registered but not approved: "
                f"{', '.join(report.unapproved_tools)} — extend "
                f"{type(agent).__name__}.DENIED",
                flush=True,
            )
        if report.audit.named_the_game:
            # Recall, not misconduct. Printed because it is the line between a
            # session that read the screen and one that remembered the game.
            print(f"[{label}] named the game: {report.audit.named_the_game[:160]}",
                  flush=True)
        if not report.audit.clean:
            print(f"[{label}] AUDIT FAILED — this run is not evidence:", flush=True)
            for name, hits in report.audit.findings.items():
                # Flattened: a hit is a 200-character excerpt of what the session
                # ran, and a heredoc's newlines would otherwise spill the finding
                # across a dozen unprefixed lines of the launch log.
                excerpt = " ".join(hits[0].split())
                print(f"[{label}]   {name}: {excerpt[:140]}", flush=True)
        return report


def absorb(agent, report: Report, line: str) -> None:
    """Fold one streamed event into the run's telemetry.

    Also reads back what the session registered. Which tools exist depends on the CLI
    version and on the environment it starts in, so holding the agent to a handful of
    them cannot rest on a denylist written here being complete: this records what
    actually turned up outside the approved set.
    """
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if event.get("type") == "system" and event.get("subtype") == "init":
        approved = set(getattr(agent, "ALLOWED", "").split(","))
        report.unapproved_tools = sorted(set(event.get("tools") or []) - approved)
    agent.absorb(report, event)


def summarise(reports: list[Report], root: Path) -> None:
    """The table, written where no session can read it.

    Under `.rig/` rather than the launch root for the same reason the label mapping
    is: the root is one `..` from every workspace, and the summary carries every
    session's achievement set.
    """
    rig_dir(root).joinpath("summary.json").write_text(
        json.dumps([r.model_dump() for r in reports], indent=2)
    )
    if not reports:
        print("\nno world produced a report — every one of them failed to run")
        return
    print(f"\n{'world':16}  label  mean%    n  best%  union%  actions  lives  deaths  "
          f"depth  sess   cost   minutes")
    for r in sorted(reports, key=lambda r: (r.variant, r.seed)):
        mark = "" if r.audit.clean else "  AUDIT FAILED"
        world = f"{r.variant}:{r.seed}"
        print(
            f"{world:16}  {r.label}  {r.mean_episode_pct:5.1f}  "
            f"{r.episodes_completed:>3}  {r.best_episode_pct:5.1f}  {r.score_pct:6.1f}  "
            f"{r.actions_used:>7}  {r.lives:>5}  {r.deaths:>6}  {r.max_level:>5}  "
            f"{r.sessions:>4}  "
            f"${r.cost_usd:>5.2f}  {r.seconds / 60:>7.1f}{mark}"
        )
    best = max(r.best_episode_pct for r in reports)
    frames = sum(r.audit.frames_in_context for r in reports)
    named = [r.label for r in reports if r.audit.named_the_game]
    print(
        f"\n{len(reports)} worlds, best single life {best:.1f}% of the maximum, "
        f"${sum(r.cost_usd for r in reports):.2f}, artifacts in {root}"
    )
    dirty = [r.label for r in reports if not r.audit.clean]
    print(
        f"{'AUDIT FAILED: ' + ', '.join(dirty) if dirty else 'audit clean'}; "
        f"{frames} observations read into context rather than parsed; "
        f"{len(named)}/{len(reports)} named the game"
    )


# --------------------------------------------------------------------------- #
# What to play
# --------------------------------------------------------------------------- #


def specs(args: argparse.Namespace) -> list[tuple[str, int]]:
    """`craftax` or `classic:3` -> (variant, seed) pairs.

    A bare variant takes `--seed`. Naming one variant several times over with
    different seeds is the normal shape of a matrix, since the benchmark's metric is
    an average over worlds.
    """
    out = []
    for token in args.worlds:
        variant, _, seed = token.partition(":")
        if variant not in VARIANTS:
            raise SystemExit(f"run: {variant!r} is not one of {list(VARIANTS)}")
        out.append((variant, int(seed) if seed else args.seed))
    if not out:
        out = [("craftax", args.seed)]
    return list(dict.fromkeys(out))


def unfinished(root: Path) -> list[tuple[str, tuple[str, int]]]:
    """Sessions in a launch directory whose run never reached an ending.

    A workspace names nothing, so which world it holds comes from the launch's own
    record rather than from anything inside it.
    """
    path = rig_dir(root) / LABELS
    if not path.exists():
        raise SystemExit(f"run: {path} is missing — this is not a launch directory")
    known = json.loads(path.read_text())
    out = []
    for label, row in known.items():
        state_file = root / label / "state.json"
        if not state_file.exists():
            continue
        if json.loads(state_file.read_text())["terminal"]:
            continue
        out.append((label, (row["variant"], row["seed"])))
    return out


def continuable(root: Path) -> list[tuple[str, tuple[str, int]]]:
    """Worlds in a launch that can be carried on from exactly where they stopped.

    Any world that has played an action, *including* one that reached its ending —
    a run that spent its budget is the case this exists for, since the way a run is
    extended is to resume it under a larger one. `unfinished` asks the other
    question, which worlds never reached an ending at all, and is what `--replay`
    wants: there the world starts over and only the notes survive.
    """
    path = rig_dir(root) / LABELS
    if not path.exists():
        raise SystemExit(f"run: {path} is missing — this is not a launch directory")
    known = json.loads(path.read_text())
    return [(label, (row["variant"], row["seed"]))
            for label, row in known.items() if played(root / label) > 0]


async def main() -> int:
    parser = argparse.ArgumentParser(prog="run", description=__doc__.splitlines()[0])
    parser.add_argument(
        "worlds", nargs="*",
        help="variants, optionally with a seed: craftax craftax:1 classic:3. "
             "With none, craftax at --seed",
    )
    parser.add_argument(
        "--replay", metavar="LAUNCH_DIR",
        help="play every unfinished world in a previous launch again, keeping its notes",
    )
    parser.add_argument(
        "--continue", dest="carry_on", metavar="LAUNCH_DIR",
        help="carry every played world in a previous launch on from where it "
             "stopped, under a larger --budget. The world, the life and the "
             "inventory are exactly as they were left",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=0,
                        help=f"default {DEFAULT_BUDGET} by variant")
    parser.add_argument(
        "--stint", type=int, default=0,
        help="actions one agent session may play before another takes over the same "
             "run (0: the whole budget, in one session). A run longer than a session "
             "needs this",
    )
    parser.add_argument("--max-sessions", type=int, default=100,
                        help="how many sessions a stinted run may chain")
    parser.add_argument("--retries", type=int, default=2,
                        help="sessions in a row that may play nothing before the "
                             "chain gives up")
    parser.add_argument("--idle-wait", type=int, default=60,
                        help="seconds to wait before retrying a session that played "
                             f"nothing, doubling each time to {MAX_IDLE_WAIT // 60} "
                             "min. A rate limit is a window, not an error")
    parser.add_argument("--obs", nargs="+", choices=CHANNELS, default=list(DEFAULT_CHANNELS),
                        help="which of the package's own observations the sessions get")
    parser.add_argument("--fresh-world", action="store_true",
                        help="deal a new world on each life instead of replaying this one")
    parser.add_argument("--agent", default="claude", choices=sorted(AGENTS))
    fencing = parser.add_mutually_exclusive_group()
    fencing.add_argument(
        "--fence", action="store_true",
        help="fence an agent that is not fenced by default — a new Claude run, whose "
             "model has no record yet of staying out of the package",
    )
    fencing.add_argument(
        "--no-fence", action="store_true",
        help="run an agent that is normally fenced with the whole disk instead. "
             "For finding out what a session reaches for; not for a measured run",
    )
    parser.add_argument("--model", help="model for the agent sessions (default: the agent's own)")
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="worlds at once")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build the workspaces and start the worlds, play none",
    )
    args = parser.parse_args()

    load_dotenv(REPO.parents[1] / ".env")
    args.model = args.model or AGENTS[args.agent].model
    api_key = os.environ.get("ANT_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if args.agent == "claude" and not api_key:
        # arc-code hands each session its own key so the bill is per session. With no
        # key the CLI falls back to its own stored login, which works but bills a plan
        # rather than a request — so the cost column becomes whatever the CLI chooses
        # to report, not a measurement.
        if not args.dry_run:
            print(
                "run: no ANT_API_KEY — sessions will use the CLI's own stored login, "
                "and the cost figures are only as good as what it reports",
                flush=True,
            )
        # Outside that guard on purpose: a dry run is where you would want to find
        # out that the login will not last the night, before committing the money.
        access, refresh = credential_life(Path.home() / ".claude" / ".credentials.json")
        now = time.time()
        if not refresh:
            print("run: no stored login found either — sessions will fail to "
                  "authenticate. Log in, or set ANT_API_KEY.", flush=True)
        elif refresh <= now:
            print(f"run: the stored login's refresh token expired "
                  f"{(now - refresh) / 86400:.1f} days ago — every session will fail "
                  f"to authenticate. Log in again before starting.", flush=True)
        else:
            # The access token is the short one and it is *not* the one that has to
            # outlast the run: sessions refresh their own and each inherits the last
            # one's (see session_config). The refresh token is the deadline.
            print(f"run: stored login — access token {(access - now) / 3600:+.1f}h, "
                  f"refresh token {(refresh - now) / 86400:.1f} days. Sessions "
                  f"refresh their own, so the refresh token is the one that has to "
                  f"outlast the run.", flush=True)
    if args.agent == "claude" and args.fence:
        print("run: --fence — these Claude sessions are fenced to their workspace, as "
              "the Codex arm's are. The published Claude pass was not; the report's "
              "`fenced` says which side of that this run is on.", flush=True)
    if args.agent == "codex":
        # This CLI has no per-request key to hand a session: it signs in once and
        # every session inherits a copy of that login. So there is one thing to check
        # and one thing to say about the number it will produce.
        agent = AGENTS["codex"]
        auth = Path.home() / agent.CONFIG_DIR / agent.CREDENTIAL
        if not agent.freshness(auth):
            print(f"run: no usable codex login at {auth} — every session will fail "
                  f"to authenticate. Run `codex login` before starting.", flush=True)
        else:
            print(f"run: codex login last refreshed "
                  f"{(time.time() - agent.freshness(auth)) / 3600:.1f}h ago. Sessions "
                  f"inherit a copy and refresh their own; the cost column is computed "
                  f"from this model's published prices, not reported by the CLI.",
                  flush=True)
        # Said out loud because it is the asymmetry between the two arms, and a run
        # whose log does not mention it is a run somebody will later compare to the
        # Claude pass without knowing.
        if agent.FENCED and not args.no_fence:
            print("run: sessions are fenced to their workspace — the package and "
                  "this harness are unreadable to them. The Claude arm is not "
                  "fenced; see fenced_argv.", flush=True)
        else:
            print("run: --no-fence — sessions can read the package and this "
                  "harness, which the audit will record as a finding.", flush=True)

    if args.replay and args.carry_on:
        raise SystemExit(
            "run: --replay starts the world over and --continue carries it on — "
            "they are opposites"
        )
    if args.carry_on:
        root = Path(args.carry_on)
        if not args.budget:
            raise SystemExit(
                "run: --continue needs a --budget to continue to. Resuming under the "
                "budget a run already spent would rebuild it and stop."
            )
        plan = continuable(root)
        spent = {label: played(root / label) for label, _ in plan}
        plan = [(label, spec) for label, spec in plan if spent[label] < args.budget]
        for label, done in sorted(spent.items()):
            if done >= args.budget:
                print(f"[{label}] already played {done} of the {args.budget} asked "
                      f"for — nothing to continue", flush=True)
        if not plan:
            raise SystemExit(f"run: nothing in {root} to continue to {args.budget}")
        print(f"continuing {len(plan)} worlds in {root} to {args.budget} actions")
    elif args.replay:
        root = Path(args.replay)
        plan = unfinished(root)
        if not plan:
            raise SystemExit(f"run: nothing left to play in {root}")
        print(f"replaying {len(plan)} worlds in {root}")
    else:
        root = RUNS / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        chosen = specs(args)
        plan = [(label, spec) for label, spec in assign_labels(root, chosen).items()
                if spec in chosen]

    longest = max(args.budget or DEFAULT_BUDGET[variant] for _, (variant, _) in plan)
    if not args.stint and longest > 2 * DEFAULT_BUDGET["craftax"]:
        print(
            f"run: {longest} actions with no --stint is one agent session playing all "
            f"of them, which at the pilot's rate is over a day in one process. "
            f"--stint {DEFAULT_BUDGET['craftax']} plays it as a chain of sessions.",
            flush=True,
        )

    env = child_env(api_key)
    limit = asyncio.Semaphore(args.concurrency)
    siblings = [label for label, _ in plan]
    # One session blowing up must not discard the others. Failures are reported and
    # still fail the run; they just do not take the results with them.
    results = await asyncio.gather(
        *(play(label, spec, root, args, env, siblings, limit) for label, spec in plan),
        return_exceptions=True,
    )
    reports = [r for r in results if isinstance(r, Report)]
    for (label, spec), result in zip(plan, results, strict=True):
        if not isinstance(result, Report):
            print(f"[{label}] {spec[0]}:{spec[1]} FAILED TO RUN: {result}", flush=True)

    if not args.dry_run:
        summarise(reports, root)
    print(f"workspaces in {root}")
    lost = len(plan) - len(reports)
    return 0 if not lost and all(r.ok and r.audit.clean for r in reports) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
