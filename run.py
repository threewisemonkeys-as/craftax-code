#!/usr/bin/env python3
"""Launch coding-agent sessions to play Craftax.

Adapted from cc_humanrl's run.py, which took it from cc_autumn and arc-code. One
world is one long session with its own workspace: the prompt, the log it writes, the
observations it is given, and the notes it keeps. Worlds are independent, so they run
concurrently.

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
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent
# rig/ is not a package, so that a sandbox can overwrite any of it between builds.
sys.path.insert(0, str(REPO / "rig"))

from agents import AGENTS  # noqa: E402
from audit import Audit, audit_session  # noqa: E402

from act import DEFAULT_BUDGET, RESULT  # noqa: E402
from craftax_game import CHANNELS, DEFAULT_CHANNELS, VARIANTS  # noqa: E402

# Outside the repository on purpose. A workspace under it puts this harness's source,
# its git history, the interpreter that can import the package and every sibling
# session one `..` away.
RUNS = Path(os.environ.get("CRAFTAX_RUNS") or Path.home() / "craftax-runs")
# The interpreter the workspace's `python` shim points at: numpy and Pillow, and
# deliberately not the package. Built by tools/make_agent_venv.sh.
AGENT_PYTHON = REPO / ".agent-venv" / "bin" / "python"

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
STREAM_LIMIT = 1 << 22  # single stream-json lines can be large


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
    turns: int = 0
    tool_calls: int = 0
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
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CLAUDE_CODE_", "CLAUDECODE", "CLAUDE_"))
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
    """
    if not AGENT_PYTHON.exists():
        raise SystemExit(
            f"run: no agent interpreter at {AGENT_PYTHON} — the observations cannot "
            "be opened without one. Build it with tools/make_agent_venv.sh"
        )
    ws = root / label
    ws.mkdir(parents=True)
    (ws / "CLAUDE.md").write_text(brief(channels) + "\n" + (REPO / "PROMPT.md").read_text())
    for name, interpreter, script in (
        ("act", sys.executable, f' "{REPO / "act.py"}"'),
        ("python", AGENT_PYTHON, ""),
    ):
        shim = ws / name
        shim.write_text(f'#!/bin/sh\nexec "{interpreter}"{script} "$@"\n')
        shim.chmod(0o755)
    return ws


def session_config(root: Path, label: str) -> Path:
    """A config directory of this session's own, seeded with the CLI's credential.

    The CLI keeps its sessions and its memory of a project under its config
    directory, which by default is the operator's ``~/.claude`` — where the
    operator's own notes on this very harness live. A session is told about its
    memory directory, so listing the parent is a short walk to a sibling project's
    notes; pointing the whole tree somewhere else removes that walk.

    Mitigation and not prevention: an absolute path still reaches the real one, and
    until a sandbox exists only the audit stands there.

    The link is re-seeded rather than left alone, because the CLI does not keep it:
    when it refreshes a token it writes the file anew, which replaces the symlink
    with a copy of whatever was current then. That copy expires where the operator's
    does not, so a launch resumed the next day failed to authenticate in every
    session that had ever refreshed.
    """
    config = root / ".sessions" / label / ".claude"
    config.mkdir(parents=True, exist_ok=True)
    credential = Path.home() / ".claude" / ".credentials.json"
    link = config / ".credentials.json"
    if credential.exists() and link.resolve() != credential.resolve():
        link.unlink(missing_ok=True)
        link.symlink_to(credential)
    return config


async def act(ws: Path, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(REPO / "act.py"),
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
        init = [
            "init",
            "--variant", variant,
            "--seed", str(seed),
            "--budget", str(budget),
            "--obs", *args.obs,
            "--env-dir", str(env_dir),
        ]
        if args.fresh_world:
            init.append("--fresh-world")
        if args.replay:
            ws = root / label
            init.append("--again")
            print(f"[{label}] replaying with the last session's notes", flush=True)
        else:
            ws = make_workspace(root, label, args.obs)
        await act(ws, *init)

        report = Report(label=label, variant=variant, seed=seed, obs=list(args.obs),
                        workspace=str(ws), agent=args.agent, model=args.model,
                        budget=budget)
        agent = AGENTS[args.agent]
        argv = agent.argv(REPLAY_TASK if args.replay else TASK, args.model, ws)
        env = env | {"CLAUDE_CONFIG_DIR": str(session_config(root, label))}
        if args.dry_run:
            await act(ws, "stop")
            print(f"[{label}] ready, not played: {' '.join(argv[:6])} ...", flush=True)
            return report

        started = time.monotonic()
        stderr = ""
        try:
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
            with (ws / "agent_stream.jsonl").open("w", buffering=1) as stream:
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace")
                    stream.write(line)
                    absorb(agent, report, line)
            stderr = (await proc.stderr.read()).decode(errors="replace")
            await proc.wait()
            report.exit_code = proc.returncode
        finally:
            # The environment is a daemon holding a JAX process; a batch of six would
            # otherwise leave six behind, each with its compiled step function.
            #
            # Not allowed to raise. The first real pilot ended with a daemon that had
            # already died, `act stop` failed, the exception left this block, and the
            # report — read and written below — was never produced at all. A run that
            # played 2757 actions reported nothing because its shutdown was untidy.
            report.seconds = time.monotonic() - started
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
            f"{report.turns} turns, ${report.cost_usd:.2f}, "
            f"{report.seconds / 60:.1f} min{flag}",
            flush=True,
        )
        if not report.ok:
            print(f"[{label}] stderr tail: {stderr.strip()[-500:]}", flush=True)
        if report.unapproved_tools:
            print(
                f"[{label}] registered but not approved: "
                f"{', '.join(report.unapproved_tools)} — extend Claude.DENIED",
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
          f"depth   cost   minutes")
    for r in sorted(reports, key=lambda r: (r.variant, r.seed)):
        mark = "" if r.audit.clean else "  AUDIT FAILED"
        world = f"{r.variant}:{r.seed}"
        print(
            f"{world:16}  {r.label}  {r.mean_episode_pct:5.1f}  "
            f"{r.episodes_completed:>3}  {r.best_episode_pct:5.1f}  {r.score_pct:6.1f}  "
            f"{r.actions_used:>7}  {r.lives:>5}  {r.deaths:>6}  {r.max_level:>5}  "
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=0,
                        help=f"default {DEFAULT_BUDGET} by variant")
    parser.add_argument("--obs", nargs="+", choices=CHANNELS, default=list(DEFAULT_CHANNELS),
                        help="which of the package's own observations the sessions get")
    parser.add_argument("--fresh-world", action="store_true",
                        help="deal a new world on each life instead of replaying this one")
    parser.add_argument("--agent", default="claude", choices=sorted(AGENTS))
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
    if args.agent == "claude" and not api_key and not args.dry_run:
        # arc-code hands each session its own key so the bill is per session. With no
        # key the CLI falls back to its own stored login, which works but bills a plan
        # rather than a request — so the cost column becomes whatever the CLI chooses
        # to report, not a measurement.
        print(
            "run: no ANT_API_KEY — sessions will use the CLI's own stored login, "
            "and the cost figures are only as good as what it reports",
            flush=True,
        )

    if args.replay:
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
