#!/usr/bin/env python3
"""Deterministic actuator for Craftax.

Adapted from cc_humanrl's act.py, which took it from cc_autumn and arc-code: same
contract — every action goes through here, every action is appended verbatim to
``logs.txt`` next to the observation it produced, and nothing degrades gracefully.
If something is wrong, it raises.

Three things differ from the humanRL port it is cut down from:

* **There is no win.** Those games ended when the player touched the princess, so a
  reward above zero ended the run. Craftax is open-ended: a life ends in death, in
  the timeout, or on the boss, and the run carries on into the next life until the
  budget is spent. ``ending`` says which of those happened; nothing here decides a
  run was "solved".
* **The reward is shown.** In the humanRL arm reward was the score and so was
  withheld. Here it is the channel an RL policy is given — a tiered achievement
  bonus plus a tenth of the change in health (F1) — and the whole point is to play
  the game as shipped, so every non-zero reward goes in the block header. What
  stays out of the workspace is the *achievement set*: naming what has been
  unlocked would hand over the tech tree a rung at a time.
* **The observation is whatever the run was configured to give.** ``CraftaxGame``
  owns that; this file writes one ``[channel] path`` line per channel and never
  asks what is in any of them.

The rule the whole harness rests on is unchanged: **the log tells the agent about
the protocol, never about the world.**
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from itertools import count
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from craftax_game import CHANNELS, DEFAULT_CHANNELS, VARIANTS, CraftaxGame

LOG = "logs.txt"
STATE = "state.json"
SOCKET = ".act.sock"
# Both live outside the workspace: result.json carries the achievement set, which
# is the answer to the game being played.
DAEMON_LOG = "daemon.log"
RESULT = "result.json"
SEP = "=" * 80

# Random dies every ~250 actions and unions four achievements; a route to iron is a
# few hundred. 3000 is roughly a dozen lives on Craftax — enough for a curve across
# them, and not enough to grind the tree. Classic is smaller in every dimension.
DEFAULT_BUDGET = {"craftax": 3000, "classic": 1500}
# Not an action. Craftax's action space is 43 actions and none of them abandons a
# life, so offering one was an addition to the interface rather than a reading of it —
# and the first real session found what the addition was worth. Because a new life
# re-earns every achievement (the environment's reward is the diff against the
# episode's own achievement vector), `reset left do` paid +1 every three actions
# forever: 715 lives, 226 of them one action long, and a union score that never moved
# off 18. The name stays here because the runs that recorded it still have to be
# readable (tools/readout.py replays their histories).
RESET = "reset"
# A batch may repeat a token: `left*12`. Locomotion is most of what a run spends,
# and a 200-action traverse should not cost 200 tokens of command line.
REPEAT = re.compile(r"^([a-z0-9_]+)\*(\d+)$")
MAX_REPEAT = 500


class ActError(SystemExit):
    """Loud, non-zero exit. The message is the agent's feedback channel."""

    def __init__(self, message: str) -> None:
        super().__init__(f"act: {message}")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class RunState(BaseModel):
    """The record of a run. Not the world — the world is the daemon's CraftaxGame."""

    variant: str = "craftax"
    seed: int = 0
    obs: list[str] = list(DEFAULT_CHANNELS)
    fresh_world: bool = False
    budget: int = 0
    env_dir: str = ""

    actions_used: int = 0
    lives: int = 1
    # Two totals, because they are not the same question. `life_reward` is this life's
    # return, which is the unit the benchmark reports and the one `status` shows. The
    # run total is kept for analysis and not shown: summed across lives it counts the
    # same achievement once per life, and a session shown that number will optimise it
    # (see RESET).
    life_reward: float = 0.0
    reward: float = 0.0
    terminal: bool = False
    outcome: str = ""  # "" | "budget"
    plan: str | None = None
    history: list[str] = []

    # -- the score ----------------------------------------------------------- #
    # Read off the engine after every action and never shown. The achievement names
    # in particular are the tech tree: told that `make_wood_pickaxe` had fired, a
    # session would know both that pickaxes exist and that it had one.
    episodes: list[dict] = []
    achievements: list[str] = []
    score: int = 0
    max_score: int = 0
    unique_cells: int = 0
    max_level: int = 0
    deaths: int = 0

    PRIVATE: ClassVar[set[str]] = {
        "env_dir", "episodes", "achievements", "score", "max_score",
        "unique_cells", "max_level", "deaths", "reward",
    }

    def save(self, ws: Path) -> None:
        tmp = ws / f"{STATE}.tmp"
        tmp.write_text(self.model_dump_json(indent=2, exclude=self.PRIVATE))
        tmp.replace(ws / STATE)

    def record(self) -> None:
        """The full record, including the score, beside the environment.

        Written after every action rather than at the end: a session that is killed
        still leaves a readable result, and the launcher reads this rather than
        state.json for anything the agent must not see.
        """
        if not self.env_dir:
            return
        best = max((e["score"] for e in self.episodes), default=0)
        # The benchmark reports mean return per *episode*, and only a life the world
        # ended is an episode: the last one is usually cut off by the budget, and
        # averaging a truncated life in with finished ones reports a number the
        # comparison cannot carry. How many there were is recorded beside the mean,
        # because on a 3000-action run it is often one or two.
        done = [e["score"] for e in self.episodes if e["ended"]]
        mean = sum(done) / len(done) if done else 0.0
        body = self.model_dump() | {
            "episode_scores": [e["score"] for e in self.episodes],
            "best_episode": best,
            "best_episode_pct": round(100 * best / self.max_score, 2) if self.max_score else 0.0,
            "episodes_completed": len(done),
            "mean_episode": round(mean, 2),
            "mean_episode_pct": round(100 * mean / self.max_score, 2) if self.max_score else 0.0,
            "score_pct": round(100 * self.score / self.max_score, 2) if self.max_score else 0.0,
        }
        Path(self.env_dir, RESULT).write_text(json.dumps(body, indent=2))

    @property
    def actions_left(self) -> int:
        return self.budget - self.actions_used


# --------------------------------------------------------------------------- #
# The log format every parser downstream depends on
# --------------------------------------------------------------------------- #


def wrap(items: tuple[str, ...], width: int = 68, lead: str = "#   ") -> str:
    out, line = [], lead
    for item in items:
        if len(line) + len(item) + 1 > width and line != lead:
            out.append(line.rstrip())
            line = lead
        line += item + " "
    out.append(line.rstrip())
    return "\n".join(out)


def preamble(state: RunState, tokens: tuple[str, ...], channels: dict[str, str]) -> str:
    seen = "\n".join(
        f"#   [{channel}] {Path(name).parent}/NNNNNN{Path(name).suffix} — one per action"
        for channel in channels
        for name in [channels[channel]]
    )
    return (
        f"# logs.txt — the complete record of this run, written by the actuator.\n"
        f"# One block per action: a ruler of = signs, then\n"
        f"#   action N | budget used/total | reward R | <what was played> | step i/k\n"
        f"# then the plan that batch was given, then the observation that action\n"
        f"# produced — one line per channel, each naming a file.\n"
        f"#\n"
        f"# observations:\n{seen}\n"
        f"#   Action 000000 is the state before anything was played. Nothing else\n"
        f"#   about an observation is written here: what is in it is for you to work\n"
        f"#   out.\n"
        f"#\n"
        f"# actions:\n{wrap(tokens)}\n"
        f"#   `./act do left do*4 noop` plays them in order; any token may be\n"
        f"#   repeated with *N. These are all of them, and none of them ends a life:\n"
        f"#   a life ends when the world ends it. A batch stops early if that\n"
        f"#   happens, and the rest of it is dropped — the state you planned\n"
        f"#   against is gone.\n"
        f"# reward: what the action was worth. It is not only good news: some of it\n"
        f"#   is the change in your condition since the last action. `status` totals\n"
        f"#   it for the life you are in, and a new life starts that total again.\n"
        f"# budget: {state.budget} actions for the whole run, across every life.\n"
        f"#   Every action counts. There is one phase and nothing is held back for\n"
        f"#   later.\n"
        f"# [event] lines are things the actuator did that you did not ask for:\n"
        f"#   life over — this life ended and the next one began in the same world.\n"
        f"#     The actions already spent stay spent. That block holds two\n"
        f"#     observations: first the one the action produced, which is the state\n"
        f"#     the life ended in, and then a `-restart` one, the state the next\n"
        f"#     life began in.\n"
        f"#   run over — the budget is spent and nothing further can be played.\n"
    )


def header(state: RunState, reward: float, played: str, index: int, total: int) -> str:
    bits = [
        f"action {state.actions_used}",
        f"budget {state.actions_used}/{state.budget}",
    ]
    if reward:
        bits.append(f"reward {reward:+g}")
    bits.append(played)
    if total:
        bits.append(f"step {index}/{total}")
    return " | ".join(bits)


def ending(alive: bool, actions_left: int) -> str | None:
    """How this action ended the life, or the run, or neither.

    The rules in one place, because the baselines have to play by exactly the ones
    the agent plays by or the floor they measure is a floor for a different game.
    Unlike the humanRL games there is no reward that means victory: a life ends
    because the world ended it, and the run ends only when the budget does.
    """
    if not alive:
        return "restart"
    return "budget" if actions_left <= 0 else None


# --------------------------------------------------------------------------- #
# Parsing actions
# --------------------------------------------------------------------------- #


def parse_tokens(tokens: list[str], playable: tuple[str, ...]) -> list[str]:
    """``left`` ``do*12`` -> a flat list of tokens. Anything else raises.

    Validation happens before a single action reaches the environment, so that a
    typo halfway through a batch of sixty costs nothing and says why.
    """
    out: list[str] = []
    for raw in tokens:
        token, times = raw.strip().lower(), 1
        match = REPEAT.match(token)
        if match:
            token, times = match.group(1), int(match.group(2))
            if not 1 <= times <= MAX_REPEAT:
                raise ActError(f"{raw!r} repeats {times} times — want 1 to {MAX_REPEAT}")
        if token not in playable:
            raise ActError(f"{token!r} is not an action — use {' '.join(playable)}")
        out.extend([token] * times)
    if not out:
        raise ActError("no actions given")
    return out


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class Session:
    """The world, the log and the record — everything the daemon owns."""

    def __init__(self, ws: Path, state: RunState) -> None:
        self.ws = ws
        self.state = state
        self.game = CraftaxGame(
            state.variant,
            seed=state.seed,
            obs=tuple(state.obs),
            fresh_world=state.fresh_world,
        )
        state.max_score = self.game.max_score
        self.last = {}
        self.stopped = False

    @classmethod
    def create(cls, ws: Path, state: RunState) -> "Session":
        session = cls(ws, state)
        session.last = session.game.observe(ws, 0)
        session.measure()
        block = (
            f"{preamble(state, session.game.tokens, session.last)}{SEP}\n"
            f"{header(state, 0.0, 'start', 0, 0)}\n\n"
            f"{session.seen(session.last)}\n"
        )
        (ws / LOG).write_text(block)
        state.save(ws)
        state.record()
        return session

    # -- helpers ------------------------------------------------------------- #

    @staticmethod
    def seen(written: dict[str, str]) -> str:
        return "".join(f"[{channel}] {name}\n" for channel, name in written.items())

    @property
    def playable(self) -> tuple[str, ...]:
        """The game's own actions, and nothing else (see RESET)."""
        return self.game.tokens

    def summary(self) -> str:
        state = self.state
        line = (
            f"budget {state.actions_used}/{state.budget} "
            f"life {state.lives} reward {state.life_reward:+g} this life "
            f"over {state.terminal}"
        )
        if state.terminal:
            line += f" outcome {state.outcome}"
        return line

    def check_budget(self, wanted: int) -> None:
        if self.state.terminal:
            raise ActError("this run is over — nothing left to play")
        if wanted > self.state.actions_left:
            raise ActError(
                f"{wanted} actions requested but only {self.state.actions_left} "
                f"left in the budget"
            )

    def measure(self) -> None:
        """Read the score off the engine. None of this reaches the workspace."""
        state, game = self.state, self.game
        state.episodes = [
            {
                "index": e.index,
                "actions": e.actions,
                "score": e.score,
                "reward": round(e.reward, 3),
                "achievements": list(e.achievements),
                "ended": e.ended,
            }
            for e in game.episodes
        ]
        state.achievements = game.achievements_union()
        state.score = game.score_union()
        state.unique_cells = game.unique_cells
        state.max_level = game.max_level
        state.deaths = game.deaths

    # -- playing ------------------------------------------------------------- #

    def play(self, token: str, index: int = 0, total: int = 0) -> str | None:
        """Send one action, log it, record it. The only path to the environment.

        Returns a reason the batch should stop, or None.
        """
        state = self.state
        reward, alive = self.game.play(token)

        state.actions_used += 1
        state.reward += reward
        state.life_reward += reward
        state.history.append(token)
        self.measure()

        # Written before anything else happens to the world: this is the state the
        # action produced, and for the action that ends a life it is the only view
        # of how that life ended.
        produced = self.game.observe(self.ws, state.actions_used)
        body = [self.seen(produced)]
        self.last = produced

        stop = None
        match ending(alive, state.actions_left):
            case "restart":
                # The life ended and the run carries on into the next one. Two
                # observations, in the order they happened: how it ended, then what
                # it began again in.
                how = self.game.episodes[-1].ended
                self.game.restart()
                state.lives += 1
                state.life_reward = 0.0
                self.measure()
                restarted = self.game.observe(self.ws, state.actions_used, tag="-restart")
                body += [f"[event] life over: {how}\n", self.seen(restarted)]
                self.last = restarted
                stop = f"life over: {how}"
            case "budget":
                state.terminal, state.outcome = True, "budget"
                stop = "budget spent"

        block = [f"\n{SEP}\n{header(state, reward, token, index, total)}\n\n"]
        if state.plan:
            block.append(f"plan: {state.plan}\n\n")
        block.extend(body)
        if state.terminal:
            block.append(f"[event] run over: {state.outcome}\n")
        block.append("\n")
        with (self.ws / LOG).open("a") as log:
            log.write("".join(block))

        state.save(self.ws)
        state.record()
        if stop and total - index:
            stop += f", {total - index} action(s) dropped"
        return stop


# --------------------------------------------------------------------------- #
# Commands, as the daemon runs them
# --------------------------------------------------------------------------- #


def cmd_do(args: argparse.Namespace, s: Session) -> str:
    steps = parse_tokens(args.actions, s.playable)
    s.check_budget(len(steps))
    if args.plan:
        s.state.plan = args.plan

    done, stopped = [], None
    for index, token in enumerate(steps, start=1):
        stopped = s.play(token, index, len(steps))
        done.append(token)
        if stopped:
            break
    out = f"ran {len(done)}/{len(steps)}: {' '.join(done)}\n"
    if stopped:
        out += f"stopped early: {stopped}\n"
    return out + s.summary() + "\n"


def cmd_status(args: argparse.Namespace, s: Session) -> str:
    out = s.summary() + "\n"
    if not s.state.terminal:
        out += f"./act do {' '.join(s.playable)}\n"
    return out


def cmd_board(args: argparse.Namespace, s: Session) -> str:
    """Where the latest observation is. Not what is in it."""
    return f"action {s.state.actions_used}\n{s.seen(s.last)}"


def cmd_stop(args: argparse.Namespace, s: Session) -> str:
    # Unlinked before replying, not on the way out: otherwise the caller returns
    # while the socket is still there and the next command connects to a daemon
    # that is already leaving.
    (s.ws / SOCKET).unlink(missing_ok=True)
    s.stopped = True
    return "stopped\n"


COMMANDS = {
    "do": cmd_do,
    "status": cmd_status,
    "board": cmd_board,
    "stop": cmd_stop,
}


# --------------------------------------------------------------------------- #
# Daemon and client
# --------------------------------------------------------------------------- #


def serve(args: argparse.Namespace) -> int:
    """Own the world for the life of the run and answer commands."""
    ws = Path.cwd()
    # Bound and connected to by its bare name, because a unix socket address is
    # limited to about a hundred bytes and a workspace can be nested deeper than
    # that. Every command runs from the workspace, so the name is enough.
    path = ws / SOCKET
    path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET)
    server.listen(1)

    env_dir = Path(args.env_dir) if args.env_dir else ws / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    try:
        state = RunState(
            variant=args.variant,
            seed=args.seed,
            obs=list(args.obs),
            fresh_world=args.fresh_world,
            budget=args.budget or DEFAULT_BUDGET[args.variant],
            env_dir=str(env_dir),
        )
        # Building the game compiles `step` and the renderer, which is half a
        # minute. It happens here, before the first command is answered, so that a
        # session's first `./act do` is fast rather than looking hung (F5).
        session = Session.create(ws, state)
        print(f"serving a run on {path}", flush=True)
        while not session.stopped:
            conn, _ = server.accept()
            with conn:
                try:
                    payload = json.loads(read_line(conn))
                    command = COMMANDS[payload["command"]]
                    wanted = argparse.Namespace(**payload["args"])
                    reply = {"stdout": command(wanted, session), "rc": 0}
                except SystemExit as exc:
                    reply = {"stdout": f"{exc}\n", "rc": 1}
                except Exception as exc:  # a bug here must not silently strand the agent
                    reply = {"stdout": f"act: {type(exc).__name__}: {exc}\n", "rc": 1}
                try:
                    conn.sendall(json.dumps(reply).encode() + b"\n")
                except OSError as exc:
                    # The client went away before its reply: the agent's shell was
                    # killed, or a long batch outlived whatever was waiting on it.
                    # The actions it asked for are already played and recorded, and
                    # there is nobody left to tell. Losing a reply is not losing the
                    # run, so the loop carries on — before this, one hung-up client
                    # took the daemon down with a BrokenPipeError, the launcher's
                    # shutdown then failed, and the failure discarded the report.
                    print(f"a client hung up before its reply: {exc}", flush=True)
    finally:
        # Whatever happened, leave no socket behind for the next command to trust.
        server.close()
        path.unlink(missing_ok=True)
    return 0


def read_line(conn: socket.socket) -> bytes:
    chunks = []
    while not (chunks and chunks[-1].endswith(b"\n")):
        chunk = conn.recv(1 << 16)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def call(ws: Path, command: str, args: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    if not (ws / SOCKET).exists():
        raise ActError(f"no game running in {ws} — run `act init` first")
    # See serve(): addressed by its bare name, from the workspace.
    target = SOCKET if ws == Path.cwd() else str(ws / SOCKET)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(target)
        conn.sendall(json.dumps({"command": command, "args": args}).encode() + b"\n")
        reply = read_line(conn)
    except OSError as exc:
        raise ActError(f"the environment is not answering: {exc}") from exc
    finally:
        conn.close()
    if not reply:
        raise ActError("the environment closed the connection without answering")
    return json.loads(reply)


# What one attempt leaves behind, and what it is called once another begins. The
# log and the observations are act.py's and the stream is run.py's, but they are
# one attempt and have to be numbered together: rotating only the log let a replay
# overwrite the verdict of the session it was replaying.
#
# `report.json` is deliberately not here, because it is deliberately not in the
# workspace: it carries the achievement set, and a replay is *told* to read what
# its predecessor left behind. The launcher keeps it under the launch's `.rig/`.
ATTEMPT = {
    LOG: "logs-attempt{n}.txt",
    "agent_stream.jsonl": "agent_stream-attempt{n}.jsonl",
    "frames": "frames-attempt{n}",
    "text": "text-attempt{n}",
    "symbolic": "symbolic-attempt{n}",
}


def restart(ws: Path) -> None:
    """Clear the workspace's *world* while leaving its memory alone.

    Notes and helper scripts stay put; the previous attempt is kept beside the new
    one rather than appended to or replaced, because action numbering restarts and
    two runs interleaved in one file would parse as neither. The observations go
    with it for the same reason: frames/000012.png would otherwise mean two states.
    """
    if (ws / SOCKET).exists():
        try:
            call(ws, "stop", {})
        except SystemExit:
            pass
        (ws / SOCKET).unlink(missing_ok=True)
    taken = (n for n in count(1)
             if not any((ws / name.format(n=n)).exists() for name in ATTEMPT.values()))
    number = next(taken)
    for name, pattern in ATTEMPT.items():
        found = ws / name
        if found.exists():
            found.rename(ws / pattern.format(n=number))
    (ws / STATE).unlink(missing_ok=True)


def cmd_init(args: argparse.Namespace) -> int:
    ws = Path.cwd()
    if (ws / STATE).exists():
        if not args.again:
            raise ActError(f"{ws} already holds a game — use a fresh directory, or --again")
        restart(ws)

    argv = [
        sys.executable,
        os.path.abspath(__file__),
        "serve",
        "--variant", args.variant,
        "--seed", str(args.seed),
        "--budget", str(args.budget or DEFAULT_BUDGET[args.variant]),
        "--obs", *args.obs,
    ]
    if args.fresh_world:
        argv.append("--fresh-world")
    env_dir = Path(args.env_dir or tempfile.mkdtemp(prefix="act-env-"))
    # mkdtemp makes its own; a directory named by the launcher may not exist yet,
    # and the daemon log below is opened before the daemon runs.
    env_dir.mkdir(parents=True, exist_ok=True)
    argv += ["--env-dir", str(env_dir)]
    with (env_dir / DAEMON_LOG).open("w") as daemon_log:
        proc = subprocess.Popen(
            argv,
            cwd=ws,
            stdin=subprocess.DEVNULL,
            stdout=daemon_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def died(problem: str) -> ActError:
        return ActError(f"{problem}:\n{(env_dir / DAEMON_LOG).read_text()[-2000:]}")

    deadline = time.monotonic() + args.start_timeout
    while not (ws / SOCKET).exists():
        if proc.poll() is not None:
            raise died("the environment failed to start")
        if time.monotonic() > deadline:
            proc.kill()
            raise ActError(f"the environment did not start within {args.start_timeout}s")
        time.sleep(0.05)

    # The socket is bound before the world is built, so this first command is what
    # waits for it — and what surfaces a failure while building it.
    try:
        reply = call(ws, "status", {})
    except SystemExit:
        raise died("the environment failed to start") from None
    sys.stdout.write("ready\n" + reply["stdout"])
    reply = call(ws, "board", {})
    sys.stdout.write(reply["stdout"])
    return int(reply["rc"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="act", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_env_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--variant", choices=VARIANTS, default="craftax")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--budget", type=int, default=0,
                       help=f"default {DEFAULT_BUDGET} by variant")
        p.add_argument("--obs", nargs="+", choices=CHANNELS, default=list(DEFAULT_CHANNELS),
                       help="which of the package's own observations to write")
        p.add_argument("--fresh-world", action="store_true",
                       help="deal a new world on each life instead of replaying this one")
        p.add_argument(
            "--env-dir", default=None,
            help="where the environment's log and the result record go, outside the "
                 "workspace",
        )

    p_init = sub.add_parser("init", help="start a run in this directory")
    add_env_flags(p_init)
    p_init.add_argument("--again", action="store_true", help="replay, keeping the notes")
    p_init.add_argument("--start-timeout", type=float, default=300.0)

    p_serve = sub.add_parser("serve", help=argparse.SUPPRESS)
    add_env_flags(p_serve)

    p_do = sub.add_parser("do", help="play actions, in order, until this life ends")
    p_do.add_argument("actions", nargs="+", metavar="ACTION", help="e.g. left do*12 noop")
    p_do.add_argument("--plan", help="briefing recorded in the log beside these actions")

    sub.add_parser("status", help="one-line summary and what is playable now")
    sub.add_parser("board", help="where the latest observation is")
    sub.add_parser("stop", help="shut the environment down")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return serve(args)
    if args.command == "init":
        return cmd_init(args)
    values = {k: v for k, v in vars(args).items() if k != "command"}
    reply = call(Path.cwd(), args.command, values)
    sys.stdout.write(reply["stdout"])
    return int(reply["rc"])


if __name__ == "__main__":
    sys.exit(main())
