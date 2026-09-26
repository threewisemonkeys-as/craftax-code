#!/usr/bin/env python3
"""Keep the replay page level with runs that are still playing.

    .env-venv/bin/python tools/watch_replay.py \
        ~/craftax-runs/20260909-053629 ~/craftax-runs/20260922-144509 \
        --out ../replay.html --interval 1800

`tools/replay.py` reads only files the actuator has finished writing, and rebuilding
it is cheap twice over — a frame is encoded once and kept, and the world replay is
cached against the length of the history it was computed from — so the page can
simply be built again every so often. This is the loop that does it, and there are
two things it has to get right: when not to build, and when to stop.

**When not to build.** A scan that finds nothing new costs three stats a workspace; a
build costs whatever has arrived since the last one. So each world is fingerprinted
by the things that only ever grow — the size of its log, the size of its stream, the
actions its record says were played — and a cycle where every fingerprint is
unchanged does nothing at all. That is the common case overnight, when a session has
died and the next has not started yet.

**When to stop**, which is the part that is easy to get wrong. Not from the
launcher's pid: a run outlives its launch here, and a 30,000-action run is continued
more than once on the way, so a watcher keyed to the process that started it would
quit at the first join between two launches. Not from `pgrep -f` either — a pattern
naming the run also matches the shell that greps for it, which on this project has
reported a finished run alive and a live one dead (see the same lesson recorded in
offline_learning/launch/watch_agent_replay.py).

The record says it: a run is over when every world in it has played its budget. Two
guards stand behind that, because a run can stop without finishing — `--max-hours`
bounds the whole watch, and `--idle-hours` ends it when nothing has moved for long
enough that nothing is going to. All three say which one it was on the way out,
because a page that stopped being rebuilt looks exactly like a run that stopped
making progress, and the difference is the whole question.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

import run  # noqa: E402

REPLAY = HERE / "replay.py"


def log(line: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {line}", flush=True)


def worlds(root: Path) -> list[str]:
    """The labels of a launch, or none if it has not written them yet."""
    try:
        return list(json.loads((root / run.RIG / run.LABELS).read_text()))
    except (OSError, json.JSONDecodeError):
        return []


def record(root: Path, label: str) -> dict:
    """What the actuator has written down for this world.

    Read forgivingly. The file is rewritten after every action and the write is
    atomic, but this loop does not own it, and a launch caught between `mkdir` and
    its first action has no file at all.
    """
    try:
        return json.loads((root / ".envs" / label / run.RESULT).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def fingerprint(roots: list[Path]) -> dict[str, tuple]:
    """Every world being watched, by the things about it that only grow.

    Not the frame count, which would mean listing thirty thousand files a cycle to
    learn what the log's own length already says.
    """
    out = {}
    for root in roots:
        for label in worlds(root):
            ws = root / label
            got = record(root, label)
            out[f"{root.name}/{label}"] = (
                size(ws / "logs.txt"),
                size(ws / "agent_stream.jsonl"),
                int(got.get("actions_used") or 0),
                # A world-model arm's learning: a pause begins an update and a line
                # in the updates log ends one, and neither moves the three above.
                len(got.get("pauses") or []),
                size(root / run.RIG / "updates" / f"{label}.jsonl"),
            )
    return out


def finished(roots: list[Path]) -> bool:
    """Has every world played the budget it was given?

    A world with no record yet is not finished — it is a launch that has not started,
    which is the one case where believing the absence would stop the watch before the
    run it is watching had begun.
    """
    for root in roots:
        for label in worlds(root):
            got = record(root, label)
            if int(got.get("actions_used") or 0) < int(got.get("budget") or 1):
                return False
    return True


def rebuild(roots: list[Path], out: Path, replay_world: bool) -> bool:
    """Build the page, and put what it said in this log rather than swallowing it."""
    cmd = [sys.executable, str(REPLAY), *(str(root) for root in roots),
           "--out", str(out)]
    if not replay_world:
        cmd.append("--no-replay")
    try:
        # Generous: the first build of a finished 30,000-action arm replays the whole
        # world, and every frame of it has still to be encoded.
        done = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=7200, cwd=ROOT)
    except subprocess.SubprocessError as exc:
        log(f"  build failed: {exc}")
        return False
    if done.returncode != 0:
        log(f"  build exited {done.returncode}: {' '.join(done.stderr.split())[-400:]}")
        return False
    for line in done.stdout.strip().splitlines():
        if line.strip():
            log(f"  {line}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("launch", nargs="+",
                        help="the launch directories the page is built from")
    parser.add_argument("--out", default="../replay.html")
    parser.add_argument("--interval", type=float, default=1800,
                        help="seconds between scans (default 1800)")
    parser.add_argument("--max-hours", type=float, default=72.0)
    parser.add_argument("--idle-hours", type=float, default=6.0,
                        help="stop when no world has moved for this long")
    parser.add_argument("--no-replay", action="store_true",
                        help="build without the score and condition tracks")
    args = parser.parse_args()

    roots = [Path(one).expanduser().resolve() for one in args.launch]
    for root in roots:
        if not (root / run.RIG / run.LABELS).exists():
            raise SystemExit(f"watch: {root} is not a launch directory")
    out = Path(args.out).expanduser().resolve()

    log(f"watching {len(roots)} launch(es) -> {out}, every {args.interval / 60:.0f} min")
    started = moved = time.time()
    was: dict[str, tuple] = {}
    while True:
        now = fingerprint(roots)
        if now != was:
            grew = [k for k, v in now.items() if was.get(k) != v]
            log(f"building — {', '.join(grew)} moved"
                if was else "building the first page")
            rebuild(roots, out, not args.no_replay)
            was, moved = now, time.time()

        # After the build and not before it, so the last page of a run is the one
        # that has its last action on it.
        if finished(roots):
            log("every world has played its budget — stopping")
            return 0
        idle, ran = time.time() - moved, time.time() - started
        if idle > args.idle_hours * 3600:
            log(f"nothing has moved in {idle / 3600:.1f}h — stopping. The page is "
                f"current as of the last build; the run is not finished.")
            return 1
        if ran > args.max_hours * 3600:
            log(f"watched for {ran / 3600:.1f}h — stopping. The page is current as "
                f"of the last build; the run is not finished.")
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
