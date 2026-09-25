#!/usr/bin/env python3
"""What each session saw, what it played, and what it did between the two.

    .env-venv/bin/python tools/replay.py ~/craftax-runs/20260909-053629 \
        ~/craftax-runs/20260922-144509 --out replay.html

Adapted from cc_humanrl's tools/replay.py. The shape is the same — pick a run, scrub
the trajectory, and at every action the page holds the plan that batch was given, the
actions in it, and the work the session did in its workspace before playing it — and
two things about Craftax force the rest to differ.

**The pictures cannot be inlined.** MonsterKong's camera is fixed, so an action moves
a sprite and little else: 212 pixels change on the worst game, and the whole launch
fits in one file as edits against a base frame. Craftax centres the view on the player,
so a step scrolls everything — a measured **292,365 of 585,728 pixels change per
action** — and a 3000-action run is 255 MB of PNG. Nothing recovers that. Sheeting the
frames does not help (WebP has no inter-frame prediction) and a tile atlas does not
either, because the renderer multiplies every tile by a light map and adds per-frame
noise at night, so 430 frames already hold 6017 distinct 64x64 tiles.

So there are two modes, and the default is the honest one:

* **A directory** (default). `replay.html` beside `replay_frames/<label>/`, one WebP per
  action at half size. Every frame, good quality, loaded as the cursor reaches it.
  `<img>` works from `file://` where `fetch` does not, which is what this has to run on.
* **`--inline`**: one self-contained file, publishable, at the cost of some frames. The
  *timeline stays complete* — every action, plan, command and result is still there —
  and only the pictures thin out, never at a moment that mattered: anything that paid a
  reward, unlocked an achievement, changed level, ended a life or began a batch is kept
  whole, and only runs of plain movement are sampled down to fit the budget.

**The score is replayed, not read.** `result.json` is rewritten in place after every
action, so nothing on disk says what the score was at action 700. `tools/readout.py`
replays the recorded history through a fresh engine, which is bit-exact in seed and
actions, and that gives the achievement timeline, the level, and the health/food/drink
the session was carrying. It costs three minutes of JAX for thirty thousand actions
and is worth it; `--no-replay` skips it and the page loses those tracks.

**Several launches, one page.** An arm is a launch directory, and the comparison this
page exists for — one world, two CLIs — is two of them, so `launch` takes as many as
you have. Nothing has to be disambiguated: a label is drawn per launch from a space
big enough that two arms do not collide, and it is what keys the reports, the frame
directories and the matrix alike.

**A run that is still playing** can be read the whole time, and the page says which
rows are still moving rather than showing a prefix as if it were a result. Two things
make rebuilding cheap enough to do on a timer — which is what `tools/watch_replay.py`
does: a frame is written once and never edited, so only the ones that arrived since
the last build are encoded (all thirty thousand is twenty-three minutes; the two
hundred since the last build is ten seconds), and the world replay is cached against
the length of the history it was computed from, so a finished arm is replayed once
and never again.

**Reasoning.** The CLI returns thinking blocks with their content encrypted — all 453
in this launch are an empty string beside a signature, against 303,101 thinking tokens
the result event reports — so the page cannot show what a session thought and says so
rather than leaving a gap that reads like silence. What it can show is what the session
*wrote down*: the `--plan` it had to state to move at all, its shell, its scripts, their
output and its notes.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

import run  # noqa: E402

BODY = ROOT / "tools" / "replay_body.html"
CSS = ROOT / "tools" / "replay.css"
# The six human trajectories the Craftax README links, cached as (reward, done) by
# tools/humans.py. Optional: without them the page simply has no human backdrop.
HUMAN = Path(os.environ.get("CRAFTAX_HUMAN") or Path.home() / "craftax-human")

# Published results, mean episode return as a percentage of the 226. Craftax-1B is a
# billion environment steps of training, Craftax-1M a million; both are *training*
# budgets, and both are measured on Craftax-**Symbolic** — the paper limits its
# benchmarks to symbolic observations, so a pixels run is a different condition and
# the table says so rather than quietly comparing across it.
BASELINES = [
    ("PPO-GTrXL", 18.3, "1B"),
    ("PQN-RNN", 16.0, "1B"),
    ("PPO-RNN", 15.3, "1B"),
    ("RND", 12.0, "1B"),
    ("PPO", 11.9, "1B"),
    ("Simulus", 6.6, "1M"),
    ("Efficient MBRL", 5.4, "1M"),
    ("PPO-RNN", 2.3, "1M"),
]

BLOCK = re.compile(r"^action (\d+) \| budget (\d+)/(\d+)(.*)$")
# The actuator writes one line per channel, each naming a file: `[pixels] frames/...`.
SEEN = re.compile(r"^\[(pixels|text|symbolic)\] (\S+)$", re.M)
EVENT = re.compile(r"^\[event\] (.+)$", re.M)
PLAN = re.compile(r"^plan: (.*)$", re.M | re.S)
STEP = re.compile(r"step (\d+)/(\d+)")
# The actuator writes one of these where a session's stint ended and the next rebuilt
# the world. It is not an action: it repeats the action number the run stopped at, so
# left in the stream it becomes a second action 6000 whose played token is "session 3".
HANDOVER = re.compile(r"^session (\d+)$")
REWARD = re.compile(r"reward ([-+][\d.]+)")
BUDGET = re.compile(r"budget (\d+)/(\d+)")
IS_FRAME = re.compile(r"frames/(\d{6})(-restart)?\.png")

# A tool result can be a 200-line dump. Generous, because the point of the panel is to
# show the actual work, and bounded, because a 529-turn session of it has to fit.
CAP = 4000
# Half size: 352x416, which keeps a 64px tile at 32px — every block still legible.
SCALE = 2
QUALITY = 72
# `--inline` is the make-it-fit build, so it spends its budget on *more of the run*
# rather than on a sharper picture of less of it: a third size still puts a block at
# 21px, and at 13 KB a frame the must-keeps alone — every batch start, every
# achievement, every descent and death — came to 9.1 MB before this.
INLINE_SCALE = 3
INLINE_QUALITY = 62
# What `--inline` may spend on pictures, before base64 inflates it by a third and the
# text of the run — a megabyte and a half of plans, commands and output — is added on
# top. Set to land near 12 MB against a 16 MB page limit, because the estimate below is
# a mean over samples and a run can be denser than its samples.
INLINE_BYTES = 7_000_000


# --------------------------------------------------------------------------- #
# The log: what was played, in what batch, under what plan
# --------------------------------------------------------------------------- #
def read_log(path: Path) -> list[dict]:
    """One record per action block, in order, with the observations it produced.

    The log is the actuator's own writing and is the authority on what happened: the
    agent's stream says what was *asked* for, and a batch stops early when a life ends.
    """
    blocks = []
    for chunk in path.read_text(errors="replace").split("\n" + "=" * 80 + "\n")[1:]:
        head, _, rest = chunk.partition("\n")
        found = BLOCK.match(head.strip())
        if not found:
            continue
        step = STEP.search(found[4])
        got = REWARD.search(found[4])
        # The played token is the last `|` field that is not budget/reward/step.
        fields = [f.strip() for f in found[4].split("|") if f.strip()]
        played = next((f for f in fields if not f.startswith(("reward", "step"))), "start")
        plan = PLAN.search(rest)
        seen = SEEN.findall(rest)
        handover = HANDOVER.match(played)
        blocks.append({
            "n": int(found[1]),
            "used": int(found[2]),
            "budget": int(found[3]),
            "tok": played,
            "kind": "handover" if handover else "action",
            "session": int(handover[1]) if handover else 0,
            "i": int(step[1]) if step else 0,
            "k": int(step[2]) if step else 0,
            "reward": float(got[1]) if got else 0.0,
            "plan": plan[1].strip().split("\n\n")[0].strip() if plan else "",
            "frames": [name for channel, name in seen if channel == "pixels"],
            "events": EVENT.findall(rest),
        })
    return blocks


def batches(blocks: list[dict]) -> list[dict]:
    """The batches the actuator was given, read off the log rather than the stream.

    One `./act do` plays several actions, and a session is free to wrap that call in a
    script of its own — this one wrote a `go.py` and played 2568 of its 3066 actions
    through it, so cutting the trajectory at commands that *look* like `./act do` would
    miss most of them. The log's own `step i/k` counter says where a batch begins and
    ends whoever asked for it.
    """
    out: list[dict] = []
    for block in blocks:
        if block["n"] == 0:  # the state before anything was played
            continue
        if block["kind"] == "handover":  # a session boundary, not an action
            continue
        cont = (
            out
            and block["k"]
            and block["k"] == out[-1]["asked"]
            and block["i"] == out[-1]["last"] + 1
        )
        if cont:
            batch = out[-1]
            batch["to"], batch["last"] = block["n"], block["i"]
            batch["toks"].append(block["tok"])
            batch["plan"] = batch["plan"] or block["plan"]
        else:
            out.append({
                "from": block["n"] - 1, "to": block["n"],
                "asked": block["k"], "last": block["i"],
                "toks": [block["tok"]], "plan": block["plan"],
                "cmd": "", "why": "", "out": "", "work": [],
            })
    for batch in out:
        batch["played"] = len(batch["toks"])
    return out


# --------------------------------------------------------------------------- #
# The stream: what the session did between batches
# --------------------------------------------------------------------------- #
def moments(stream: Path, agent: str, ws: Path) -> list[dict]:
    """The session's turns, flattened into what it said, ran and was told back."""
    if not stream.exists():
        return []
    out: list[dict] = []
    pending: dict[str, dict] = {}
    for line in stream.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if agent == "codex":
            item = event.get("item", {})
            if event.get("type") != "item.completed":
                continue
            if item.get("type") == "agent_message":
                out.append({"kind": "say", "text": str(item.get("text", ""))})
            elif item.get("type") == "command_execution":
                out.append({
                    "kind": "tool", "name": "Bash", "title": str(item.get("command", "")),
                    "body": "", "out": str(item.get("aggregated_output", ""))[:CAP],
                    "images": 0,
                })
            elif item.get("type") in ("file_change", "patch_apply"):
                out.append({"kind": "tool", "name": "Edit", "title": "",
                            "body": str(item)[:CAP], "out": "", "images": 0})
            continue

        kind = event.get("type")
        if kind == "assistant":
            for block in event.get("message", {}).get("content", []):
                shape = block.get("type")
                if shape == "text" and block.get("text", "").strip():
                    out.append({"kind": "say", "text": block["text"]})
                elif shape == "thinking":
                    # Content arrives encrypted — an empty string beside a signature —
                    # so this records that the session reasoned here and claims nothing
                    # about what it reasoned.
                    out.append({"kind": "think", "sealed": not (block.get("thinking") or "").strip(),
                                "text": block.get("thinking", "")})
                elif shape == "tool_use":
                    got = block.get("input", {}) or {}
                    moment = {
                        "kind": "tool",
                        "name": block.get("name", "?"),
                        "title": str(got.get("command") or got.get("file_path")
                                     or got.get("pattern") or got.get("path") or ""),
                        "why": str(got.get("description", "")),
                        "body": str(got.get("content", "") or got.get("new_string", "")),
                        "out": "", "images": 0,
                    }
                    out.append(moment)
                    if block.get("id"):
                        pending[block["id"]] = moment
        elif kind == "user":
            for block in event.get("message", {}).get("content", []) or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                body, images = block.get("content", ""), 0
                if isinstance(body, list):
                    images = sum(1 for p in body
                                 if isinstance(p, dict) and p.get("type") == "image")
                    body = "\n".join(str(p.get("text", "")) for p in body
                                     if isinstance(p, dict))
                target = pending.get(block.get("tool_use_id", ""))
                if target is None:
                    continue
                target["out"] = str(body)[:CAP]
                target["images"] = images
                target["error"] = bool(block.get("is_error"))
    # Sessions write absolute paths, and every one starts with the same forty characters
    # of workspace. Trimmed in the label only; a file's contents stay verbatim.
    here = str(ws) + "/"
    for moment in out:
        if moment["kind"] == "tool":
            moment["title"] = moment["title"][:CAP].replace(here, "./")
            moment["body"] = moment["body"][:CAP]
            moment["out"] = moment["out"].replace(here, "./")
    return out


def assign(made: list[dict], said: list[dict]) -> None:
    """Hand each batch the call that played it and the work that preceded it.

    A call played if the budget it reported back went up — true of `./act do` and
    equally of anything that ran it, so this does not care how the session chose to
    drive. Everything between two such calls is work done in the workspace for free, and
    belongs to the batch it was preparing.
    """
    cursor, gap, at = 0, [], 0
    for moment in said:
        if moment["kind"] != "tool":
            gap.append(moment)
            continue
        seen = [int(n) for n, _ in BUDGET.findall(moment.get("out", ""))]
        end = max(seen, default=cursor)
        if end <= cursor:
            gap.append(moment)
            continue
        first = True
        while at < len(made) and made[at]["to"] <= end:
            made[at].update(cmd=moment["title"], why=moment.get("why", ""),
                            out=moment["out"])
            if first:
                made[at]["work"], first = gap, False
            at += 1
        gap, cursor = [], end
    if gap and made:
        # The last thing a session does is write up what it learned; there is no batch
        # after it, so it hangs off the one it followed.
        (made[at] if at < len(made) else made[-1]).setdefault("work", []).extend(gap)


def attribute(said: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    """Which frames a session put in front of its own eyes.

    Reading the screen is a legitimate channel — it is the channel `play_craftax` gives
    a person — so it is counted rather than fenced off. But a session rarely reads a
    frame as it is: it crops or magnifies one first, and what enters context is
    `/tmp/tile.png`. So a read of a file that is not a frame is traced back to the
    command that made it, and those guesses are kept in a separate count from the reads
    that are certain.
    """
    looked: dict[str, int] = {}
    derived: dict[str, int] = {}
    makers: list[tuple[str, list[str]]] = []
    for moment in said:
        if moment["kind"] != "tool":
            continue
        text = moment.get("title", "") + "\n" + moment.get("body", "")
        seen = [m.group(0) for m in IS_FRAME.finditer(text)]
        if not moment.get("images"):
            if seen and ".png" in text.replace("frames/", ""):
                makers.append((text, seen))
            continue
        hit = IS_FRAME.search(moment.get("title", ""))
        if hit:
            looked[hit.group(0)] = looked.get(hit.group(0), 0) + 1
            continue
        path = moment.get("title", "").strip()
        source = next((frames for text, frames in reversed(makers) if path and path in text), None)
        if source is None:
            source = makers[-1][1] if makers else []
        if len(source) > 1:
            nums = {int(n) for n in re.findall(r"\d+", Path(path).name)}
            narrowed = [f for f in source if int(IS_FRAME.search(f)[1]) in nums]
            source = narrowed or source
        for name in source:
            derived[name] = derived.get(name, 0) + 1
    return looked, derived


# --------------------------------------------------------------------------- #
# The world, replayed: what the score was at every action
# --------------------------------------------------------------------------- #
def trace(record: dict, root: Path, label: str) -> list[dict]:
    """The per-action score, level and condition, from `tools/readout.py`.

    Nothing on disk holds this: `result.json` is rewritten in place after every action.
    A world is bit-exact in its seed and its actions, so playing the recorded history
    again recovers it.

    Cached beside the run, because it costs three minutes for thirty thousand actions
    and a page rebuilt on a timer would otherwise spend that on a *finished* arm every
    time it caught the live one up. The key is how long the history was rather than
    when the file changed: a history is only ever appended to, so a cache taken at the
    same length is the same trajectory — and the one thing that makes a history
    shorter, `--replay` dealing the world again, misses the key and is recomputed.
    """
    from readout import replay  # noqa: PLC0415

    played = len(record["history"])
    cache = root / run.RIG / "traces" / f"{label}.json"
    try:
        was = json.loads(cache.read_text())
        if was["played"] == played:
            return was["rows"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass  # unreadable is the same as absent — replay it and write it again
    rows = replay(record)
    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_suffix(".part")
    part.write_text(json.dumps({"played": played, "rows": rows}, separators=(",", ":")))
    part.rename(cache)
    return rows


# --------------------------------------------------------------------------- #
# Where a run sits: against the humans, and against the published RL
# --------------------------------------------------------------------------- #
def best_curve(rewards, ends) -> list[float]:
    """The best single-life return reached by step N.

    Monotonic, and it is the quantity the paper reports — mean *episode* return. The
    obvious alternative, cumulative return over the whole session, cannot be compared
    across runs that died different numbers of times: a new life re-earns every
    achievement (F12), so our seven-life pilot sums to 224 of 226 while its best life
    is 40. The same inflation is in the human table in the plan's F10, mildly: run5's
    session sums to 31.9% where its best life is 22.6%.
    """
    best = current = 0.0
    out = []
    for reward, ended in zip(rewards, ends, strict=True):
        current += float(reward)
        best = max(best, current)
        out.append(round(best, 2))
        if ended:
            current = 0.0
    return out


def thin(curve: list[float], points: int = 320) -> list[list[float]]:
    """Evenly spaced (step, score) pairs, for the chart's linear step axis.

    Every point where the curve *moves* is kept whatever the spacing says: those are
    the achievements, and they are the whole shape of the line.
    """
    if not curve:
        return []
    n = len(curve)
    wanted = {0, n - 1}
    wanted |= {i for i in range(1, n) if curve[i] != curve[i - 1]}
    wanted |= {min(n - 1, (n * k) // points) for k in range(points)}
    return [[i + 1, curve[i]] for i in sorted(wanted)]


def thousands(n: int) -> str:
    """3000 as `3k`, 2500 as `2.5k`, 800 as `800`."""
    return f"{n / 1000:g}k" if n >= 1000 else str(n)


def humans() -> list[dict]:
    """The six human runs, if the cache is on this disk."""
    import numpy as np  # noqa: PLC0415

    out = []
    for path in sorted(HUMAN.glob("cache_run*.npz")):
        data = np.load(path)
        curve = best_curve(list(data["reward"]), list(data["done"]))
        out.append({
            "name": path.stem.replace("cache_", ""),
            "steps": len(curve),
            "final": curve[-1],
            "curve": thin(curve),
        })
    return sorted(out, key=lambda h: -h["final"])


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def shrink(path: Path, scale: int = SCALE, quality: int = QUALITY) -> bytes:
    from PIL import Image  # noqa: PLC0415

    im = Image.open(path).convert("RGB")
    im = im.resize((im.width // scale, im.height // scale), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=4)
    return buf.getvalue()


def keep(frames: list[dict], budget: int, sample: int) -> list[int]:
    """Which frames `--inline` can afford, keeping every one that mattered.

    Anything that paid, unlocked, descended, ended a life or began a batch is kept
    whole. What is left is plain movement, and it is thinned evenly until the pictures
    fit. The timeline itself never thins — every action is still on the track, and one
    without a picture of its own shows the last one there was, marked as such.
    """
    must = {
        i for i, f in enumerate(frames)
        if i in (0, len(frames) - 1)
        or f["reward"] or f["fired"] or f["events"] or f["kind"] != "play"
        or f["i"] == 1 or f.get("dlevel")
    }
    room = max(0, budget // max(1, sample) - len(must))
    rest = [i for i in range(len(frames)) if i not in must]
    if room >= len(rest):
        return list(range(len(frames)))
    stride = len(rest) / room if room else 0
    thinned = {rest[int(k * stride)] for k in range(room)} if room else set()
    return sorted(must | thinned)


def pictures(ws: Path, frames: list[dict], out: Path, label: str, inline: bool) -> dict:
    """Write the frames, and say where they went.

    Returns what the page needs: the size to draw at, and either a directory to load
    from or the bytes themselves.
    """
    names = [f["name"] for f in frames]
    from PIL import Image  # noqa: PLC0415

    scale = INLINE_SCALE if inline else SCALE
    quality = INLINE_QUALITY if inline else QUALITY
    with Image.open(ws / names[0]) as probe:
        w, h = probe.width // scale, probe.height // scale

    if not inline:
        where = out.parent / f"{out.stem}_frames" / label
        where.mkdir(parents=True, exist_ok=True)
        # Only what is not already there, which is what lets a page be rebuilt while
        # the run is still playing: at 45 ms a frame, all thirty thousand is
        # twenty-three minutes and the two hundred that arrived since the last build
        # is ten seconds.
        #
        # The test is the source's own mtime rather than the file merely existing,
        # because a label outlives the world it named: `--replay` deals a new world
        # into the same workspace under the same label, and every frame of it is
        # written over a frame of the old one. Newer source, re-encode — so the one
        # case where the pictures on disk are a lie about the run is also the one
        # case this notices.
        made = 0
        for name in names:
            target = where / (Path(name).stem + ".webp")
            source = ws / name
            if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
                continue
            # Under a temporary name and renamed, because a build killed part-way
            # through an encode would otherwise leave a half-frame that the next
            # build, finding it present and newer, would keep forever.
            part = target.with_suffix(".part")
            part.write_bytes(shrink(source))
            part.rename(target)
            made += 1
        print(f"replay: {label} encoded {made} new frames, {len(names)} on the page")
        return {"w": w, "h": h, "dir": f"{out.stem}_frames/{label}",
                "shown": list(range(len(frames))), "bytes": 0}

    # Over several frames, not one: a cave frame and a daylit forest frame differ by
    # more than a factor of two, and sizing the whole run off whichever happened to sit
    # at the midpoint was how the first build came out at 13.7 MB.
    probes = [len(shrink(ws / names[i], scale, quality))
              for i in range(0, len(names), max(1, len(names) // 8))]
    sample = sum(probes) // len(probes)
    chosen = keep(frames, INLINE_BYTES, sample)
    blobs, total = {}, 0
    for i in chosen:
        data = shrink(ws / names[i], scale, quality)
        total += len(data)
        blobs[str(i)] = base64.b64encode(data).decode()
    return {"w": w, "h": h, "dir": "", "shown": chosen, "img": blobs, "bytes": total}


# --------------------------------------------------------------------------- #
# Building the page
# --------------------------------------------------------------------------- #
def agent_of(stream: Path) -> str:
    """Which CLI played this workspace, read off its own stream.

    Not from the report, which is where this used to come from, because a report is
    written when a session *ends* — so a run being watched while it plays has none
    until the first stint hands over, and the default was the other CLI. The symptom
    was silent and total: a live Codex run built a page with every frame and every
    action on it and not one plan, command or line of output, because every event in
    its stream had been read for a shape it does not have.
    """
    if not stream.exists():
        return "claude"
    from readout import whose  # noqa: PLC0415

    return whose(stream.read_text(errors="replace").splitlines())


def model_of(root: Path, label: str, agent: str, report: dict | None) -> str:
    """Which model played this workspace, from the first record that says.

    Two runs of one CLI on one world are the same row on this page unless something
    tells them apart, and what does is the model. The launcher writes it down before
    the first session (`run.write_arm`); a run launched before that existed has only
    its report, once the launch is over. Before that the CLI's own records do: Codex writes the model into every
    session it keeps under the config directory the launcher gave it, and Claude
    names it in its stream's `init` event. The adapter's default is the last resort,
    and is only a guess about a run that was launched with `--model`.
    """
    arm = root / run.RIG / "arms" / f"{label}.json"
    if arm.exists():
        return str(json.loads(arm.read_text()).get("model") or run.AGENTS[agent].model)
    if report and report.get("model"):
        return str(report["model"])
    kept = root / ".sessions" / label / run.AGENTS[agent].CONFIG_DIR / "sessions"
    for rollout in sorted(kept.glob("*/*/*/rollout-*.jsonl")):
        with rollout.open(errors="replace") as lines:
            for line in lines:
                if found := re.search(r'"model":"([^"]+)"', line):
                    return found[1]
    stream = root / label / "agent_stream.jsonl"
    if stream.exists():
        with stream.open(errors="replace") as lines:
            for line in lines:
                if '"subtype":"init"' in line.replace(" ", ""):
                    try:
                        if found := json.loads(line).get("model"):
                            return str(found)
                    except json.JSONDecodeError:
                        pass
                    break
    return run.AGENTS[agent].model


def fenced_of(root: Path, label: str, agent: str, report: dict | None) -> bool:
    """Whether this run's sessions were fenced: the launcher's record of the arm, then
    the report, then the adapter's default — which is only right for a run launched
    without `--fence` or `--no-fence`, and is why the record exists."""
    arm = root / run.RIG / "arms" / f"{label}.json"
    if arm.exists():
        return bool(json.loads(arm.read_text()).get("fenced"))
    return bool((report or {}).get("fenced", run.AGENTS[agent].FENCED))


def one_launch(root: Path, out: Path, inline: bool, replay_world: bool,
               only: str = "") -> list[dict]:
    labels = json.loads((root / run.RIG / run.LABELS).read_text())
    if only:
        # Empty rather than a KeyError: with several launches on the page, a label
        # names a workspace in exactly one of them and the others have nothing to
        # contribute. `main` has already checked that some launch knows it.
        labels = {only: labels[only]} if only in labels else {}
    reports = {
        path.stem: json.loads(path.read_text())
        for path in sorted((root / run.RIG / "reports").glob("*.json"))
    }
    runs = []
    for label, meta in labels.items():
        report = reports.get(label)
        ws = root / label
        if not (ws / "logs.txt").exists():
            print(f"replay: {label} has no log — skipped")
            continue
        blocks = read_log(ws / "logs.txt")
        if not blocks:
            print(f"replay: {label} logged no actions — skipped")
            continue
        agent = agent_of(ws / "agent_stream.jsonl")
        said = moments(ws / "agent_stream.jsonl", agent, ws)
        made = batches(blocks)
        assign(made, said)

        record = json.loads((root / ".envs" / label / "result.json").read_text())
        rows = trace(record, root, label) if replay_world else []
        by_action = {r["action"]: r for r in rows}

        # Every frame the log names, in order. Longer than the action count: the block
        # that ends a life holds two — the state it ended in, and what it began again in.
        # Where one session stopped and the next took over the same run. Kept as
        # marks on the transport rather than frames: the observation a handover
        # writes is the world *unmoved* — the next session rebuilt the state the last
        # action left — so it is pixel-for-pixel the frame before it, and putting it
        # in the timeline would be a duplicate image standing for nothing new.
        handovers = [
            {"n": block["n"], "session": block["session"],
             "note": " ".join(" ".join(block["events"]).split())[:400]}
            for block in blocks if block["kind"] == "handover"
        ]

        frames = []
        for block in blocks:
            if block["kind"] == "handover":
                continue
            for i, name in enumerate(block["frames"]):
                row = by_action.get(block["n"], {})
                frames.append({
                    "n": block["n"], "name": name, "tok": block["tok"],
                    "i": block["i"], "k": block["k"], "used": block["used"],
                    "reward": block["reward"],
                    "kind": ("restart" if name.endswith("-restart.png")
                             else "start" if block["n"] == 0 else "play"),
                    "events": block["events"] if i == len(block["frames"]) - 1 else [],
                    "score": row.get("score", 0), "life": row.get("life", 1),
                    "fired": row.get("fired", ""), "level": row.get("level", 0),
                    "hp": row.get("health", 0), "food": row.get("food", 0),
                    "drink": row.get("drink", 0), "energy": row.get("energy", 0),
                })
        # A page built while the world is being played can be one picture ahead of
        # the disk: the log names a frame in the same breath as the actuator writes
        # it. Dropped from the end rather than tolerated, because everything after
        # this — the transport, the batch index, the encoder — assumes a frame it can
        # open, and the next build will have it.
        while frames and not (ws / frames[-1]["name"]).exists():
            frames.pop()
        if not frames:
            print(f"replay: {label} has actions logged but no frames yet — skipped")
            continue
        for i, frame in enumerate(frames):
            frame["b"] = next(
                (j for j, b in enumerate(made) if b["from"] < frame["n"] <= b["to"]), -1
            )
            frame["dlevel"] = i and frame["level"] != frames[i - 1]["level"]

        looked, derived = attribute(said)
        # The same quantity as the human backdrop, from the same definition.
        curve = thin(best_curve([r["reward"] for r in rows],
                                [not r["alive"] for r in rows])) if rows else []
        art = pictures(ws, frames, out, label, inline)
        runs.append({
            "label": label, "variant": meta["variant"], "seed": meta.get("seed", 0),
            # Which CLI, and which launch it was played out of. Both are on the run
            # rather than on the page, because a page can now hold more than one of
            # each and the whole point of it is telling them apart.
            "agent": agent, "launch": root.name,
            # And which model, because a page can hold two runs of one CLI — and the
            # name a row goes by, which is the two together with the vendor's prefix
            # dropped, then the stint its sessions played and whether a death dealt a
            # new world: `claude opus-5 · stint 3k · fresh world`. Four runs of one
            # model differ in nothing else the name could say. A run from before
            # fresh worlds existed has no flag, and replayed its one world.
            "model": (model := model_of(root, label, agent, report)),
            "arm": f"{agent} {model.removeprefix('claude-')} · stint "
                   f"{thousands(record.get('stint') or record.get('budget') or 0)} · "
                   f"{'fresh' if record.get('fresh_world') else 'same'} world",
            # Whether the sessions ran inside the filesystem fence — no longer a fact
            # about the CLI alone, since a Claude run can be launched with `--fence`.
            "fenced": fenced_of(root, label, agent, report),
            # Whether the run reached the budget it was given. A page built on a
            # timer is mostly built over an unfinished run, and a matrix that showed
            # 4,000 actions the same way it shows 30,000 would be offering a prefix
            # as a result.
            "done": int(record.get("actions_used") or 0)
            >= int(record.get("budget") or 0),
            "report": report or {}, "record": {
                k: record.get(k) for k in (
                    "actions_used", "budget", "score", "score_pct", "best_episode",
                    "best_episode_pct", "mean_episode", "mean_episode_pct",
                    "episodes_completed", "episode_scores", "lives",
                    "deaths", "max_level", "unique_cells", "max_score",
                )
            } | {
                # From the replay and not from `record["achievements"]`. A run
                # records the names its own code knew, and the pilot's were written
                # before the naming fix (91a0473), so a third of them name the wrong
                # achievement — 46 points' worth of vector read out as 77 points'
                # worth of names. The vector was always right, so replaying it under
                # the current spec is what says what actually fired.
                "achievements": sorted({a for r in rows for a in r["fired"].split() if a}),
            },
            "frames": frames, "batches": made, "looked": looked, "derived": derived,
            "handovers": handovers,
            "curve": curve,
            "notes": (ws / "notes.md").read_text(errors="replace")[:40000]
            if (ws / "notes.md").exists() else "",
            "files": sorted(p.name for p in ws.glob("*.py")),
            **art,
        })
    return runs


def build(roots: list[Path], out: Path, inline: bool, replay_world: bool,
          only: str = "") -> dict:
    runs: list[dict] = []
    for root in roots:
        runs += one_launch(root, out, inline, replay_world, only)
    # Agent first, then world. The two arms are what this page is for, and sorting by
    # world would interleave them on the day a launch holds more than one seed.
    runs.sort(key=lambda g: (g["agent"], g["model"], g["variant"], g["seed"]))
    people = humans() if replay_world else []
    if replay_world and not people:
        print(f"replay: no human runs under {HUMAN} — the chart will have no backdrop")
    return {
        "launch": [root.name for root in roots], "inline": inline,
        "replayed": replay_world, "runs": runs, "humans": people,
        "baselines": BASELINES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("launch", nargs="+",
                        help="one or more launch directories under ~/craftax-runs")
    parser.add_argument("--out", default="replay.html")
    parser.add_argument("--label", help="one workspace (default: all of them)")
    parser.add_argument(
        "--inline", action="store_true",
        help="one self-contained file; thins runs of plain movement to fit",
    )
    parser.add_argument(
        "--no-replay", action="store_true",
        help="skip the world replay, and with it the score and condition tracks",
    )
    args = parser.parse_args()

    roots = [Path(one).expanduser().resolve() for one in args.launch]
    out = Path(args.out).expanduser().resolve()
    if args.label:
        known = {
            label: root.name
            for root in roots
            for label in json.loads((root / run.RIG / run.LABELS).read_text())
        }
        if args.label not in known:
            raise SystemExit(
                f"replay: {args.label} is not a workspace of any of these launches — "
                f"try one of {', '.join(sorted(known))}"
            )
    # Filtered before building rather than after: building a run means encoding every
    # frame of it, and a launch holds one run per seed.
    bundle = build(roots, out, args.inline, not args.no_replay, args.label or "")
    if not bundle["runs"]:
        raise SystemExit("replay: nothing to show")

    body = BODY.read_text()
    for marker, text in (("/*__CSS__*/", CSS.read_text()),
                         ("/*__DATA__*/", json.dumps(bundle, separators=(",", ":")))):
        if marker not in body:
            raise SystemExit(f"replay: {BODY.name} has no {marker} to substitute")
        # `</` inside the JSON island or the stylesheet would close its element.
        body = body.replace(marker, text.replace("</", "<\\/"))
    out.write_text(body)

    size = len(body) / 1e6
    for one in bundle["runs"]:
        shown, total = len(one["shown"]), len(one["frames"])
        print(
            f"{one['label']}  {one['agent']:<7} {one['variant']}:{one['seed']}"
            f"  {total:>5} frames"
            + (f" ({shown} inlined)" if args.inline else "")
            + f"  {len(one['batches']):>4} batches"
            f"  {sum(len(b['work']) for b in one['batches']):>5} moments off the board"
            + ("" if one["done"] else
               f"  · still playing, {one['record']['actions_used']}"
               f"/{one['record']['budget']}")
        )
    print(f"\nwrote {out} ({size:.1f} MB)")
    if args.inline:
        pics = sum(one["bytes"] for one in bundle["runs"]) / 1e6
        print(f"      {pics:.1f} MB of it pictures, {size - pics * 4 / 3:.1f} MB the run itself")
    if not args.inline:
        print(f"      {out.parent / (out.stem + '_frames')}/ holds the frames — "
              f"the page needs them beside it")
    elif size > 14:
        print("replay: close to the 16 MB page limit — lower QUALITY or SCALE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
